# 工作记录：给 KernelBench 加 cuTile backend，评测 Qwen3-Coder-Next

目标：在 KernelBench 上新增 `cutile` backend，用 Level 1 + Level 2 共 200 道题，
在"cuTile 文档喂进 context"的条件下，评测 Qwen3-Coder-Next 生成 cuTile Python
kernel 的能力。

---

## 环境

**开发机** GB200 NVL4 节点

| 项 | 值 |
| --- | --- |
| GPU | 4× NVIDIA GB200，各 189 GB，compute capability 10.0 |
| 架构 | aarch64 (Grace) |
| CUDA / Driver | 13.2 / 595.84.01 |
| CPU / 内存 | 144 核 / 956 GB |

工作目录在本机与开发机映射同一路径。

**镜像**

- 执行 cuTile：`nvcr.io/nvidia/pytorch:26.06-py3`（已自带 `cuda-tile` 1.4.0，
  `tileiras` 在 `/usr/local/cuda/bin/tileiras`，不需要 `pip install cuda-tile[tileiras]`）
- 部署模型：arm64 的 vLLM 0.26.0 镜像（已注册 `Qwen3NextForCausalLM`），
  用 `VLLM_IMAGE` 传给 `scripts/serve_qwen.sh`

**NFS 注意事项**：工作目录是 NFS 挂载且 root 被 squash，容器必须以调用者的
uid/gid 运行（`in_container.sh` 里用 `$(id -u):$(id -g)`），否则连写文件都会
Permission denied。

---

## 阶段 0：环境 go/no-go

结论：**通过**。用 `scripts/cutile_smoke.py` 在 GB200 上验证：

```
torch 2.13.0a0+8145d630e8.nv26.06 | device NVIDIA GB200 | cc (10, 0)
vadd max err: 0.0
mma torch.float32      OK
mma torch.bfloat16     OK
mma torch.float16      OK
```

- 向量加法在**非整除**长度（n=10000, TILE=256）下配 `PaddingMode.ZERO` 完全正确
- `ct.mma` 对 fp32 / bf16 / fp16 三种输入都能跑通

### 关键发现：TF32 会让 torch 参考解成为"不精确的那一方"

初看 fp32 `ct.mma` 相对 `A @ B` 误差 0.024，像是 cuTile 走了 TF32。用 float64
做基准复核后发现**结论相反**（`scripts/probe_mma.py`）：

| 实现 | 相对 float64 基准的最大误差 |
| --- | --- |
| torch `A @ B`（容器默认） | 2.10e-2 |
| cuTile `ct.mma` | 1.65e-5 |
| cuTile `ct.matmul` | 1.65e-5 |

不精确的是 **torch**：NGC 容器默认把
`torch.backends.cuda.matmul.allow_tf32 = True`、
`float32_matmul_precision = "high"`，走了 TF32 tensor core。

这对评测是致命的，因为 KernelBench 的正确性判据是
`allclose(torch_reference, generated, atol=rtol=1e-4)`（fp32 容差见
`eval.py:get_tolerance_for_precision`）。参考解自己就偏了 2e-2，任何**算得准**的
cuTile matmul 反而会被判错——Level 1/2 里矩阵乘类题目会大面积假失败。

验证修复（`scripts/probe_tf32.py`）：

```
allow_tf32=True  -> |torch_ref - cutile| max 0.03182   allclose(1e-4)=False
allow_tf32=False -> |torch_ref - cutile| max 0.0001144 allclose(1e-4)=True
```

**处理**：eval 环境强制 `torch.backends.cuda.matmul.allow_tf32 = False`、
`torch.backends.cudnn.allow_tf32 = False`、
`torch.set_float32_matmul_precision("highest")`。这同时也让结果回到 PyTorch 上游
默认语义，与已发表的 KernelBench 数字口径一致。

---

## 阶段 1：仓库与依赖

- KernelBench clone 到本地（上游起点 `423217d`，已钉在 `upstream.lock`）
- 数据集自带：level1 100 题、level2 100 题、level3 50 题、level4 20 题
- `pyproject.toml` 写死 `requires-python = "==3.10.*"`，容器是 Python 3.12。
  最终没有把 kernelbench 装进镜像，而是用 `PYTHONPATH` 指向挂载目录里的源码直接跑
  ——`get_package_resource_path()` 有 repo-relative 回退，prompt 资源能正确解析，
  改代码也不用重建镜像。
- 镜像 `docker/Dockerfile.cutile-eval` 只补 `pydra-config / tomli / python-dotenv /
  openai / litellm / transformers / modal`。刻意不装 `gpu` extra
  （`nvidia-cutlass-dsl`、`tilelang`、`cupy`）——那些是别的 backend 用的，在
  aarch64 + CUDA 13 上装不干净。`modal` 虽然只跑本地评测也必须装，因为
  `eval_from_generations.py` 无条件 import 它。

---

## 阶段 2：给 KernelBench 接 cutile backend

KernelBench 的 backend 抽象是 TOML 驱动的，接入点很干净。改动清单：

| 文件 | 改动 |
| --- | --- |
| `prompts/prompts.toml` | 新增 `[backends.cutile]`；新增 `backend_doc_block` 模板；新增 `[custom_prompts.cutile_docs]` |
| `prompts/model_new_ex_add_cutile.py` | 新写 one-shot 示例（vector add） |
| `prompts/cutile_concepts.md` | 新写编程模型导读 |
| `prompts/cutile_api_reference.md` | 由 `scripts/build_cutile_docs.py` 生成 |
| `eval.py` | 三处 backend 列表加 `cutile`；接入 fp32 精度强制 |
| `prompt_constructor_toml.py` | 支持 backend 声明 `doc_reference`，把文档作为 context **值**注入 |
| `kernel_static_checker.py` | 新增 `check_cutile_impl` + 注册到 `BACKEND_IMPL_CHECK` |
| `utils.py` | 新增 `enforce_reference_precision()`；新增 `local_chat` server type；`top_k` 透传 |
| `timing.py` | 基线计时同样强制 fp32 |
| `scripts/generate_samples.py` | `supported_backends` 加 `cutile`；暴露 top_p/top_k/server 地址；修 `problem_number` 未定义的 bug |

几个值得记的点：

**文档必须作为 context 值注入，不能作为模板。** 最终 prompt 会走一次
`.format(**context)`，文档里有大量 `{...}`，当模板会直接炸或被误替换。`str.format`
不会递归处理替换进去的值，所以把文档放进 `context["backend_doc"]`、模板里只写
`{backend_doc}` 就安全了。

**`check_kernel` 必须关掉。** `generate_samples.py` 在静态检查失败时是 `assert`，
样本会被直接丢弃、根本不落盘。而"模型写了个纯 torch 的 ModelNew"恰恰是我们要统计
的现象，丢掉它会让分母失真。所以生成阶段 `check_kernel=False` 全量留存，cuTile
使用度门控放到分析阶段做。

**instruct 模型要走 chat 接口。** 原来的 `local` server type 在 prompt 是字符串时
调 `client.completions.create`，不套 chat template——那是给 base model 用的。新增
`local_chat`，把字符串包成 user message 走 `/v1/chat/completions`。同时该分支原本
把 `top_k` 丢了，改成用 `extra_body` 透传。

### 通过判据：完全用 cuTile 实现

KernelBench 允许模型只替换一部分算子，所以一个只调 `torch.matmul` 的 `ModelNew`
能拿到"正确 + 约 1.0x 加速"却一行 cuTile 都没有，一个只移植了一半的实现也算完整通过。
标准 pass@k 会因此**高估** cuTile 能力。

判据定为三条同时成立：

1. `check_cutile_impl`（新写）——import 了 `cuda.tile`、有 `@ct.kernel` 定义、有
   `ct.launch(...)` 调用、用到了真正的 tile 算子。别名（如
   `import cuda.tile as cutile`）也能识别。
2. `check_torch_computation_ops`（KernelBench 自带）——没有残留 torch 计算算子
3. `check_pytorch_wrap`（KernelBench 自带）——没有残留 `torch.nn` 计算层

后两条上游本来是 WARNING 级，这里提成硬性要求。它们已经放行了 cuTile launcher 必需的
宿主端脚手架：`nn.Module`/`nn.Parameter`、`torch.empty_like`、`.contiguous()`。

`scripts/test_cutile_checker.py` 六个用例全过，其中关键的两个负例是"纯 torch
passthrough"和"定义了 kernel 但从不 launch"。

这条线的代价不小：**88 个数值正确的样本因此不算通过**，其中 60 个来自 Level 2——
融合题里模型的典型做法就是把 elementwise 部分移植成 cuTile、把 conv/norm 留给 PyTorch。
这正是要抓的规避行为，它绕开了最需要能力的部分。

---

## 阶段 3：部署模型

`Qwen/Qwen3-Coder-Next` BF16（159 GB，40 个 shard）下到 `models/`，用了 13 分 21 秒。

vLLM 起在 4 卡 TP=4，`--max-model-len 40960`（prompt 约 15k + 输出 8k，够用且比原生
262k 省下大量 KV cache），启动约 260 秒。

**踩坑（先记这条，下面还有）：serving 容器必须以 root 跑。** 一开始用普通 uid，引擎在 `profile_run`
阶段崩溃，根因是 FlashInfer 要往 `/usr/local/.../flashinfer_cubin/cubins/` 里写
cubin，非 root 建不了目录。而 `FLASHINFER_CUBIN_DIR` 环境变量**不管用**——
`flashinfer_cubin` 包的优先级高于环境变量。那个目录 2.5 GB，chmod 进镜像层代价太大。
最后确认 root 能读通 NFS 上的模型（root squash 只挡写不挡读），于是 serving 容器直接
以 root 跑、工作目录只读挂载。注意这跟执行 cuTile 的容器相反，后者要写 runs/，必须
用调用者自己的 uid/gid。

---

## 阶段 4：生成

Level 1 首轮 800 个样本约 4 分钟跑完（32 并发，vLLM prefix caching 让 ~14k token 的
文档前缀只算一次）。

### 踩坑：`extract_first_code` 会把公式当成 kernel

首轮结果里有 22/800 个文件小于 200 字节，最小的只有 39 字节，内容长这样：

```
output = softmax(Q @ K^T / sqrt(d)) @ V
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

原因是 `extract_first_code` 取的是**字面意义上第一个** ``` 代码块，不看语言标签。
模型习惯先用一个无标签代码块写出数学公式，再给真正的实现，于是被抓走的是公式。
这属于在惩罚模型的排版习惯，而不是在测它会不会写 cuTile，必须修。

新增 `extract_best_code`，按内容挑：优先取**最后一个**定义了 `class ModelNew` 的块
（模型经常先给一版再修正），其次取最后一个带目标语言标签的块，最后退化到最长的块。
`scripts/test_extract_best_code.py` 五个用例全过，其中"公式块在前"和"改了一版取最后"
两例是老实现会做错的。

同时加了 `log_raw_response`，把模型原始回复一并落盘——抽取是有损的，不留原文的话
事后根本分不清是"样本本身差"还是"抽取抽错了"。

修完删掉首轮结果重新生成，保证全量样本口径一致。

重生成后：L1 795/800（5 个样本撞上 8192 token 输出上限、代码被截断，只有一个没闭合的
代码围栏，算失败），L2 800/800。没有再出现小于 200 字节的样本。

---

## 阶段 5：基线

`scripts/gen_baseline_gb200.py`（自己写的，上游 `generate_baseline_time.py` 的
`__main__` 是交互式的、默认 bf16、还要跑三个 level 加 torch.compile 变体）。
torch eager + 真 fp32，Level 1 计时成功 99/100（`95_CrossEntropyLoss` 返回 None），
Level 2 100/100。耗时约 8 分钟。

---

## 阶段 6：评测

### 修了 KernelBench 两个会破坏统计口径的 bug

1. **`add_to_eval_results_file` 遇到 None 直接崩。** worker 超时或挂掉时返回 None，
   代码直接取 `.compiled` 抛 AttributeError，整个 eval 进程死掉——L1 第一次跑到
   325/800 就是这么断的。改成把 None 记成一个失败样本：直接丢掉记录会让分母悄悄
   变小，把所有比率都算高。
2. **`check_if_eval_exists_local` 只看 problem_id、不看 sample_id。** 断点续跑时，
   只要某道题有过一个样本结果，剩下 7 个样本就会被当成已完成跳过。不修的话续跑
   等于静默截断。

### 中途开发机掉了

L2 跑到 746/800 时 SSH 断开，报
`Access denied by pam_slurm_adopt: you have no active jobs on this node`——
开发机的 Slurm 分配到期了。因为 `docker run` 是前台附着运行的，ssh 一断容器也跟着没了。

**教训**：长任务要用 `docker run -d` 或 `nohup` 脱离 ssh 会话跑。

数据没丢（都在共享 NFS 上）：

| | 状态 |
| --- | --- |
| Level 1 | 800/800 全部评完 |
| Level 2 | 746/800，93/100 道题的 8 个样本齐全，缺 94–100 题 |
| 基线 | 完整 |

换到另一台同规格的 GB200 NVL4 节点后重建镜像（约 2.5 分钟），
用修好的 sample 级续跑逻辑补齐——日志确认
`Start evaluation on 54 unevaluated samples`，正好是缺的那些。L2 最终 800/800。

`in_container.sh` 加了 `DETACH=1`，长任务用 `docker run -d` 脱离 ssh 会话。

---

## 阶段 7：可解性交叉验证

对模型 8 个样本全挂的题，手写 golden cuTile 解，用 `eval_kernel_against_ref` 以
**完全相同**的标准检验（`scripts/check_golden.py`）：

| 题目 | 模型 | golden | 结论 |
| --- | --- | --- | --- |
| 23_Softmax | 0/8 | 通过 5/5，10.1 ms | 可解 |
| 3_Batched_matmul | 0/8 | 通过 5/5，57.5 ms | 可解 |
| 42_MaxPool2D | 0/8 | 通过 5/5，14.5 ms | 可解 |

MaxPool2D 这道最关键：它证伪了"grid 只能 3 维所以 4D 张量做不了"——host 端把 N、C
折叠成一维就行（165 个样本栽在这个上面）。

### 顺带发现：朴素 cuTile 本来就打不过 torch

| 题目 | torch eager | 手写 cuTile | 比值 |
| --- | --- | --- | --- |
| 23_Softmax | 3.90 ms | 10.10 ms | 0.39x |
| 3_Batched_matmul | 4.11 ms | 57.50 ms | 0.07x |
| 42_MaxPool2D | 7.90 ms | 14.50 ms | 0.54x |

所以速度指标主要在测"没调优的 tile kernel vs cuBLAS/cuDNN"，不能单独当成模型的
性能能力。报告里已注明。

---

## 阶段 8：分析与报告

分析脚本 `scripts/analyze_cutile_run.py`。中途根据真实数据修正了两轮分类逻辑：

1. 一开始 `classify_error` 拿正则同时匹配错误信息**和源码**，结果任何提到 `dtype`
   的 kernel 都被归成 dtype 错误（一度显示 526 个）。改成：只有"写了别的 DSL"这种
   源码属性才扫源码，其余一律只看错误信息。
2. 错误信息分散在三个 metadata 键里——`runtime_error`（运行期）、
   `compilation_error`（import 期）、`other_error`。一开始只读第一个，导致 93 个
   失败样本的信息是空的、全落进 "other"。
3. 通过判据一开始只要求"有 cuTile kernel 被派发"，没有要求**全部**计算都在 cuTile 里。
   收紧后 Level 2 的 pass@1 从 11.1% 掉到 5.5%——差的那一半全是半移植样本。

最终报告：[../results/REPORT.md](../results/REPORT.md)

核心数字（200 题 × 8 样本，判据为"数值正确且完全用 cuTile 实现"）：
完全 cuTile 实现 81.6%，能 import 91.4%，数值正确 18.1%，
**pass@1 = 12.6%、pass@8 = 29.5%**，200 题里 59 题至少通过一次。

---

## 阶段 9：整理成 cuTileForge 仓库

评测只是第一步，仓库要装得下后续的数据合成和训练，所以没有直接把 KernelBench 的
fork 推上去，而是拆成 **overlay + patch**：

| | 文件数 | 行数 |
| --- | --- | --- |
| `overlay/` 新增文件 | 21 | 3511 |
| `patches/` 对上游文件的修改 | 8 | 282 增 / 30 删 |

实质内容 93% 是新文件，可以作为正常文件被阅读和 review；真正动上游的只有 282 行。
`scripts/setup_kernelbench.sh` 负责 clone 上游到 `upstream.lock` 钉死的 commit、
铺 overlay、打 patch。好处是仓库只有 1.2 MB、不 vendor 别人的代码、上游关系明确；
代价是跟上游同步时要手工 rebase 一次 patch。

数据只留约 1.1 MB 汇总产物（`results/`），够核验报告里每个数字。1600 份原始回复、
1595 个 kernel、完整 eval 结果共 21.5 MB 不进仓库——那是可再生的产物，而这个项目
的模式是"评测→训练→再评测"，提交第一轮等于给后面每轮都开口子。

### 两个只有实跑才暴露的 bug

1. **golden 目录的软链用了绝对路径。** checkout 会被 bind mount 进容器，容器里的
   绝对路径不同，链接直接悬空。改成相对链接。
2. **`in_container.sh` 里写死了 `/ws/cutile-eval/src`。** 重构后 checkout 改名叫
   `kernelbench/`，这个路径已经不对了。更隐蔽的是**第一次验证还"通过"了**——因为
   我把 `CUTILE_WS` 指向旧工作目录，那里恰好还存在 `cutile-eval/`，于是测的是旧
   代码而不是重建出来的。改成从脚本自身位置推导 checkout 相对 workspace 的路径。

第二个是个教训：验证脚本时，要确认它测的确实是新产物，而不是碰巧命中了旧的。

