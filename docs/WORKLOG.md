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

---

# 第二阶段：拒绝采样 SFT

目标：提升模型生成 cuTile 的能力。约束是**不用外部训练语料**——cuTile 官方文档和
KernelBench 的题目定义不算，人工手写的范例代码算。所以 `golden/` 里那三个解只能留作
评测，不能进训练集，冷启动只能靠模型自己 + 验证器。

**KernelBench 200 题全部留作测试集**，训练任务全部从 torch API 程序化生成，否则第一
阶段的基线（pass@1 12.6%）就失去对比意义。

## 阶段 0 的三个闸门

### G2：验证器吞吐（先说这个，结论最干脆）

原以为 tileiras 编译会是瓶颈，实测**编译只要约 65 毫秒**，且与张量大小无关
（`verify/bench_compile.py`）。那么基准评测的约 6 样本/分钟是被什么吃掉的？是
**每个样本重开一个进程（torch import）+ 100 次性能计时**。

据此写了 `verify/fast_verify.py`：常驻 worker、不计时、正确性试验 2 次。

| | 吞吐 |
| --- | --- |
| 基准评测harness | 约 0.1 样本/秒 |
| 快速验证器（KernelBench 真实大 shape） | 2.7 样本/秒 |
| 快速验证器（合成小 shape） | **23–28 样本/秒** |

对照 400 个已知结果验证一致性：**96.3% 一致，且零假阳性**——从不把基准判失败的样本
判成通过，这是训练数据纯度上唯一要紧的方向。14 个假阴性里 12 个是 OOM（16 worker
挤 4 张卡跑 KernelBench 的多 GB 题目），2 个是三角矩阵乘的边界数值。后来把 OOM 单独
标记，并把 worker 降到 8 就归零了。

### G1：课程学习可行性（最关键）

整套方案押在"把题目缩小到模型能做对的规模"。结果**部分成立**：

| 类别 | 真实 KernelBench | tier-2 极小 shape | 降秩后（tier-1） |
| --- | --- | --- | --- |
| 归一化 | 0% | 17.6% | **39.9%** |
| 池化 | 0% | 2.8% | 5.3% |
| 卷积 | 2.8% | **0.0%** | **0.0%** |

- 归一化彻底解锁：LayerNorm 在 4D 下 0/32，降到 2D 后 **38.5%**；RMSNorm 0/24 → 41.7%。
- 池化勉强脱零。
- **卷积仍然是硬零**，110 + 153 个样本一个没过。

卷积那个零特别说明问题：`PointwiseConv` 0/24，而 1×1 卷积本质就是逐像素的矩阵乘，
模型做 2D `Matmul` 的通过率是 87.5%。所以卡住的不是算术，是 4D 布局。为此加了
**降秩阶梯**（tier 1：把算子搬到 rank 2/3 上），归一化因此解锁，但卷积连 rank-3 的
`Conv1d`、`PointwiseConv1d` 也是 0/153。

失败原因分布：34 个直接调 `F.convNd` 交差（被纯度门控拦下），其余是 rank/shape 错误
为主。**结论：卷积没有种子可自举，拒绝采样对它无能为力，需要修复循环或更细的分解，
这一轮不做。**

### G3：LoRA 训练冒烟

通过：loss 13.27 → 11.04，无 NaN，可训练 4.14%，峰值显存 67 GB。

**踩坑：网上流传的 Qwen3-Next LoRA target module 名字是错的。** 社区配置（针对
Qwen3.5/3.6）写的是 `in_proj_qkv` / `in_proj_z`，在这个模型上**一个都匹配不到**。
transformers 的 Qwen3Next 把线性注意力输入融合成了 `in_proj_qkvz`（q,k,v,z）和
`in_proj_ba`（beta,alpha）。用错名字的后果不是报错，是静默地只覆盖 12 层全注意力：

| target 配置 | Linear 参数覆盖率 |
| --- | --- |
| 社区名字（匹配不到 DeltaNet 输入） | 38.97% |
| 实测正确名字 | **84.45%** |

`train/lora_config.py` 里的 `resolve_targets()` 会逐个报告每个 target 名字匹配到多少
模块并给出覆盖率，训练脚本在任一名字匹配为 0 时直接拒绝启动——这类错误静默失败的
代价太大。

另外 `lora_dropout` 必须为 0：部分 target 是融合的 MoE 参数而非 `nn.Linear`，peft 用
`ParamWrapper` 包装它们，而它不接受非零 dropout。

## 任务生成与数据

`taskgen/` 按算子类别加权生成 KernelBench 格式题目，写进 `KernelBench/level9x/`——
`LocalKernelBenchDataset` 支持任意 level 编号，所以生成/评测/分析流水线完全复用。
难度分 T0–T5，T1 是给硬零类别准备的降秩阶梯。生成时会**实际执行一遍** `get_inputs()`
和 `forward()`，跑不通的题目直接丢弃，免得后面把题目 bug 混进模型失败率里。

采样出的样本经快速验证器筛选，通过者构成 SFT 语料：

| 任务集 | 样本 | 通过 | 通过率 |
| --- | --- | --- | --- |
| level90（tier-2 极小） | 479 | 90 | 18.8% |
| level91（tier-1 降秩） | 554 | 81 | 14.6% |
| level92（T0–T5 课程，900 题） | 2008 | 550 | 27.4% |

去重并限制每题最多 3 个解后，得到 **409 条 SFT 样本，覆盖 163 个不同任务**，
中位序列 16k token（其中约 489 个是计入 loss 的 completion token），共 6.6M token。
算子分布偏向课程解锁的那些：Matmul 57、LayerNorm1D 36、Softmax 34、RMSNorm1D 26、
L2Norm 23。

**服务端稳定性**：能在 GB200 上跑起来的 vLLM 需要两处改动——禁用 FlashInfer 的
autotune warmup（`docker/Dockerfile.vllm-gb200`，官方镜像在这里会 illegal memory
access）以及 `--enforce-eager`。即便如此引擎仍会在持续负载下偶发崩溃，所以采样用
`taskgen/generate_with_restart.sh` 包了一层：崩了就重启服务继续，
`generate_samples.py` 本身能按已存在的 kernel 文件断点续跑。

**踩坑**：容器里的采样客户端不会被宿主机的 `pkill` 杀掉。有一次以为停了，实际又跑了
两小时（意外多拿到 200 个样本），而真正的后果是它一直占着 GPU，导致后续训练 OOM。
现在采样容器统一命名并在脚本退出时 `docker rm -f`。

## 中断：第二台开发机也到期了

训练在加载权重阶段卡死：进程停在 `futex_do_wait`，GPU 显存已经占满 158 GB 但利用率
0%。诊断是 NFS 卡顿——期间还出现过一次 `No route to host`，而 safetensors 是 mmap
读取的，NFS 一停就整个挂住。对策是把权重复制到本地 NVMe（`/raid/tmp`），但 NFS 读取
只有约 27 MB/s，复制到 117/159 GB 时开发机的 Slurm 分配到期，连接断开。

**当前状态**（全部产物都在共享目录，没丢）：

| | 状态 |
| --- | --- |
| 任务生成器 / 快速验证器 / 训练脚本 | 完成并验证 |
| 三个闸门 G1/G2/G3 | 全部通过（G1 部分通过：卷积除外） |
| 合成任务与采样样本 | 3041 个样本，721 个通过 |
| SFT 数据集 | 409 条，`runs/sft_cutile.jsonl` |
| LoRA 训练 | **未完成**（卡在加载权重） |
| 合并与重新评测 | 未开始 |

换机器后要做的：把权重放本地盘再训练（1 epoch）、合并、在 held-out 的 200 题上重跑
评测并与 pass@1 12.6% / pass@8 29.5% 对比。

## 恢复后：训练、合并、重新评测

**权重必须放本地盘。** 从 NFS 加载会卡死在 `futex_do_wait`（safetensors 是 mmap 读，
NFS 一停就整个挂住）。放到 `/raid/tmp` 后加载只要 **39 秒**。复制时有个坑：中断过的
拷贝会留下**大小对不上的截断分片**，`rsync` 对这种文件做增量校验极慢（15 分钟才动
1 GB）。直接按文件大小比对找出两个截断分片重拷，2 分钟搞定。

**训练中的非有限 loss。** 第一次跑在第 7 个 micro-batch 就因 loss 非有限退出——注意
此时还没做过任何 optimizer step，权重是原始的，所以**不是学习率问题**。序列长度和
label 数都正常，判断是 bf16 在 DeltaNet 长序列上的偶发数值问题。改成跳过并计数（超过
5% 才中止），最终 371 个 micro-batch 里跳了 13 个（3.5%），训练正常收敛。

最终：52 步，loss 0.19 → 0.126，峰值显存 90 GB，约 29 分钟。适配器 27.5 GB
（r=32 覆盖 84% 的 Linear 参数，含融合的 MoE 专家）。合并后 159.4 GB。

**评测口径**：微调后采样 k=4（基线是 k=8），对比时把基线降采样到 4，保证可比。
`train/compare_runs.py` 一开始有两个会误导结论的 bug，都修了：对两侧套用同一个 n
（导致 pass@8 显示 100%），以及直接比较错误的**绝对条数**而两次运行样本数不同
（1600 vs 800，看起来"减半"其实是分母变了）。现在按比例比，并自动对齐 k。

## 结果

完整报告见 [../results/REPORT_SFT.md](../results/REPORT_SFT.md)。要点：

- 总体 pass@1 12.9% → **13.9%**（+1.0pp），pass@4 基本持平
- 「完全用 cuTile 实现」80.2% → **82.5%**，退回 PyTorch 的频率下降
- **归一化 0% → 12.5%**，10 道纯归一化题里 4 道从 0/8 变成能解
- 卷积（零训练数据）几乎没动，池化（11 条数据）仍是 0%
- rank_mismatch −3.0pp、array_used_as_tensor −1.4pp（这两类规则学到了）
- **grid_rank_exceeded +2.1pp**——训练数据几乎全是低维任务，模型对 4D 的错误做法
  反而被强化了。这是数据分布的直接后果，下一轮必须显式覆盖高维 idiom。

一句话：**方法成立，规模不够，且能力精确地跟着数据分布走**——119 条数据的归一化涨
12.5pp，11 条的池化纹丝不动，0 条的卷积没动。

> **后续更正**：这一段里"卷积 0 条数据"的成因判断是错的。不是模型造不出卷积样本，
> 是快速验证器有个随机种子 bug 把所有正确的卷积样本都判成了失败。详见下一节。

# 第三阶段：多轮编译反馈修复循环

完整报告见 [../results/REPORT_REPAIR.md](../results/REPORT_REPAIR.md)。

## 新环境

换到 `gb200-nvl4-ts2-104`，镜像全部重建。

执行镜像基座换成 `nvcr.io/nvidia/pytorch:26.07-py3`。cuda-tile 仍是 1.4.0，smoke 测试
里 fp32 `ct.mma` 正常，历史数字可比。但有个变化要记一笔：**torch 2.13 里
`allow_tf32` 的语义变了**——标志仍然默认读作 `True`，可开关它已经不再影响 fp32 matmul
的结果（开和关的误差都是 1.144e-4）。`enforce_reference_precision()` 照旧显式强制，
所以判据没变。

**vLLM nightly 让之前所有的 workaround 都不需要了。** `vllm/vllm-openai:nightly-aarch64`
（0.26.1rc1）在 GB200 上直接起得来：FlashInfer autotune 的非法访存已经修了，
`--enforce-eager` 也可以去掉。代价是启动要 ~20 分钟（torch.compile + CUDA graph 捕获），
换来的是并发 128 下 **15650 tok/s，比之前快约 60 倍**。`docker/Dockerfile.vllm-gb200`
连同那个 autotune 补丁一起删掉了。采样吞吐从此不再是瓶颈。

## 验证器要能被循环调用，也要扛得住崩溃

把验证逻辑从 `fast_verify.py` 抽到 `verify/worker.py`，批处理 CLI 和修复循环共用一份
判据——两份实现迟早会漂移。抽完在 probe_l90 上重跑，455 个共有样本**逐个结果完全一致**。

顺带发现旧的 probe_l90 结果文件有 479 行却只有 455 个不重复的 key：24 个样本验了两遍、
另外 24 个根本没验。修正后是 100/479（20.9%），此前记的是 90/479。类别结论没变。

跑起来之后撞上两个问题：

1. **生成的 kernel 能把 worker 直接搞死**。CUDA context 里一条非法指令下去，连
   `torch.cuda.empty_cache()` 都会抛 `AcceleratorError`，worker 进程当场退出——它手上
   那个任务的结果永远不会回来，主进程在 `result_q.get()` 上死等。整个批次就这么挂住了，
   第一次跑 50 分钟连第一轮都没跑完。改成：检测停顿、重启死掉的 worker、把丢失的任务
   重投一次，实在拿不回来的标成 `worker_crash`。
2. **验证器和 vLLM 抢显存**。vLLM 吃掉 90% 显存，验证 worker 一起 OOM。把 vLLM 降到
   0.55，每张卡留出 ~82 GB。同时 OOM 和 worker 崩溃都算"不确定"而**不作为报错喂回给
   模型**——告诉模型"你显存不够"是假反馈，会污染整个实验。

## 决定性发现：随机种子 bug

修复循环第一次跑完，卷积仍然是 0/480。去翻那 853 个"编译通过但数值不对"的卷积样本：
中位误差 1.78，**没有一个进到 1e-2 以内**。误差大到不像算错，像在跟一个无关的东西比。

翻开一个 tier-2 的 Conv2d 生成结果，`ModelNew.__init__` 里自己 `torch.randn` 建权重，
而参考模型用的是 `nn.Conv2d` 自己的权重。再回头看我的 worker：

```python
set_seed(0)
ref_model = Model(*init_inputs).cuda()      # 建 nn.Conv2d 时消耗了 RNG
new_model = ModelNew(*init_inputs).cuda()   # 于是拿到另一组随机权重
```

KernelBench 自己的 `eval_kernel_against_ref` 是在**每次构造模型前**都重设种子
（eval.py 503 行、566 行）。少的这一次 `set_seed` 意味着：**只要任务里的模块自带可
学习参数，数值就永远对不上**。卷积必然有 weight，所以必然全灭。

**第二阶段"卷积冷启动"的结论因此是错的**——不是模型造不出卷积，是验证器认不出。而拒绝
采样只把验证器认可的样本喂给训练，所以 SFT 数据里卷积恰好 0 条，冷启动是自己造成的。

修掉之后：probe_l90 从 100/479 变成 111/479；已经通过的 313 个 kernel 仍然全部通过。

**用 KernelBench 自己的 evaluator 交叉验证：卷积 16/16 一致，混合 40 个样本 40/40 一致。**
`verify/cross_check.py` 就是为此写的——快速验证器是为速度重写的协议，出过一次这样的偏差
之后，它给的数字不跟官方 harness 对齐就不值得信。

## 结果

修好验证器重跑，250 题 × k=4 = 1000 个候选，最多 3 轮修复，23.9 分钟：

| 轮次 | 新增 | 累计通过率 |
| --- | ---: | ---: |
| 初始 | 236 | 23.6% |
| +修复 1 | 100 | 33.6% |
| +修复 2 | 56 | 39.2% |
| +修复 3 | 35 | **42.7%** |

**卷积 8.8% → 25.6%**（120 道题里 81 道拿到正样本，tier 1/2/3/5 都有），归一化
19.0% → 48.4%，池化 1.2% → 21.2%。每个类别都涨，原本最弱的两类涨得最多。

但要看清收益来自哪里：1970 次修复尝试里，45.1% 是**又报同一个错**，38.5% 换了个别的
错，真修好的只有 9.7%。修复循环赢在"多给几次带反馈的采样机会"，不是模型很会 debug。

尤其反直觉的是，规则性最强的 `grid_rank_exceeded`（43 次）和 `array_used_as_tensor`
（11 次）**修复成功率都是 0**。报错已经写得很清楚了，模型读得懂却改不对——它缺的不是
提示，是对应的正确写法。这正是 SFT 要补的。

产出：**427 个正样本 kernel，覆盖 183 道题**，其中 191 个来自修复轮；1000 条完整轨迹
留作后续 agentic SFT 的素材。

## 下一步

冷启动破了，下一轮直接做 SFT。但数据构成会和第二阶段完全不同——那一轮 409 条里卷积
0 条，这一轮光合成任务就有 123 条卷积正样本。

两件必须做的事：

1. 训练数据要**显式覆盖高维 idiom**。`grid_rank_exceeded` 修复率为 0，而第二阶段 SFT
   后这类错误还涨了 2.1pp，两处证据都指向同一个缺口。
2. **第二阶段所有基于快速验证器的通过率都要重算**。probe_l90 已经从 20.9% 变成 23.2%，
   凡是自带参数的任务类别（卷积、带 affine 的归一化）此前都被系统性低估了。

# 第四阶段：第二轮 SFT

要回答的问题是：新一轮 SFT 该从基座重训，还是在第二阶段的 adapter 上续训。两个都跑，
用结果说话。

## 旧数据里的卷积一直都在

先用修好的验证器把三个旧 run 重验一遍。卷积从 0 恢复到 38 条（probe_l90 +11、
probe_l91 +15、synth_l92 +12），总正样本 721 → 797。**那些卷积样本一直躺在硬盘上，
只是被误判了。**

加上修复循环的 427 条，可用池子 1224 条。按类别配额压掉已到天花板的矩阵乘（231→100）
和逐元素（131→60），得到两个数据集：

- **A（从基座）**：642 条 / 381 道题——卷积 157（24.5%）、归一化 249、池化 36
- **B（续训）**：374 条 / 183 道题，只用修复循环的新数据

对照第二阶段的 409 条：卷积 0、池化 11。

## 训练：一连串环境坑

**别把训练镜像也升到 26.07。** 我为了"统一"把它从 26.06 改成 26.07，结果 ~16k 序列有
三分之一 loss 非有限（26.06 上是 3.5%）。transformers/peft/torch 版本两边一致，差别只
在 torch 构建。改回 26.06。

**真正的元凶是 loss 计算，不是检查点。** 逐项隔离后发现：eval 模式全部有限，训练模式
开梯度检查点就 NaN、关掉就正常。但关掉检查点后激活要 180 GB，跑到半个 epoch 就 OOM，
调 `expandable_segments`、`foreach=False`、降 `max_len` 都只是把 OOM 往后推
（micro-batch 47 → 311 → 348）。

出路是**只对带标签的位置过 LM head**。prompt 是 14k token 的固定文档、只有 completion
计 loss，而 `labels=` 会让模型对全部 16k 个位置算 logits——152k 词表下就是 5 GB 的张量，
交叉熵升 fp32 再翻倍。改成先按 label 掩码选位置再过 LM head，张量小两个数量级。
`train/test_sparse_loss.py` 验证过与原路径数值一致（最大差 6e-08，PEFT 包装下同样）。

换上稀疏 loss 之后，**检查点可以开了**：峰值 87 GB，训练跑完。

## 丢弃的样本没有偏向某一类

开检查点仍会丢一部分 micro-batch（A 是 30.5%），所以训练结束会打印按类别的丢弃率——
不能让它悄悄重建我们正要填的覆盖缺口：

| 类别 | A 丢弃率 | B 丢弃率 |
| --- | ---: | ---: |
| 归一化 | 34.5% | 14.2% |
| 矩阵乘 | 39.0% | 18.0% |
| **卷积** | **27.4%** | 21.8% |
| 池化 | 25.0% | 29.4% |

卷积低于平均，丢弃是分散的，这个取舍站得住。

## 结果：续训胜出，而我原来的判断是错的

评测中途机器到期过一次，恢复后续跑完成（`eval_results.json` 按 (problem_id, sample_id)
跳过已完成的，直接重跑 `run_eval.sh` 就能接上）。200 题 k=4 全集：

| | pass@1 | pass@4 | 解出题数 | 完全用 cuTile |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 12.9% | 23.5% | 47/200 | 80.2% |
| A（从基座重训） | **14.2%** | 24.0% | 48/200 | 78.2% |
| **B（在 adapter 上续训）** | 14.0% | **26.0%** | **52/200** | **81.4%** |

**我事先推荐从基座重训，数据否定了这个判断。** 两者 pass@1 打平（14.2 vs 14.0，噪声
范围内），但 B 在真正重要的地方全面领先：多解出 4 道题（52 vs 48）、纯度更高
（81.4% vs 78.2%——A 反而比基线的 80.2% 退步了）。

我当时的三条理由里，"第二阶段模型带着已知负迁移"这条站不住：那点负迁移被新数据覆盖了，
而重训丢掉的东西比想象的多。剩下两条（数据全包含、可比性干净）成立但不足以定胜负。

### 卷积：整条主线的落点

| 类别 | 题数 | 基线 | A | B |
| --- | ---: | ---: | ---: | ---: |
| **卷积** | 98 | 1.8%，解出 5 | 2.3%，解出 8 | **4.1%，解出 14** |
| 激活 | 29 | 37.1% | **44.0%** | 39.7% |
| 归一化 | 24 | 2.1% | **4.2%** | 2.1% |
| 池化 | 10 | 0.0% | **2.5%** | 0.0% |
| 损失函数 | 6 | 16.7% | 4.2% | 4.2% |

**卷积占 200 题里的 98 道，解出题数从 5 涨到 14。** 这是整条主线——第二阶段判定"冷启动
无解"、第三阶段发现是验证器 bug、这一轮把恢复出来的数据喂回去——的最终落点。

两个都退步的地方是**损失函数（16.7% → 4.2%）**：6 道题、零训练数据，被挤掉了。这与
第二阶段"能力精确跟着数据分布走"的结论一致，只是这次是负向的。

A 的强项集中在归一化、池化、激活——都是它数据里比 B 多的类别（B 只喂了修复循环的 374
条）。所以两者的差异基本可以用数据构成解释，而不是"从哪起步"本身。

### grid_rank_exceeded 终于降下来了

第二阶段这类错误**涨了 2.1pp**（训练任务全是低维的，4D 的错误做法被强化）；第三阶段
发现修复循环对它的修复率是 **0/43**，结论是"模型读得懂报错却改不对，缺的是正确写法，
这该由 SFT 补"。这一轮补上了：

| 错误类别 | 基线 | 第二阶段 | A | B |
| --- | ---: | ---: | ---: | ---: |
| **grid_rank_exceeded** | 9.8% | 11.9% | **5.0%** | **4.5%** |
| rank_mismatch | 14.1% | 11.1% | 12.6% | 14.4% |
| array_used_as_tensor | 3.5% | 2.1% | 2.5% | 2.9% |

**几乎腰斩。** 原因很直接：这一轮的训练数据里有 157 条卷积，全是 NCHW 四维张量，
逼着模型学会"把 N 和 C 折叠进 3 维 grid"这个 idiom——正是第一阶段就指出、文档里却没有
例子的那个。

代价出现在 `timeout`（B +2.8pp）和 `host_shape_error`（+2.0pp 左右）：卷积 kernel 更
复杂，写错的方式也更多。

（这一段用 `train/compare_runs.py` 出，它读 `analyze_cutile_run.py` 的分类结果；
`compare_partial.py` 不做错误分类，两者是互补的。）

# 第五阶段：让"快"成为训练目标（评测未完成）

## 起因：一直在优化错的东西

把"不慢于 torch"加进判据，用已有数据重算（零 GPU，`speedup` 本来就在每条 per-sample
记录里）：

| | pass@1 | fast_1.0 |
| --- | ---: | ---: |
| 基线 | 12.9% | 5/200 |
| 第二阶段 | 13.9% | 8/200 |
| 第四阶段 A | **14.2%** | **4/200** |
| 第四阶段 B | 14.0% | 9/200 |

**A 正确率最高，速度却比基线还差。** 只优化正确性不仅不带来性能，还会把性能交易掉。
判据改成双指标：正确性口径不动（历史可比），fast_1.0 与之并列为头条。

## 大形状融合任务

旧任务在性能上完全无用：tier 2 的形状是 `(2, 4, 16, 16)`，2048 个元素，纯 launch
延迟主导。新建 tier 6 专门做速度课程，形状全部 ≥16M 元素，六个融合 builder：长逐元素
链、softmax 链、norm+residual、matmul+bias、matmul+residual、conv+bias。

`taskgen/audit_timing.py` 用来确认任务够大——200 道题参考耗时中位 0.385 ms、最小
0.0591 ms，没有一道低于 0.05 ms 的下限。**这一步不能省**：形状不够大的话加速比是噪声，
训练信号也是噪声。

## 结果：模型能赢在哪里，一目了然

800 个候选、最多 3 轮修复，470 个正确（58.8%，远高于卷积任务集的 42.7%）。计时之后：

| 融合模式 | n | 中位 | 最大 | 快过 torch |
| --- | ---: | ---: | ---: | ---: |
| LongChain8 | 26 | **2.748x** | 3.98x | **100%** |
| LongChain6 | 35 | 2.478x | 3.26x | 97% |
| LongChain4 | 43 | 1.798x | 2.18x | 98% |
| NormResidual | 49 | 0.629x | 3.66x | 27% |
| SoftmaxChain | 56 | 0.281x | 1.36x | 4% |
| ConvBias | 33 | 0.160x | 0.42x | **0%** |
| MatmulBias | 137 | 0.070x | 0.09x | **0%** |
| MatmulResidual | 91 | 0.031x | 0.08x | **0%** |

两个结论都很硬：

1. **融合收益随链长单调上升**：4 个算子 1.80x、6 个 2.48x、8 个 2.75x。完全符合"省下的
   中间结果越多、赢得越多"，说明模型是真的在融合而不是碰巧。
2. **凡是碰 matmul 或 conv 的，一次都没赢过**。那些走 cuBLAS/cuDNN，模型手写的 GEMM 慢
   10–30 倍。这不是数据量问题，是几十年手工调优的汇编。

聚合中位数是 0.082x，看上去像彻底失败；底下却藏着 98% 胜率的 3.98x。**只看总数会把两
个相反的事实一起抹掉**，`verify/speed_report.py` 就是为此写的。

## 训练集：103 条"证明比 torch 快"的样本

不能拿 470 条全喂——那等于教它写慢 kernel。按 `--min-speedup 1.0` 筛出 103 条（中位
2.21x），但 85% 是逐元素，单训会把别的能力挤掉。所以配 246 条正确性数据压舱，合成
349 条：逐元素 31%、卷积 23%、归一化 22%、池化 10%、矩阵乘 9%、归约 6%。

`build_sft_dataset.py` 也从"按 sample_id 先后取"改成**在正确解里取最快的几条**——同一
道题的多个正确解之间，快慢差异就是最直接的速度信号。

## 验证器：两段式计时

正确性筛查要把 GPU 超卖，计时要独占 GPU，两者矛盾。做成两段：并行筛（8 worker / 4 卡、
28 候选/秒）→ 存活者独占计时（1 worker / 卡）。只有约三成能过第一段，所以贵的那段样本
量不大。`VerifierPool` 在 `measure_time` 且 worker 多于 GPU 时直接报错，防止误用。

计时复用 KernelBench 自己的 `time_execution_with_cuda_event`，参考时间就地测。
`verify/cross_check.py` 扩展成同时比对速度，实测**中位比值 0.997、最大偏差 5%**，
口径没有分叉。顺带修了它一个陷阱：原先假设目录里都是验证器接受过的 kernel，指向生成
目录就会把所有失败样本报成"不一致"；现在两边都跑、双向比对。

## 状态：训练完成，评测未做

adapter C 训练跑完了，但写盘到一半开发机 Slurm 分配到期。那个 27.5 GB 的临时文件大小与
声明的数据末尾精确吻合（504 个张量），说明字节已经写完、只差原子重命名，已抢救为
`models/lora-C-speed/adapter_model.safetensors`，附属文件从 B 复制（同一份 LoRA 配置）。

恢复后要做的：合并 adapter C → 200 题 k=4 采样 → 评测 → 用
`train/compare_partial.py --by-category` 出双指标对比。要看的是 fast_1.0 能否从 9/200
往上走，以及 Level 2 是否有单独的改善。

**预期要诚实**：训练数据里能赢的部分几乎全是逐元素融合，而 held-out 的 200 题里纯逐元素
链很少。所以这一轮更可能验证的是"融合 idiom 能否迁移"，而不是 fast_1.0 大幅上涨。

