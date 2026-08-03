# cuTileForge

提升语言模型生成 **cuTile**（NVIDIA `cuda.tile` Python DSL）kernel 的能力。

cuTile 比大多数模型的训练数据都新，所以值得问的不是"模型记没记住这个 DSL"，而是
"给了文档之后能不能用起来、以及怎么让它用得更好"。

当前进度：

- [x] **评测**——给 KernelBench 加 `cutile` backend，量出 Qwen3-Coder-Next 的基线能力
- [x] **数据合成 + 拒绝采样 SFT**——程序化生成任务、编译器验证、LoRA 微调
- [x] **多轮编译反馈修复**——把编译器报错回灌给模型，合成任务通过率 23.6% → 42.7%
- [ ] 用修复循环产出的正样本做第二轮 SFT
- [ ] RLVR

## 第二阶段结果：拒绝采样 SFT

不使用任何外部训练语料——任务从 torch API 程序化生成，标注全部来自 cuTile 编译器与
PyTorch 数值比对，KernelBench 200 题全程留作测试集。

| 指标 | 基线 | 微调后 |
| --- | --- | --- |
| pass@1 | 12.9% | **13.9%** |
| 完全用 cuTile 实现 | 80.2% | **82.5%** |
| **归一化算子** | **0.0%** | **12.5%** |

**方法成立，但幅度受数据量限制，而且能力精确地跟着数据分布走**：训练集里 119 条
归一化数据，归一化就从彻底的零涨到 12.5%（10 道题里 4 道从 0/8 变成能解）；池化只有
11 条，纹丝不动；卷积一条都没有，基本没动。

副作用也很说明问题：`grid_rank_exceeded` 反而涨了 2.1pp——训练任务几乎全是低维的，
模型对 4D 张量的错误做法被强化了。下一轮的任务生成必须显式覆盖高维 idiom。

详见 [results/REPORT_SFT.md](results/REPORT_SFT.md)。

> 当时判断"卷积拿不到种子"是模型的问题，**这是错的**：快速验证器构造两个模型时只设了
> 一次随机种子，凡是自带可学习参数的任务（卷积必然有 weight）数值都永远对不上。见下节。

---

## 第三阶段结果：多轮编译反馈修复

把编译器的报错原样回灌给模型，让它在同一段对话里改自己的 kernel，最多 3 轮。
250 道合成题 × k=4：

| 轮次 | 累计通过率 |
| --- | ---: |
| 初始 | 23.6% |
| +修复 1 | 33.6% |
| +修复 2 | 39.2% |
| +修复 3 | **42.7%** |

**卷积 8.8% → 25.6%**（120 道题里 81 道拿到正样本），归一化 19.0% → 48.4%，
池化 1.2% → 21.2%。产出 427 个正样本、183 道题覆盖，供下一轮 SFT。

这一轮最重要的发现不是修复循环本身，而是**第二阶段"卷积冷启动"是我自己验证器的
bug**——KernelBench 在构造参考模型和候选模型前各重设一次随机种子，我的快速验证器只设了
一次，于是参考模型建 `nn.Conv2d` 时推进了 RNG，候选模型拿到另一组权重，**kernel 写得
再对也判失败**。修掉之后卷积单轮就有 8.8%。

修复循环的收益也要看清来源：1970 次修复尝试里 45% 是又报同一个错，真修好的只有 9.7%。
它赢在多给了几次带反馈的采样机会，不是模型很会 debug。

详见 [results/REPORT_REPAIR.md](results/REPORT_REPAIR.md)。

---

## 基线结论

Qwen3-Coder-Next（80B-A3B，BF16）在 KernelBench Level 1+2 共 200 题、每题 8 个样本、
文档喂进 context 的条件下：

| 指标 | 数值 |
| --- | --- |
| 完全用 cuTile 实现 | 81.6% |
| 模块能 import | 91.4% |
| pass@1 | **12.6%** |
| pass@8 | **29.5%** |
| 200 题中至少通过一次 | 59 题 |

**通过判据是"数值正确 **且** 完全用 cuTile 实现"。** KernelBench 自身允许保留一部分
PyTorch 算子，但那样一个只移植了一半的实现也算通过，回答不了"模型会不会写 cuTile"，
所以这里一律记为失败——有 88 个样本数值正确但因此没被计入。

两个短板性质不同。一是一次写对的概率低，失败高度集中在几个 cuTile 特有约束上：tile
与 array 的 rank 必须一致、grid 最多 3 维、Array 不是 tensor。二是**遇到难算子会退回
PyTorch**，把 conv / norm / pooling 留给库、只移植好写的部分——Level 2 尤其明显
（数值正确 13.0%，但完全移植后通过只有 5.5%）。

手写 golden 解验证过，模型 0/8 全挂的题目里抽查的三道在 cuTile 里都可解——瓶颈在模型，
不在 DSL。

**能力沿算子类别断层分布**，不是均匀地差：

| 类别 | 题数 | 通过样本占比 |
| --- | ---: | ---: |
| 激活（逐元素） | 13 | 53.8% |
| 矩阵乘 | 78 | 19.1% |
| 损失函数 | 6 | 14.6% |
| 归约/统计 | 11 | 3.4% |
| 卷积 | 76 | 2.8% |
| 归一化 | 10 | **0.0%** |
| 池化 | 6 | **0.0%** |

分界线不在算法难度，而在能不能套用「一个 block 管一个 tile」这个最简单的映射。
激活函数可以；卷积、池化、归一化要处理多维索引、跨 tile 归约和边界，模型就塌了。

逐题清单（200 道题的算子链、输入形状、通过数、最好加速比、主要失败原因）见
[docs/PROBLEMS.md](docs/PROBLEMS.md)，完整报告见 [results/REPORT.md](results/REPORT.md)，
全过程记录（含所有踩坑）见 [docs/WORKLOG.md](docs/WORKLOG.md)。

---

## 模型实际输入输出示例

### 输入长什么样

每条 prompt 约 15k token，由六段拼成（组合定义在
`overlay/src/kernelbench/prompts/prompts.toml` 的 `[custom_prompts.cutile_docs]`）：

```
1. 任务说明        "把下面这个架构里的 PyTorch 算子换成 cuTile kernel"
2. cuTile 文档     ~14k token：编程模型导读 + 96 个 op 的 API reference
3. 一个 worked example   vector add 的 PyTorch 版 → cuTile 版
4. 待优化的架构    KernelBench 的参考 Model
5. 精度要求        FP32
6. 指令            "命名为 ModelNew，只输出代码"
```

文档那 14k token 是必需的：cuTile 比模型的训练数据新，不给文档基本写不出来。
vLLM 的 prefix caching 让这段固定前缀只算一次。

### 例 1：写对了，而且比 torch 快 4.95 倍

输入的参考实现（Level 1 第 88 题，minGPT 的 GELU，8192×8192）：

```python
class Model(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
```

模型输出（节选，完整 50 行）：

```python
import cuda.tile as ct

TILE_SIZE = 256
GELU_CONST = math.sqrt(2.0 / math.pi)

@ct.kernel
def gelu_kernel(x, out):
    i = ct.bid(0)
    # Load tile with zero padding for boundary cases
    x_tile = ct.load(x, index=(i,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)

    x_cubed = x_tile * x_tile * x_tile
    inner = x_tile + 0.044715 * x_cubed
    gelu_result = 0.5 * x_tile * (1.0 + ct.tanh(GELU_CONST * inner))

    # Store result (out-of-bounds writes are silently discarded)
    ct.store(out, index=(i,), tile=gelu_result)

def cutile_gelu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    x_flat = x.view(-1)
    grid = (ct.cdiv(x_flat.numel(), TILE_SIZE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, gelu_kernel, (x_flat, out.view(-1)))
    return out
```

**5/5 正确，4.95x**。该会的都会了：tile 尺寸取 2 的幂并写成模块级常量、边界用
`PaddingMode.ZERO`、把整个 4 个算子的链融进一个 kernel、host 端 flatten 成一维。
注释里那句 "out-of-bounds writes are silently discarded" 说明它读懂了 store 的语义
而不是照抄示例。

### 例 2：栽在 grid 最多 3 维上（两个 level 合计 165 个样本）

输入的参考实现（Level 1 第 34 题 InstanceNorm，输入 112×64×512×512）：

```python
class Model(nn.Module):
    def __init__(self, num_features: int):
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_features, height, width)
        return self.inorm(x)
```

张量是 4D 的 (N, C, H, W)，模型很自然地想一个 block 管一块空间区域，于是开了个
4 维的 grid：

```python
# 模型输出（节选）
grid = (batch_size, num_features, ct.cdiv(height, TILE_H), ct.cdiv(width, TILE_W))
ct.launch(torch.cuda.current_stream(), grid, instance_norm_forward_kernel, ...)
```

```
TileTypeError: Grid dimensions must be at most 3, got length 4
```

这是最高频的单一失败模式。**但它不是 cuTile 的表达能力限制**——host 端把 N 和 C
折叠成一维就行。[golden/level1_42_maxpool2d.py](golden/level1_42_maxpool2d.py)
就是这么写的，同样 4D 输入，通过：

```python
grid = (n * c, ct.cdiv(out_h, TH), ct.cdiv(out_w, TW))
ct.launch(..., (x.view(-1), out.view(-1), ...))
```

模型没有想到这个 idiom。文档里给了 grid 上限，但没给"高维张量怎么映射"的例子。

### 例 3：把 cuTile Array 当成 torch tensor

输入的参考实现（Level 1 第 3 题 batched matmul，128×512×1024 @ 128×1024×2048）：

```python
class Model(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A: (batch_size, m, k), B: (batch_size, k, n)
        return torch.bmm(A, B)
```

模型在 kernel 内部对传进来的 Array 调了 `.view()`：

```python
# 模型输出（kernel 内部）
num_k_tiles = ct.num_tiles(A.view(batch_size, m, k), axis=1, shape=(TM, TK))
```

```
TileTypeError: No such attribute 'view' for object of type Array[float32,(?,?,?):(?,?,1)]
```

kernel 里的 Array 只支持 load/store 一类操作，不能像 tensor 那样 reshape、切片。
同类报错还有 `Arrays are not directly subscriptable. Use load() or gather() instead.`
——PyTorch 的肌肉记忆很难消除。

### 幻觉 API 的两种来源

不存在的 `ct.*` 调用总量不大，但分布很说明问题：

| 类型 | 例子 | 次数 |
| --- | --- | --- |
| SIMT 思维残留 | `ct.tid`、`ct.thread_idx`、`ct.threadIdx`、`ct.num_threads` | 45 |
| NumPy/torch 惯性 | `ct.zeros_like`、`ct.sigmoid`、`ct.mean`、`ct.var`、`ct.erf` | 40 |

第一类尤其值得注意：tile 编程模型里根本没有线程的概念，模型却还在找线程索引。

反过来看一个**没有**发生的现象：1600 个样本里 Triton 味的泄漏是 **0**——没有任何一个
样本出现 `tl.load` 或 `@triton.jit`。混进 CUDA C++（`__global__`、`threadIdx`）的
也只有 12 个，占 0.75%。说明文档喂进 context 确实生效了，模型清楚自己在写的不是
Triton 也不是 CUDA C++。

---

## 仓库结构

本仓库**不 vendor KernelBench 的代码**。改动拆成两部分：

```
overlay/     cuTile backend 新增的文件（21 个，约 3500 行，占改动的 93%）
patches/     对 KernelBench 自身文件的修改（8 个文件，约 280 行）
golden/      手写的 cuTile 参考解，用于验证题目可解性
docker/      评测容器
results/     基线评测的汇总产物（约 1.1 MB，让报告里的数字可核验）
docs/        工作记录
upstream.lock  钉死的上游 commit
```

`kernelbench/`（clone 出来的上游代码）和 `models/`、`runs/` 都不进版本库。

## 快速开始

```bash
# 1. 重建打好补丁的 KernelBench checkout
scripts/setup_kernelbench.sh

# 2. 建评测镜像（自带 cuda-tile 与 tileiras，无需单独装 CUDA Toolkit）
docker build -f docker/Dockerfile.cutile-eval -t cutile-eval:latest .

# 3. 冒烟测试：确认 cuTile 能在本机 GPU 上编译运行
cd kernelbench && scripts/in_container.sh python3 scripts/cutile_smoke.py

# 4. 单测
GPUS=none scripts/in_container.sh python3 scripts/test_cutile_checker.py
GPUS=none scripts/in_container.sh python3 scripts/test_extract_best_code.py
```

`CUTILE_WS` 指向存放 `models/` 与 `runs/` 的目录，默认是本仓库根目录。

## 复现基线评测

```bash
cd kernelbench

# 起模型（需要一个含 Qwen3Next 支持的 vLLM 镜像）
VLLM_IMAGE=<your-vllm-image> scripts/serve_qwen.sh

# 生成：200 题 × 8 样本
scripts/run_generate.sh cutile_l1 1 8 log_raw_response=True
scripts/run_generate.sh cutile_l2 2 8 log_raw_response=True

# 停掉 vLLM 腾出 GPU，量 torch 基线
scripts/in_container.sh python3 scripts/gen_baseline_gb200.py

# 评测（长任务，用 DETACH=1 脱离 ssh 会话）
DETACH=1 NAME=l1eval scripts/run_eval.sh cutile_l1 1 8
DETACH=1 NAME=l2eval scripts/run_eval.sh cutile_l2 2 8

# 分析
scripts/in_container.sh python3 scripts/analyze_cutile_run.py \
    --run-name cutile_l1 --level 1 --num-samples 8 \
    --baseline /ws/runs/baseline_gb200_torch_fp32.json
```

硬件要求：cuTile 需要 Blackwell 或 Ampere/Ada、CUDA 13.1+、driver r580+。
本次基线跑在 GB200（sm_100）上。

---

## 两个方法学要点

评测这类任务时，这两点不处理会让数字完全失真，细节见 WORKLOG。

**参考解必须跑在真 fp32 上。** NGC 容器默认 `allow_tf32=True`，torch 的 fp32 矩阵乘
相对 float64 误差 2.1e-2，而 cuTile `ct.mma` 只有 1.65e-5——不精确的是**参考解**。
KernelBench 的 fp32 容差是 1e-4，不处理的话所有算得准的 cuTile 矩阵乘反而被判错。

**光看数值正确不够，必须要求完全用 cuTile 实现。** KernelBench 允许保留 PyTorch 算子，
所以一个只调 `torch.matmul` 的 `ModelNew` 能拿到"正确 + 约 1.0x 加速"却一行 cuTile 都
没写，一个只移植了一半的实现也算完整通过。判据因此是三条同时成立：`check_cutile_impl`
（定义了 `@ct.kernel` 且真的被 `ct.launch` 派发）、`check_torch_computation_ops`
（没有残留 torch 计算算子）、`check_pytorch_wrap`（没有残留 `torch.nn` 计算层）。
后两条是 KernelBench 自带的定义，已放行 `nn.Module`、`torch.empty_like`、
`.contiguous()` 这些 launcher 必需的宿主端脚手架。

这条线比原始口径严格得多：88 个数值正确的样本因为把 conv / norm 留在 PyTorch 里而
未被计入，其中 60 个来自 Level 2。

---

## 出处与许可

`overlay/` 与 `patches/` 是针对
[ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
（MIT）的扩展，上游 commit 钉在 `upstream.lock`。`scripts/setup_kernelbench.sh` 会把
上游连同其 LICENSE 一并 clone 下来。本仓库自身代码按 [LICENSE](LICENSE) 授权。

cuTile 文档包（`overlay/src/kernelbench/prompts/cutile_api_reference.md`）由
`scripts/build_cutile_docs.py` 从已安装的 `cuda-tile` 包 introspect 生成，
以保证与实际编译所用版本一致。
