# Qwen3-Coder-Next 生成 cuTile Python kernel 的能力评测

**结论先说**：在文档喂进 context 的条件下，Qwen3-Coder-Next 能写出**语法基本正确、
确实在用 cuTile 编程模型**的 kernel——81.6% 的样本是完全用 cuTile 实现的，91.4% 能
成功 import。但**一次就写对的比例只有 12.6%**，给 8 次机会也只有 29.5%。性能上更弱：
通过的 kernel 里只有 19/202 比 torch eager 快。

失败原因高度集中在**几个 cuTile 特有语义**上，而不是散布在通用 GPU 编程能力上。
手写 golden 解验证过：模型 0/8 全挂的题目里，抽查的三道**在 cuTile 里都是可解的**，
所以瓶颈是模型对这个 DSL 的掌握，不是 cuTile 表达不了。

---

## 1. 评测设置

| 项 | 值 |
| --- | --- |
| 模型 | `Qwen/Qwen3-Coder-Next` BF16（80B 总参 / 3B 激活） |
| 部署 | vLLM 0.26，TP=4 |
| 采样 | temperature=1.0, top_p=0.95, top_k=40（model card 推荐），max_tokens=8192 |
| 题目 | KernelBench Level 1（100 题）+ Level 2（100 题） |
| 采样数 | 每题 8 个，共 1600 个样本 |
| 条件 | 文档喂进 context（约 14k token：编程模型导读 + 96 个 op 的 API reference + 1 个 worked example） |
| 硬件 | GB200（Blackwell, sm_100），CUDA 13.2，cuda-tile 1.4.0 |
| 精度 | fp32，容差 atol=rtol=1e-4 |
| 正确性 | 5 组随机输入全部通过才算对 |

评测代码是给 KernelBench 加的一个 `cutile` backend，改动拆成
[overlay/](../overlay)（新增文件）与 [patches/](../patches)（对上游文件的修改），
完整清单见 [docs/WORKLOG.md](../docs/WORKLOG.md)。

### 通过判据：必须完全用 cuTile 实现

KernelBench 自身允许模型只替换一部分算子、其余留给 PyTorch。按它的规则，一个只调
`torch.matmul` 的 `ModelNew` 能拿到"正确 + 约 1.0x 加速"却一行 cuTile 都没写，一个
只移植了一半的实现也算完整通过。这两种都回答不了"模型会不会写 cuTile"，所以本报告
一律记为失败。

**一个样本只有同时满足下面三条才算通过**（判据用的是 KernelBench 自带的静态检查器）：

1. 定义了 `@ct.kernel` 并且真的被 `ct.launch` 派发（`check_cutile_impl`）
2. 没有残留 torch 计算算子（`check_torch_computation_ops`）
3. 没有残留 `torch.nn` 计算层（`check_pytorch_wrap`）

后两条已经放行了 cuTile launcher 必需的宿主端脚手架：`nn.Module`/`nn.Parameter`、
`torch.empty_like`、`.contiguous()` 等等。

这条线比 KernelBench 原始口径严格得多——有 **88 个样本数值正确但没被计入**，因为它们
把 conv、norm 之类留在了 PyTorch 里。

---

## 2. 主要结果

### 2.1 总体（200 题 × 8 样本 = 1600）

| 指标 | 比例 |
| --- | --- |
| 产出了代码 | 99.7% |
| 完全用 cuTile 实现 | 81.6% |
| 模块能 import | 91.4% |
| 数值正确 | 18.1% |
| **通过**（正确 + 完全 cuTile） | **12.6%** |

| pass@k | |
| --- | --- |
| pass@1 | **12.6%** |
| pass@2 | 17.5% |
| pass@4 | 23.3% |
| pass@8 | **29.5%** |

200 道题里，有 59 道至少被通过一次。

### 2.2 分 Level

| | Level 1（单算子） | Level 2（融合） |
| --- | --- | --- |
| 完全用 cuTile 实现 | 88.2% | 75.0% |
| 能 import | 90.8% | 92.0% |
| 数值正确 | 23.2% | 13.0% |
| pass@1 | 20.8%→**19.8%** | 11.1%→**5.5%** |
| pass@8 | **34.0%** | **25.0%** |
| 至少通过一次的题 | 34/100 | 25/100 |

Level 2 掉得比 Level 1 狠得多（数值正确 13.0% → 通过 5.5%，砍掉一半以上）。原因很
直接：Level 2 是融合题，模型的典型做法是把好写的 elementwise 部分移植成 cuTile、把
conv / norm / pooling 留给 PyTorch。这种"半移植"在 KernelBench 原始口径下算完整通过，
但它恰恰绕开了最难的部分。60 个 Level 2 样本属于这种情况。

### 2.3 按算子类别

| 类别 | 题数 | 至少通过一次 | 通过样本占比 |
| --- | ---: | ---: | ---: |
| 激活（逐元素） | 13 | 12/13 | 53.8% |
| 矩阵乘 | 78 | 29/78 | 19.1% |
| 损失函数 | 6 | 2/6 | 14.6% |
| 归约/统计 | 11 | 3/11 | 3.4% |
| 卷积 | 76 | 13/76 | 2.8% |
| 归一化 | 10 | 0/10 | **0.0%** |
| 池化 | 6 | 0/6 | **0.0%** |

能力是**沿算子类别断层分布**的，不是均匀地差。分界线不在算法难度，而在能不能套用
"一个 block 管一个 tile"这个最简单的映射：激活函数可以，卷积、池化、归一化要处理
多维索引、跨 tile 归约和边界，模型就塌了。归一化 0/10 尤其说明问题——softmax、
LayerNorm 在 cuTile 里完全写得出来（见第 4 节的 golden 解），模型只是不会。

逐题清单见 [../docs/PROBLEMS.md](../docs/PROBLEMS.md)。

### 2.4 速度

| | Level 1 | Level 2 |
| --- | --- | --- |
| fast_1（通过且快于 torch 的题占比） | 5.0% | 4.0% |
| fast_2 | 1.0% | 0.0% |
| 每样本加速比中位数 | 0.105x | 0.109x |
| 快于 torch 的样本 | 15/158 | 4/44 |

（Level 2 之前看起来加速比中位数接近 1.0x，是因为那些"半移植"样本里真正的重活仍由
cuDNN 在做。按完全移植的口径重算之后，两个 level 的中位数都在 0.1x 左右。）

**这一栏不要单独解读成模型的问题。** 我手写的三个 golden 解同样明显慢于 torch：

| 题目 | torch eager | 我手写的 cuTile | 比值 |
| --- | --- | --- | --- |
| 23_Softmax | 3.90 ms | 10.10 ms | 0.39x |
| 3_Batched_matmul | 4.11 ms | 57.50 ms | 0.07x |
| 42_MaxPool2D | 7.90 ms | 14.50 ms | 0.54x |

也就是说，**没调过 tile 尺寸和流水的朴素 cuTile kernel 本来就打不过
cuBLAS/cuDNN 支撑的 torch**。这个指标主要衡量的是"朴素 tile kernel vs 厂商库"，
模型在里面的贡献只是一部分。要公平评估性能能力，得单独设计一组
"torch 本身也没有快速路径"的题目。

---

## 3. 模型在 cuTile 上具体错在哪

按错误类型统计（占全部 1600 个样本的比例）：

| 错误类型 | Level 1 | Level 2 | 说明 |
| --- | --- | --- | --- |
| `rank_mismatch` | 14.4% | 15.4% | tile 的 rank 和 array 的 rank 对不上 |
| `grid_rank_exceeded` | 9.9% | 10.8% | 想开 4D/5D 的 grid，cuTile 最多 3D |
| `wrong_numerics` | 8.0% | 15.8% | 编译运行都过了，但数值不对 |
| `none_kernel_argument` | 9.6% | 1.5% | 把 `None` 当 kernel 参数传进去 |
| `undefined_name` | 6.1% | 5.4% | 引用了参考实现里的模块级变量 |
| `array_used_as_tensor` | 3.0% | 3.9% | 对 Array 调 `.view()` / `A[i]` |
| `wrong_arg_type` | 3.8% | 6.1% | 该传 array 的地方传了 tile 等 |

异常类型上 `cuda.tile._exception.TileTypeError` 占绝对多数（L1 212 个、L2 280 个）。

### 三个最典型的错误

**(1) grid 最多 3 维（两个 level 合计 165 个样本）**

报错是 `Grid dimensions must be at most 3, got length 4`。KernelBench 里大量
NCHW / NCDHW 的 4D、5D 张量，模型很自然地想"一个 block 管一个输出元素"，于是开 4D
甚至 5D 的 grid。cuTile 的 grid 上限是 3 维。

这**不是** cuTile 表达能力的限制——在 host 端把 N 和 C 合起来 reshape 一下就行。
我手写的 MaxPool2D golden 解就是这么做的（`x.view(-1)` + `grid=(n*c, h_tiles, w_tiles)`），
正确通过。模型没有想到这个 idiom。

**(2) 把 cuTile Array 当成 torch tensor**

典型报错：

```
No such attribute 'view' for object of type Array[float32,(?,?,?):(?,?,1)]
Arrays are not directly subscriptable. Use load() or gather() instead.
```

cuTile 的 Array 在 kernel 里只支持 load/store 一类操作，不能像 tensor 那样切片、
reshape。模型的 PyTorch 惯性很强。

**(3) 幻觉 API 集中在两类**

不存在的 `ct.*` 调用不多但很有代表性：

- **SIMT 思维残留**：`ct.tid`（28 次）、`ct.thread_idx`（14）、`ct.threadIdx`、
  `ct.num_threads`——还在想线程，而 tile 模型里根本没有线程的概念
- **NumPy/torch 惯性**：`ct.zeros_like`（12）、`ct.sigmoid`（6）、`ct.mean`（5）、
  `ct.var`（4）、`ct.reciprocal`、`ct.erf`、`ct.log1p`

值得注意的是，**Triton 味的泄漏是 0**——1600 个样本里没有任何一个出现 `tl.load` 或
`@triton.jit`。说明文档喂进 context 确实起了作用，模型知道自己在写的不是 Triton。
混进来的 CUDA C++（`__global__`、`threadIdx`）两个 level 合计 12 个，占 0.75%。

---

## 4. 可解性交叉验证：是模型不会，还是 cuTile 不行？

对模型 8 个样本全挂的题目，抽了三道有代表性的手写 golden cuTile 解，用**完全相同**
的评测标准检验：

| 题目 | 模型成绩 | 主要失败原因 | 手写 golden 解 |
| --- | --- | --- | --- |
| 23_Softmax | 0/8 | rank_mismatch 4 个 | **通过**（5/5），10.1 ms |
| 3_Batched_matmul | 0/8 | rank_mismatch 7 个 | **通过**（5/5），57.5 ms |
| 42_MaxPool2D | 0/8 | grid 超 3 维 | **通过**（5/5），14.5 ms |

三道全部可解。代码在 [golden/](../golden)。

结论很明确：**这些失败是模型对 cuTile 掌握不够，不是 cuTile 表达不了。** 尤其
MaxPool2D 这道直接证伪了"grid 只能 3 维所以 4D 张量做不了"的推测。

需要说明的是这只是三道抽样，不足以覆盖全部 110 道没做对的题；卷积类里可能确实
存在 cuTile 当前不好写的情况。

---

## 5. 方法学上必须记的几点

这几条会实质影响数字，换个环境重跑要注意。

**(1) TF32 会让 torch 参考解成为不精确的那一方。** NGC 容器默认
`torch.backends.cuda.matmul.allow_tf32 = True`。实测相对 float64 基准：torch
`A @ B` 误差 2.10e-2，而 cuTile `ct.mma` 只有 1.65e-5——**不精确的是 torch**。
KernelBench fp32 容差是 1e-4，不修的话所有算得准的 cuTile 矩阵乘反而被判错。
已在 eval 和基线计时里强制 `allow_tf32=False` + `float32_matmul_precision("highest")`。

**(2) `compiled` 对 cuTile 没有编译含义。** `@ct.kernel` 是懒编译的，
KernelBench 的 `compiled=True` 只表示模块 import 成功，真正的 tileiras 编译发生在
第一次 launch，也就是在正确性检查里面。所以本报告用的是"import 成功率"这个说法，
并另外按 stage 拆了失败位置。

**(3) 生成阶段必须关掉 `check_kernel`。** 它在静态检查失败时是 `assert`，样本会被
直接丢弃、不落盘。而"模型写了个纯 torch 的 ModelNew"正是要统计的现象，丢掉它会让
分母失真。全量留存、门控放到分析阶段做。

**(4) `extract_first_code` 会把公式当成 kernel。** 它取字面第一个 ``` 块、不看语言
标签。模型习惯先用无标签代码块写数学公式，于是被抓走的是公式——首轮 800 个样本里有
22 个因此变成 39 字节的垃圾。换成按内容挑（优先取最后一个含 `class ModelNew` 的块）
后归零。这属于在惩罚排版习惯而非能力，必须修。

**(5) 修了 KernelBench 两个会破坏统计口径的 bug**：worker 返回 None 时崩溃导致整个
eval 中断；断点续跑只按 problem_id 判重、会跳过同题剩余样本。详见 WORKLOG。

**(6) 5 个 Level 1 样本撞上 8192 token 输出上限**被截断（代码围栏没闭合），算作失败。
占 0.6%，不影响结论。

---

## 6. 总结

**能力画像**：Qwen3-Coder-Next 拿到文档后能建立起 cuTile 的基本心智模型——它知道要
`@ct.kernel` + `ct.launch`、知道用 `ct.load/store` 配 `PaddingMode`、知道 `ct.mma`
要配 fp32 累加器，而且**不会退化去写 Triton**（1600 个样本里 Triton 泄漏为 0）。
81.6% 的输出是完全用 cuTile 实现的。

**短板有两个，性质不同**：

一是**一次写对的概率只有 12.6%**，错误高度集中在几个 cuTile 特有的约束上——tile 与
array 的 rank 必须一致、grid 最多 3 维、Array 不是 tensor。这些都是文档里写了但模型
没有内化的规则，尤其"4D 张量要在 host 端折叠维度"这类 idiom，文档没有直接给例子，
模型就想不到。

二是**遇到难算子会退回 PyTorch**。88 个样本数值正确却没通过，因为把 conv、norm、
pooling 留在了 torch 里。这在 Level 2 尤其明显（数值正确 13.0% 但通过只有 5.5%）。
模型不是不会写 cuTile，而是在"好写的部分写 cuTile、难写的部分交给库"——这恰恰绕开了
最需要能力的地方。训练时要专门针对这种规避行为设计奖励。

这两点在按算子类别的拆解里合成了同一个形状：激活 53.8%、矩阵乘 19.1%、卷积 2.8%、
归一化和池化 0.0%。能套用"一个 block 管一个 tile"的算子它就会，需要多维索引、跨 tile
归约和边界处理的它就不会——**后续的数据合成应该集中打这三类**（卷积、归一化、池化），
它们合计 92 道题，目前只有 13 道被通过过。

**可改进的方向**（按预期收益排序）：

1. **上 agentic 修复循环**。失败里很大一部分是编译期就能发现的（rank 不匹配、grid
   超维、传了 None），报错信息也相当具体（cuTile 的报错带行号和列号）。给模型看一眼
   报错再改，pass@1 应该有明显提升。本次是单轮生成，没有用上模型主打的 agentic 能力。
2. **文档里补 idiom 而不只是 API**。当前 API reference 已经很全（96 个 op），但
   165 个样本栽在 grid 维度上，说明缺的是"4D/5D 张量怎么映射到 3D grid"这种模式。
   在导读里加 2-3 个这样的 worked example，成本极低。
3. **训练时把"完全移植"作为硬约束**。本次评测的判据（正确 + 完全 cuTile）可以直接
   当 RLVR 的 reward，避免模型学会用部分移植来骗过验证。
4. **性能要单独设计题目**。现在的速度指标主要在测"朴素 tile kernel vs 厂商库"，
   我手写的解也是 0.07x–0.54x，区分不出模型的性能优化能力。

---

## 附：产物位置

仓库内（本次评测的可核验产物）：

| 内容 | 路径 |
| --- | --- |
| cutile backend 新增文件 | `overlay/` |
| 对上游文件的修改 | `patches/0001-cutile-backend.patch` |
| 200 道题的逐题清单与成绩 | `docs/PROBLEMS.md` |
| 工作记录（含所有踩坑） | `docs/WORKLOG.md` |
| 文档 context 包 | `overlay/src/kernelbench/prompts/cutile_{concepts,api_reference}.md` |
| 手写 golden 解 | `golden/` |
| 逐样本分析结果 | `results/level{1,2}_per_sample.json` |
| 分析报表 | `results/analysis_l{1,2}.txt` |
| GB200 torch 基线 | `results/baseline_gb200_torch_fp32.json` |

仓库外（体积大、可重新生成，不进版本库）：

| 内容 | 位置 |
| --- | --- |
| 1600 份模型原始回复 + 1595 个抽取出的 kernel | 评测机 scratch 上的 `runs/cutile_l{1,2}/` |
| 每个样本的完整评测结果（含报错原文） | 同上，`eval_results.json` |
| 模型权重（159 GB） | 同上，`models/` |

重建方式见 [README](../README.md) 的"复现基线评测"一节。注意重跑需要
159 GB 权重、一台 Blackwell 机器和数小时评测时间。
