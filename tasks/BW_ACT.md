# 带宽受限逐点激活任务（level 83）

第二十阶段的速度训练集。KernelBench Level 1 激活题是 `(4096, 393216)`（约 16 亿元素、
1.82 ms），此前速度课的 `PERF_2D` 最大只有 `8192×8192`，差约 24 倍——那就是第十六阶段
训了却学不到访存合并的原因。

| | |
| --- | --- |
| level | 83 |
| 题数 | 250 |
| 种子 | 830020 |
| builder | `large_pointwise_activation`（tier 6） |
| 形状 | `BW_2D`，0.8e9–1.4e9 元素，**不含**测试形状 `(4096, 393216)` |
| 算子 | 21 个真正逐点的激活；不含 Softmax / LogSoftmax / Softmin |

原文在 `tasks/bw_act/level83/`。用的时候拷进 `kernelbench/KernelBench/level83/`。
生成时按哈希排除了 level 1,2 和 84–99。

第二十阶段用 model M 在这批题上采了 k=8（温度 1.2、concepts 档、对 `torch.compile`
计时）。速度区间对上了（中位 0.614x），组内变异不够（12% < 33%），**没有用于训练**。
见 [results/REPORT_PHASE20.md](../results/REPORT_PHASE20.md)。

独立评测集的速度轨（level 61）是这 250 道的变异副本：换了 `EVAL_BW_2D`、改了默认
超参、做了逐点算子置换，并多包一层 `FUSION_TAILS`。评测不要装本目录的原文，见
[eval/EVAL.md](eval/EVAL.md)。
