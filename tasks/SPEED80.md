# 速度切片任务（level 80）

给 GL-E 混训用的 catchable-speed 池。不是评测尺，也不进 held-out。

| | |
| --- | --- |
| level | 80 |
| 题数 | 500 |
| tier | 7 |
| 种子 | 800041 |
| 形状 | `CATCH_2D`，约 8e7–3e8 元素，夹在 `PERF_2D` 和 `BW_2D` 之间 |
| 族 | 只有 activation / elementwise / norm。没有独立 matmul / conv |
| 生成 | `python3 taskgen/generate_tasks.py --level 80 --tier 7 --count 500 --seed 800041` |

排除了 level 60 / 83 / 84–99 的规范化哈希。`check_holdout.py` 结论 `clean`。

原文在 `tasks/speed80/`。checkout 里的 `kernelbench/KernelBench/level80` 是工作时的副本，gitignore。
