# 独立评测集：base / M / Q

**尺子已经换过。** 现在是一套题（level 60：770 延迟 + 139 吞吐），一次对
`torch.compile` 计时。旧两轨（770 不计时 + 250 道 level 61）上的 180 / 630 / 649
和那 250 道速度数作废，不能和这套横比。也不要和 KernelBench 200 题头条混报。

冻结协议：`cutile_concepts` + `TILE_SIZE=1024`，k=4，温度 1.0。没有训练。

## 1. 怎么跑的

| | |
| --- | --- |
| 题目 | level 60，909 道（770 张图 + 139 份吞吐副本） |
| 延迟 | 770 道，正确性量级（二维 ≤ 2e7）；约 1/4 不规则 |
| 吞吐 | 139 道 `(batch_size, dim)` activation / elementwise 副本（0.8e9–1.4e9）；约 1/3 不 256 对齐 |
| 采样 | 每题 k=4，温度 1.0，`cutile_concepts` |
| 计时 | 全部对 `torch.compile` |
| 通过 | 数值对且全是 cuTile |
| 模型 | 基座 Qwen3-Coder-Next；model M；model Q |

入口是 `rl/compare_eval_suite.sh`。协议见 [tasks/eval/EVAL.md](../tasks/eval/EVAL.md)。

延迟中位 ms 和吞吐中位加速比不要横比。不规则上崩或变慢是信号。

## 2. 读数

重采完成后填。旧两轨的正确性排序（Q 649 / M 630 / base 180）和速度打平
（Q vs M 0.997x / 240）只描述那套已作废的题，不写进下面的表。
