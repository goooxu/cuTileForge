# 独立评测集（level 60 / 61）

新模型加入时用的尺子。题目不是 KernelBench Level 1–4，头条也不再报 200 题上的数。
加载和验证仍走 KernelBench 格式（`Model` / `get_inputs` / `get_init_inputs`），采样用
`overlay/scripts/run_generate.sh`，核对用 `verify/fast_verify.py`。

| 轨 | level | 题数 | 来源 | 计时 |
| --- | ---: | ---: | --- | --- |
| 正确性 | 60 | 770 | HELDOUT2 的 99 + 88 + 84，按此顺序重新编号 | 否 |
| 速度 | 61 | 250 | level 83 逐点激活 | 对 `torch.compile` |

原文仍在 `tasks/heldout2/` 和 `tasks/bw_act/level83/`。这里是变异后的冻结副本。
对照表：同目录 `manifest.json`。

生成命令（可重复，写出同一批文件）：

```bash
python3 taskgen/build_eval_suite.py
python3 taskgen/test_eval_suite.py
```

## 做了哪些变异

不重新抽题，不加新词表，不换骨干（conv / gemm / norm / pool / loss）。在已有
`forward` 上按固定规则改：

1. **图**：近邻逐点算子循环置换；逐点链对调最内两层；层数只动逐点，一次 ±1。
   速度轨每道最外包一层 `FUSION_TAILS`（深度 1→2），不加 softmax 族。
   level 88 解析不了的短嵌套只改维/超参。
2. **超参**：只替换已知默认字面量（`negative_slope` 0.01→0.02，`alpha` 1.0→1.25，
   `lambd` 0.5→0.3，HardTanh ±1→±2，`eps` 1e-5→1e-4 等）。
3. **维度**：速度轨换成 `EVAL_BW_2D`（0.8e9–1.4e9 元素，256 对齐，不含 `BW_2D`，
   不含 `(4096, 393216)`），`get_inputs` 仍在 CUDA 上分配。正确性轨只放大
   `batch_size` 和独立空间维，二维激活乘积封顶 2e7。

改完的哈希与 level 1、2、83–99、97/98 无交集。

## 冻结协议

以后每个新模型都按这个跑，否则不能横比。

- Prompt：`cutile_concepts` + overlay 里的 `TILE_SIZE = 1024`
- `k=4`，温度 `1.0`，`top_p=0.95`，`top_k=40`，`max_tokens=8192`，`check_kernel=False`
- 通过：数值对 **且** 全是 cuTile
- 计时：只在速度轨开 `--measure-time --ref-mode compile`
- 头条：正确性 pass@1 / pass@4（770）；速度 pass@1 / pass@4（250）+ 对 compile 的中位加速比
- 两个模型比速度：只用两边都解出的题做成对比较，不要拿不同解题集合上的中位数横比
- 不要和 TILE=256 时期的 M 数、也不要和 KernelBench 200 题头条混在一张表里

## 怎么跑

checkout 被 gitignore，先有 `kernelbench/`（`scripts/setup_kernelbench.sh`）。

```bash
# 装题 + 同步 TILE=1024 prompt
overlay/scripts/install_eval_suite.sh

# 一条命令：采 60、采 61、verify、打分
CUTILE_WS=... MODEL=/path/to/checkpoint ./rl/run_eval_suite.sh <tag>

# 只通脚本，不能当发表数
CUTILE_WS=... MODEL=... ./rl/run_eval_suite.sh <tag> --smoke

# 多个模型串行；已有两份 verified jsonl 的 tag 会跳过
CUTILE_WS=... rl/compare_eval_suite.sh \
    base:/path/to/Qwen3-Coder-Next \
    M:/path/to/model-M \
    Q:/path/to/model-Q
```

已经有 jsonl 时只打分：

```bash
python3 verify/eval_scorecard.py --run M:runs/M --run Q:runs/Q
```

scorecard 会找 `runs/<tag>_l60_verified.jsonl` 和 `runs/<tag>_l61_verified.jsonl`。

第一次读数（base / M / Q）在 [results/REPORT_EVAL_SUITE.md](../../results/REPORT_EVAL_SUITE.md)。
不要把那三行和 200 题头条、TILE=256 时期的 M 写进同一张表。

## 已知缺口

- **损失只有 5 道**，对损失族基本无判别力。
- 速度轨的算子族与 level 83 相同，只是题、形状、超参和层数错开。M 采过 83，
  不是干净的零样本，但是同一道题的哈希已经分开。
- HELDOUT2 的图被用作常设正确性尺子，不再承担「只开一次、查是否过拟合 200 题」。
- 测不了 cuTile API 面的泛化：词表都是训练期见过的。
