# 独立评测集（level 60）

新模型加入时用的尺子。题目不是 KernelBench Level 1–4，头条也不再报 200 题上的数。
加载和验证仍走 KernelBench 格式（`Model` / `get_inputs` / `get_init_inputs`），采样用
`overlay/scripts/run_generate.sh`，核对用 `verify/fast_verify.py`。

一套题，一次计时。读数拆开，不要折进同一个中位数。

| 角色 | 题数 | 形状 | 看什么 |
| --- | ---: | --- | --- |
| 延迟 | 770 | 正确性量级（二维乘积 ≤ 2e7） | 正确性 pass@k；对 `torch.compile` 的 `kernel_ms` / 加速比 |
| 吞吐 | 139 | 同一张 activation / elementwise 图的大张量副本（0.8e9–1.4e9） | 只做成对吞吐；不对副本另开头条正确性 |

139 道吞吐都来自 `(batch_size, dim)` 的 activation / elementwise。softmax 族、norm /
matmul / conv / pool / loss 不进吞吐。独立 level 61 已删除。

原文仍在 `tasks/heldout2/`。这里是变异后的冻结副本。对照表：同目录 `manifest.json`
（`role`、`graph_id`、`shape_kind`）。

形状分两类，按 `problem_id` 取模，不随机：

| | 常见 | 不规则 |
| --- | --- | --- |
| 延迟 | ×1.5 缩放，2e7 封顶 | 约 1/4：奇数 / 不对齐 / 怪长宽比；3D/4D 保持 rank |
| 吞吐 | `EVAL_BW_2D`，256 对齐 | 约 1/3：`EVAL_AWKWARD_2D`，乘积仍约 1e9，两维都 ≤ 65535 且不 256 对齐 |

不规则上崩掉或变慢是信号。任一张量维不当唯一 grid 轴去撞 65536（`567:3` 那种 IMA）。

生成命令（可重复，写出同一批文件）：

```bash
python3 taskgen/build_eval_suite.py
python3 taskgen/test_eval_suite.py
```

## 做了哪些变异

不重新抽题，不加新词表，不换骨干（conv / gemm / norm / pool / loss）。在已有
`forward` 上按固定规则改：

1. **图**：近邻逐点算子循环置换；逐点链对调最内两层；层数只动逐点，一次 ±1。
   level 88 解析不了的短嵌套只改维/超参。
2. **超参**：只替换已知默认字面量（`negative_slope` 0.01→0.02，`alpha` 1.0→1.25，
   `lambd` 0.5→0.3，HardTanh ±1→±2，`eps` 1e-5→1e-4 等）。
3. **维度**：770 道先走正确性缩放，再按 id 决定是否套不规则规则。吞吐副本只改
   `batch_size` / `dim`，图与母题相同。

改完的哈希与 level 1、2、83–99、97/98 无交集。

## 冻结协议

以后每个新模型都按这个跑，否则不能横比。

- Prompt：`cutile_concepts` + overlay 里的 `TILE_SIZE = 1024`
- `k=4`，温度 `1.0`，`top_p=0.95`，`top_k=40`，`check_kernel=False`
- 发表两张表，不能横比：
  - **表 A**：`max_tokens=32768`。Q38 开 thinking（`ENABLE_THINKING=1`，`xhigh`）。
    G4t 开 thinking（无 effort 档）。GL（Muse Glimmer）thinking 关不掉，
    `reasoning_strength=xhigh`。Next 族没有等价开关，不传 `chat_template_kwargs`
  - **表 B**：`max_tokens=8192`。Q38nt 关 thinking（`ENABLE_THINKING=0`）。
    G4（`google/gemma-4-31B-it`）关 thinking，`vllm/vllm-openai:nightly-aarch64`，
    `EXTRA_ARGS=--reasoning-parser gemma4`。v0.27.1 正式镜像起不来（`head_dim`），
    发表数仍是 nightly。Next 族用同一 8192 上限的采样（归档在 `runs/archive_eval_8192/`）
- 32768 的关 thinking 试跑归档在 `runs/archive_q38nt_32768/`，不单开第三张表
- 通过：数值对 **且** 全是 cuTile
- 计时：正确性筛完后，在新容器里 `--timing-from` 对通过的 kernel 量
  `torch.compile`。已有 `speedup` 的 key 会跳过，jsonl 每 16 条落盘。
  worker 不跨题缓存 compile 图；计时 OOM 会重建进程。同一容器里只开一个
  计时进程池（第二池会 CUDA 起不来）。`--timeout`（本套评测 180s）含
  tileiras/ptxas 编译，超时记 `timeout`、算失败。通过但计时失败的样本保留
  通过、没有 `speedup`。
- 头条顺序：延迟成对（再拆 common / awkward）→ 吞吐成对（同样拆）→ 770 张图
  pass@1 / pass@4（按族、按出处）
- 两个模型比速度：只用两边都解出的题做成对比较，阈值 1.05x
- 禁止：延迟中位 ms 和吞吐中位加速比横比；conv / matmul 进吞吐头条；只报对齐
  子集假装「也会写 remainder」
- 不要和 TILE=256 时期的 Next-M 数、也不要和 KernelBench 200 题头条混在一张表里
- 旧两轨（770 不计时 + 250 道 level 61）上的数字作废，不能和这套横比

## 表内名称

目录和脚本仍用短 tag。发表表用下面的名字：

| 表内 | tag | 是什么 |
| --- | --- | --- |
| Next | `base` | Qwen3-Coder-Next，未在本项目训练 |
| Next-M | `M` | 拒绝采样 SFT → 丢掉长文档 → 自蒸馏 → GRPO |
| Next-Q | `Q` | 在 Next-M 的通过解上第二次自蒸馏 |
| Q38 / Q38nt | `Q38` / `Q38nt` | Qwen3.8-27B，开 / 关 thinking |
| G4 / G4t | `G4` / `G4t` | Gemma-4-31B-it，关 / 开 thinking |
| GL | `GL` | Muse Glimmer 30B，未在本项目训练 |
| GL-A | `GLA` | 在 GL 上拒绝采样 SFT（`rl/run_gl_harvest.sh` 采数据） |
| GL-B | `GLB` | 在 GL-A 的通过解上自蒸馏 |
| GL-C | `GLC` | 在 GL-B 上再做一轮 SFT，补回 activation / elementwise |
| GL-D | `GLD` | 在 GL-C 上做 GRPO（可靠性，表 A） |

## 怎么跑

checkout 被 gitignore，先有 `kernelbench/`（`scripts/setup_kernelbench.sh`）。

```bash
# 装题 + 同步 TILE=1024 prompt
overlay/scripts/install_eval_suite.sh

# 一条命令：采 level 60、计时 verify、打分
CUTILE_WS=... MODEL=/path/to/checkpoint ./rl/run_eval_suite.sh <tag>

# 只通脚本，不能当发表数
CUTILE_WS=... MODEL=... ./rl/run_eval_suite.sh <tag> --smoke

# 多个模型串行；已有 l60 verified jsonl 的 tag 会跳过
CUTILE_WS=... rl/compare_eval_suite.sh \
    base:/path/to/Qwen3-Coder-Next \
    M:/path/to/model-M \
    Q:/path/to/model-Q \
    Q38:/path/to/Qwen3.8-27B \
    Q38nt:/path/to/Qwen3.8-27B \
    G4:/path/to/Gemma-4-31B-it \
    G4t:/path/to/Gemma-4-31B-it \
    GL:/path/to/Muse-Glimmer-30B
```

`Q38` 会开 thinking 并设 `max_tokens=32768`。`Q38nt` 会关 thinking 并设
`max_tokens=8192`。`G4` 关 thinking、8192、`nightly-aarch64`（v0.27.1 起不来）。
`G4t` 开 thinking、32768、同一镜像。`GL`、`GLA`、`GLB`、`GLC` 用 `muse-glimmer` 镜像、
`reasoning_strength=xhigh`、32768。其它 tag 默认 32768、不传 thinking 开关。

采训练数据是另一回事，不要照抄这里的协议：`rl/run_gl_harvest.sh` 故意**不开**
reasoning parser，并让服务器保留特殊 token，否则推理轨迹拿不到（见脚本开头）。


已经有 jsonl 时只打分：

```bash
python3 verify/eval_scorecard.py --run M:runs/M --run Q:runs/Q
```

scorecard 找 `runs/<tag>_l60_verified.jsonl`。

读数在 [results/REPORT_EVAL_SUITE.md](../../results/REPORT_EVAL_SUITE.md)。
只重验已有 kernel：`rl/reverify_eval_suite.sh TAG [TAG ...]`。
验证器把 sticky CUDA error 标成 `cuda_poison` 并重试，不当 exec 失败。

## 已知缺口

- **损失只有 5 道**，对损失族基本无判别力。
- 吞吐只覆盖逐点 2D。测不了 conv / matmul 在大张量上的表现，那是算力墙，不该
  和带宽墙混报。
- HELDOUT2 的图被用作常设尺子，不再承担「只开一次、查是否过拟合 200 题」。
- 测不了 cuTile API 面的泛化：词表都是训练期见过的。
