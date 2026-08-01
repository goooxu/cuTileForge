# cutile-forge

提升语言模型生成 **cuTile**（NVIDIA `cuda.tile` Python DSL）kernel 的能力。

cuTile 比大多数模型的训练数据都新，所以值得问的不是"模型记没记住这个 DSL"，而是
"给了文档之后能不能用起来、以及怎么让它用得更好"。

当前进度：

- [x] **评测**——给 KernelBench 加 `cutile` backend，量出 Qwen3-Coder-Next 的基线能力
- [ ] 数据合成（拒绝采样、错误→修复配对）
- [ ] 训练（SFT / RLVR，用评测管线当 verifier）

---

## 基线结论

Qwen3-Coder-Next（80B-A3B，BF16）在 KernelBench Level 1+2 共 200 题、每题 8 个样本、
文档喂进 context 的条件下：

| 指标 | 数值 |
| --- | --- |
| 样本里真的用了 cuTile | 95.3% |
| 模块能 import | 91.4% |
| pass@1（cuTile 门控） | **15.9%** |
| pass@8（cuTile 门控） | **45.0%** |
| 200 题中至少写对一次 | 90 题 |

模型确实在写真正的 cuTile，不是靠退回 PyTorch 刷分（raw 与门控口径只差约 2 个百分点），
但一次写对的概率低，失败高度集中在几个 cuTile 特有约束上：tile 与 array 的 rank 必须
一致、grid 最多 3 维、Array 不是 tensor。手写 golden 解验证过，模型 0/8 全挂的题目里
抽查的三道在 cuTile 里都可解——瓶颈在模型，不在 DSL。

完整报告见 [results/REPORT.md](results/REPORT.md)，
全过程记录（含所有踩坑）见 [docs/WORKLOG.md](docs/WORKLOG.md)。

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

**必须检查模型是否真的用了 cuTile。** KernelBench 允许保留 PyTorch 算子，所以一个只调
`torch.matmul` 的 `ModelNew` 能拿到"正确 + 约 1.0x 加速"却一行 cuTile 都没写。
`check_cutile_impl` 要求样本真的 import 了 `cuda.tile`、定义了 `@ct.kernel`、
调用了 `ct.launch` 并用到 tile 算子；报告同时给门控前后两套数字。

---

## 出处与许可

`overlay/` 与 `patches/` 是针对
[ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
（MIT）的扩展，上游 commit 钉在 `upstream.lock`。`scripts/setup_kernelbench.sh` 会把
上游连同其 LICENSE 一并 clone 下来。本仓库自身代码按 [LICENSE](LICENSE) 授权。

cuTile 文档包（`overlay/src/kernelbench/prompts/cutile_api_reference.md`）由
`scripts/build_cutile_docs.py` 从已安装的 `cuda-tile` 包 introspect 生成，
以保证与实际编译所用版本一致。
