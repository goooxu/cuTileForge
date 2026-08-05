# cuTileForge

提升语言模型生成 **cuTile**（NVIDIA `cuda.tile` Python DSL）kernel 的能力。

cuTile 比大多数模型的训练数据都新，所以值得问的不是"模型记没记住这个 DSL"，而是
"给了文档之后能不能用起来、以及怎么让它用得更好"。

## 最好的结果

held-out KernelBench Level 1+2（200 题），判据是**数值正确且完全用 cuTile**，全部用官方
harness 评测：

| | 解出题数 | 其中快过 torch |
| --- | ---: | ---: |
| 基座模型，单次生成 | 47/200 | 5 |
| 微调六轮后，单次生成 | 51/200 | 11 |
| **微调后 + 编译反馈修复** | **76/200** | **17** |

**六轮训练把解出题数从 47 推到 51，一次推理时的修复循环把它推到 76。** 在这个任务上，
让模型看到自己的报错比继续微调它划算得多。代价是 3.3 倍的模型调用，而且那是另一种协议
（见[第七阶段](#第七阶段把修复循环用到基准上)）。

卷积是主力：98 道卷积题从 5 道涨到 23 道。Level 2（融合算子链，前六轮里推不动的那一半）
从 16 道涨到 28 道。

---

当前进度：

- [x] **一：基线评测**——给 KernelBench 加 `cutile` backend，量出 Qwen3-Coder-Next 的基线能力
- [x] **二：拒绝采样 SFT**——程序化生成任务、编译器验证、LoRA 微调
- [x] **三：多轮编译反馈修复**——把编译器报错回灌给模型，让它改自己的 kernel
- [x] **四：第二轮 SFT**——卷积解出题数 5 → 14，pass@4 23.5% → 26.0%
- [x] **五：让"快"成为训练目标**——判据加入性能；Level 1 pass@1 +4.0pp，但 Level 2 未受益
- [x] **六：GRPO**——跑通了，但指标没动；顺带查出前几轮一直在全量微调 MoE 专家
- [x] **七：把修复循环用到基准上**——解出题数 51 → **76/200**，项目至今最大的一次提升

## 七个阶段一览

| | 做了什么 | 手段作用在 | 主指标 |
| --- | --- | --- | --- |
| 一：基线评测 | 加 `cutile` backend，量基线 | —— | pass@1 **12.6%** |
| 二：拒绝采样 SFT | 自造数据 + LoRA 微调 | 训练时 | pass@1 12.9% → **13.9%** |
| 三：编译反馈修复 | 把报错回灌，让模型改自己的 kernel | **推理时**（不改权重） | 合成任务通过率 23.6% → **42.7%** |
| 四：第二轮 SFT | 把修复循环产出的正样本喂回训练 | 训练时 | pass@4 23.5% → **26.0%** |
| 五：让"快"成为目标 | 判据加性能，造大形状融合任务 | 训练时 | Level 1 pass@1 **+4.0pp**，Level 2 退步 |
| 六：GRPO | 组内归一化 advantage + 分档 reward | 训练时 | **没动**（全在噪声内） |
| 七：修复循环上基准 | 把编译报错回灌，最多 3 轮 | **推理时**（不改权重） | 解出题数 51 → **76/200** |

**哪些能比、哪些不能比**：第一、二、四、五、六阶段同题集（KernelBench 200）、同判据、
k 对齐到 4，可以直接比。**第三阶段不能和它们比**——它换了题集（合成题）、换了验证器，
跑的也是基座模型。它验证的是一个独立于微调的推理时技术，产出的数据才进入第四阶段。

**整条主线的落点是卷积**：98 道卷积题（占全部 200 题的一半），解出题数从基线的 5 道
涨到 **14 道**。这条线走过"第二阶段判定冷启动无解 → 第三阶段发现是验证器 bug →
第四阶段把恢复出的数据喂回去"。

**但前四轮都在优化一件不完整的事**：判据只问"对不对"，不问"快不快"。加上"不慢于
torch"重算之后，最好的模型是 200 题里 9 道，而不是 52 道——而正确率最高的那一版
（第四阶段 A）速度反而**低于基线**。第五阶段起把 fast_1.0 与正确率并列为头条指标。

---

## 第一阶段：基线评测

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
[docs/PROBLEMS.md](docs/PROBLEMS.md)，完整报告见 [results/REPORT.md](results/REPORT.md)。

---

## 第二阶段：拒绝采样 SFT

不使用任何外部训练语料——任务从 torch API 程序化生成，标注全部来自 cuTile 编译器与
PyTorch 数值比对，KernelBench 200 题全程留作测试集。409 条自产数据，LoRA 单 epoch。

**相对第一阶段的变化**（同为 KernelBench 200，基线降采样到 k=4 保证可比）：

| 指标 | 第一阶段基线 | 第二阶段 | 变化 |
| --- | ---: | ---: | ---: |
| pass@1 | 12.9% | **13.9%** | **+1.0pp** |
| pass@2 | 17.8% | 18.5% | +0.8pp |
| pass@4 | 23.5% | 23.0% | −0.5pp |
| 完全用 cuTile 实现 | 80.2% | **82.5%** | **+2.2pp** |
| 解出的题数 | 47 | 46 | −1 |
| **归一化算子** | **0.0%** | **12.5%** | **+12.5pp** |
| 卷积 | 2.3% | 3.6% | +1.3pp |
| 池化 | 0.0% | 0.0% | ±0 |

> 基线这一列是 12.9% 而不是上面的 12.6%，因为第一阶段的头条数字按 k=8 算，这里为了
> 与微调后的 k=4 对齐做了降采样，同一个模型、同一批题。

pass@1 涨而 pass@4 微跌是典型的**分布收窄**：模型更稳定地产出它已经会的解，多采几次
撞对的多样性略降。"完全用 cuTile" 涨 2.2pp 有独立意义——**退回 PyTorch 的频率下降了**，
正是第一阶段指出的两个短板之一。

**方法成立，但幅度受数据量限制，而且能力精确地跟着数据分布走**：训练集里 119 条
归一化数据，归一化就从彻底的零涨到 12.5%（10 道题里 4 道从 0/8 变成能解）；池化只有
11 条，纹丝不动；卷积一条都没有，基本没动。

副作用也很说明问题：`grid_rank_exceeded` 反而涨了 2.1pp——训练任务几乎全是低维的，
模型对 4D 张量的错误做法被强化了。下一轮的任务生成必须显式覆盖高维 idiom。

详见 [results/REPORT_SFT.md](results/REPORT_SFT.md)。

> 当时把"卷积拿不到训练数据"归因于模型不会写卷积，**这是错的**——真正的原因是验证器的
> 一个 bug，见下一节。

---

## 第三阶段：多轮编译反馈修复

把编译器的报错原样回灌给模型，让它在同一段对话里改自己的 kernel，最多 3 轮。
250 道合成题 × k=4，跑的是**基座模型**：

| 轮次 | 累计通过率 |
| --- | ---: |
| 初始 | 23.6% |
| +修复 1 | 33.6% |
| +修复 2 | 39.2% |
| +修复 3 | **42.7%** |

按类别：**卷积 8.8% → 25.6%**（120 道题里 81 道拿到正样本），归一化 19.0% → 48.4%，
池化 1.2% → 21.2%，矩阵乘 79.5% → 86.5%。每个类别都涨，原本最弱的两类涨得最多。
产出 427 个正样本、覆盖 183 道题，供下一轮 SFT。

**这一轮最重要的产出不是这些数字，而是发现第二阶段的"卷积冷启动"是我自己验证器的
bug。** KernelBench 在构造参考模型和候选模型前各重设一次随机种子，快速验证器只设了
一次，于是参考模型建 `nn.Conv2d` 时推进了 RNG，候选模型拿到另一组权重——**只要任务
自带可学习参数，kernel 写得再对也判失败**。卷积必然有 weight，所以必然全灭，而拒绝
采样只把验证器认可的样本喂给训练，SFT 数据里卷积于是恰好 0 条。修掉之后卷积单轮就有
8.8%。已用官方 harness 复核 40/40 一致（其中卷积 16/16）。

修复循环的收益也要看清来源：1970 次修复尝试里 45% 是又报同一个错，真修好的只有 9.7%。
它赢在多给了几次带反馈的采样机会，不是模型很会 debug。

**与第二阶段没有可比的数字**：这一轮既没有在 KernelBench 200 上测，也没有用微调后的
模型。它产出的 427 条正样本进入第四阶段，那里才有同口径的对比。

详见 [results/REPORT_REPAIR.md](results/REPORT_REPAIR.md)。

---

## 第四阶段：第二轮 SFT

修复循环产出的正样本，加上用修好的验证器重验旧 run 恢复出的 38 条卷积样本，凑出 1224
条可用池子（第二阶段是 409 条、卷积 0 条）。按类别配额压掉已到天花板的矩阵乘和逐元素
之后训练，并**同时跑两个起点**来回答"该从基座重训还是在上一轮 adapter 上续训"：

| | pass@1 | pass@4 | 解出题数 | 完全用 cuTile |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 12.9% | 23.5% | 47/200 | 80.2% |
| A（从基座重训） | **14.2%** | 24.0% | 48/200 | 78.2% |
| **B（在第二阶段 adapter 上续训）** | 14.0% | **26.0%** | **52/200** | **81.4%** |

**续训胜出**：pass@1 打平，但多解出 4 道题，纯度也更高（A 反而比基线退步）。

按类别看，卷积是决定性的：

| 类别 | 题数 | 基线 | A | B |
| --- | ---: | ---: | ---: | ---: |
| **卷积** | 98 | 1.8%，解出 5 | 2.3%，解出 8 | **4.1%，解出 14** |
| 激活 | 29 | 37.1% | **44.0%** | 39.7% |
| 归一化 | 24 | 2.1% | **4.2%** | 2.1% |
| 池化 | 10 | 0.0% | **2.5%** | 0.0% |
| 损失函数 | 6 | 16.7% | 4.2% | 4.2% |

两个模型都在**损失函数上退步**（16.7% → 4.2%）——6 道题、零训练数据，被别的类别挤掉了。
这仍是第二阶段那条规律，只是方向朝下：能力精确地跟着数据分布走。

A 的强项（归一化、池化、激活）恰好是它比 B 多喂的类别，所以两者差异基本能用数据构成
解释，而不是"从哪起步"本身。

### 一个跨越三轮才闭合的问题

`grid_rank_exceeded`（"grid 最多 3 维"）是第一阶段就找出的最高频单一失败模式：

| | 基线 | 第二阶段 | 第四阶段 A | 第四阶段 B |
| --- | ---: | ---: | ---: | ---: |
| grid_rank_exceeded 占样本比 | 9.8% | **11.9%**（变差） | **5.0%** | **4.5%** |

第二阶段它反而涨了——训练任务全是低维的，模型对 4D 的错误做法被强化。第三阶段发现修复
循环对这类错误的修复成功率是 **0/43**：报错写得很清楚，模型读得懂却改不对，说明它缺的
不是提示而是**正确写法**。这一轮训练数据里有 157 条卷积、全是 NCHW 四维张量，把
"N 和 C 折叠进 3 维 grid"这个 idiom 逼了出来，于是几乎腰斩。

详见 [results/sft2_comparison.txt](results/sft2_comparison.txt) 与
[results/sft2_error_classes.txt](results/sft2_error_classes.txt)。

---

## 第五阶段：让"快"成为训练目标

判据加入性能后，前四轮的真实位置是这样的：

| | pass@1 | fast_1.0（也快过 torch） |
| --- | ---: | ---: |
| 基线 | 12.9% | 5/200 |
| 第二阶段 | 13.9% | 8/200 |
| 第四阶段 A | **14.2%** | **4/200** |
| 第四阶段 B | 14.0% | 9/200 |

旧的训练任务在性能上完全无用——tier 2 的形状是 `(2, 4, 16, 16)`，2048 个元素，纯
kernel 启动延迟主导。所以新建了一档大形状融合任务（全部 ≥16M 元素），采样 800 个候选、
470 个正确，计时之后模型能赢在哪里非常清楚：

| 融合模式 | 中位加速 | 快过 torch |
| --- | ---: | ---: |
| 8 个算子的逐元素链 | **2.75x** | **100%** |
| 6 个算子 | 2.48x | 97% |
| 4 个算子 | 1.80x | 98% |
| norm + residual | 0.63x | 27% |
| softmax 链 | 0.28x | 4% |
| conv/matmul + bias | 0.03–0.16x | **0%** |

**融合收益随链长单调上升**（4→6→8 个算子对应 1.80→2.48→2.75x），说明模型是真的在融合。
而**凡是碰 matmul 或 conv 的一次都没赢过**——那些走 cuBLAS/cuDNN，几十年手工调优的汇编，
不是数据量能弥补的。

聚合中位数是 0.082x，看上去像彻底失败，底下却藏着 98% 胜率的 3.98x。只看总数会把两个
相反的事实一起抹掉。

产出 103 条**已证明比 torch 快**的训练样本（中位 2.21x），配 246 条正确性数据压舱，
从第四阶段的 B 续训。

### 结果：打中的不是瞄的地方

| | pass@1 | pass@4 | 解出题数 | fast_1.0 |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 12.9% | 23.5% | 47/200 | 5/200 |
| B（第四阶段） | 14.0% | **26.0%** | **52/200** | 9/200 |
| **C（速度课程）** | **14.5%** | 25.0% | 50/200 | **10/200** |

fast_1.0 只从 9 涨到 10。分开看两个 level 才看得出为什么：

| | Level 1（单算子，对照） | Level 2（融合链，靶心） |
| --- | --- | --- |
| B | pass@1 22.2%，解出 33 | **5.8%，解出 19** |
| C | **24.5%，解出 35** | **4.5%，解出 15** |

**C 在 Level 1 上明显变好，在 Level 2 上反而退步。** 原因是结构性的：能赢的 103 条样本
里 85% 是长逐元素链，因为凡是以 matmul/conv 为锚点的融合模型一次都没赢过（0/261）——
那要和 cuBLAS/cuDNN 比。而 Level 2 的融合链恰恰全锚在 matmul/conv 上。

**"模型能赢的融合任务"和"长得像 Level 2 的融合任务"目前不相交。** 这一轮证明了速度是
可以教的（Level 1 逐元素题 pass@1 +4.0pp），但教不到锚在库调用上的融合。

详见 [results/phase5_comparison.txt](results/phase5_comparison.txt) 与
[results/phase5_fusion_speed.txt](results/phase5_fusion_speed.txt)。

---

## 第六阶段：GRPO（跑通了，但没动指标）

RL 的动机很具体：模型 pass@4 是 25%、pass@1 只有 14.5%——它已经能解出四分之一的题，
但四次里只有一次答对。这个缺口是它自己的余量。

20 轮之后，200 题上：

| | pass@1 | pass@4 | 解出题数 | fast_1.0 |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 12.9% | 23.5% | 47/200 | 5/200 |
| B（第四阶段） | 14.0% | **26.0%** | **52/200** | 9/200 |
| C（第五阶段） | **14.5%** | 25.0% | 50/200 | 10/200 |
| **E（GRPO）** | 14.2% | 25.5% | 51/200 | **11/200** |

**全部落在噪声内。** 训练侧的信号一致：恢复时随机种子重置，导致后 10 轮抽到的题与前
10 轮完全相同（本身是 bug，已修），这形成了一次配对对照——同一批题上平均 pass 率
0.356 → 0.352，没有改善。

**约 6.7 小时 GPU 时间换来零可测量提升**，而一轮 SFT 约 35 分钟换来 +1pp。

### 副产品：查出前几轮一直在全量微调 MoE 专家

为 RL 算存档成本时才去查 adapter 为什么有 27.5 GB：

| 组 | 参数 | 性质 |
| --- | ---: | --- |
| `experts.base_layer` | 4.83B | **可训练——全量微调** |
| `experts` lora_A/B | 2.01B | LoRA |
| 注意力 + DeltaNet | 0.03B | LoRA |

`gate_proj/up_proj/down_proj` 本意只覆盖 shared expert，却同时匹配了路由专家的**融合
张量**（不是 `nn.Linear`），peft 用 `ParamWrapper` 包装时把原始权重也放开了。所以
**99.5% 的训练量在路由专家上，其中七成根本不是 LoRA**。

RL 因此冻结专家、只训注意力与 DeltaNet：可训练量 6.88B → 34M，存档 27.5 GB → 137 MB。
**而这大概就是它没动的原因**——真正推动过指标的几轮 SFT 训的是 6.88B。下次再试 RL，
第一件事是解冻专家验证这一点，而不是在 0.04% 的参数上加轮数。

两个正面观察：纯度率全程 0.92–0.97，**没有奖励作弊**；分档 reward 让 84 道恒失败的任务
仍能提供梯度（二值 reward 会把它们全丢掉），这个设计是对的。

详见 [results/phase6_grpo_comparison.txt](results/phase6_grpo_comparison.txt)，
全过程记录（含所有踩坑）见 [docs/WORKLOG.md](docs/WORKLOG.md)。

---

## 第七阶段：把修复循环用到基准上

回看六轮，效果最大的一次干预是第三阶段的修复循环（合成任务 23.6% → 42.7%）。
**但我们只把它当数据挖矿机，从没在 held-out 的 200 题上跑过**——所有基准数字都是单次生成。

model-E 起手 k=4、最多 3 轮修复，产出的 kernel 用官方 harness 重验：

| | 解出题数 | fast_1.0 |
| --- | ---: | ---: |
| 基线 | 47/200 | 5 |
| E 单次（k=4） | 51/200 | 11 |
| **E + 编译反馈** | **76/200** | **17** |

| | Level 1（单算子） | Level 2（融合链） |
| --- | ---: | ---: |
| 基线 | 31/100 | 16/100 |
| E 单次 | 36/100 | 15/100 |
| **E + 反馈** | **48/100** | **28/100** |

按类别，卷积 5 → 8 → **23**（98 道题），池化第一次脱零（0 → 3），Level 2 从 15 到 28。

**口径要说清楚**：这不是 pass@4。2635 次模型调用对单次 k=4 的 800 次，**3.3 倍算力**。
一部分增益本来也能靠多采样拿到，严格的等算力对照需要 E 在 k=13 上的单次数据，我们没有。
第三阶段在合成任务上量过这个差别（修复是以报错为条件的，多采样不是），但这 200 题上没做
等算力对照。另外反馈里含数值比对结果，那是来自参考实现的信息——在真实开发流程里正常
（你手上就有 PyTorch 版本当测试），但不等同于一次性生成。

详见 [results/phase7_feedback_comparison.txt](results/phase7_feedback_comparison.txt)。

---

## 测试集划分

| | 用途 | 状态 |
| --- | --- | --- |
| KernelBench Level 1+2（200 题） | **dev set** | 已用于六轮迭代决策，侵蚀了 |
| KernelBench Level 3（50 题） | **final test** | 只测过一次基线（pass@1 5.0%、fast_1.0 2/50），封存 |
| KernelBench Level 4（20 题） | 不可用 | transformer 的 embedding 需要整型 token id，与本项目强制 fp32 的评测协议冲突 |

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
rl/          GRPO：分档 reward、可学边界筛选、循环驱动
docker/      评测容器
results/     各轮评测的汇总产物（约 4.1 MB，让报告里的数字可核验）
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

## 复现

### 第一阶段：基线评测

```bash
cd kernelbench

# 起模型。默认用 vllm/vllm-openai:nightly-aarch64，在 GB200 上无需任何补丁；
# 换镜像用 VLLM_IMAGE=。首次启动约 20 分钟（torch.compile + CUDA graph 捕获）
scripts/serve_qwen.sh

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

### 第三阶段：修复循环

修复循环边采样边验证，要和 vLLM 共用 GPU，所以起服务时必须把显存占比降下来给验证器
留空间，否则验证 worker 会被 OOM 拖垮。

```bash
# 起服务，留出 ~82 GB/卡 给验证器
GPU_UTIL=0.55 scripts/serve_qwen.sh

# 造一批卷积为主的合成题
python3 taskgen/generate_tasks.py --level 93 --curriculum --count 250 --clean \
    --seed 303 --category-weights conv=45,norm=8,pool=8,matmul=3

# 跑循环：250 题 × k=4，最多 3 轮修复，约 24 分钟
kernelbench/scripts/in_container.sh "python3 repair/repair_loop.py \
    --level 93 --samples 4 --max-rounds 3 --out /ws/runs/repair_l93"

# 分析转化率，并拿官方 harness 复核抽样结果
kernelbench/scripts/in_container.sh "python3 repair/analyze_repair.py \
    --run /ws/runs/repair_l93 --level 93"
kernelbench/scripts/in_container.sh "python3 verify/cross_check.py \
    --kernel-dir /ws/runs/repair_l93 --level 93 -n 40"
```

### 第七阶段：修复循环上基准（当前最好的结果）

```bash
# 起微调后的模型（留出显存给验证器，修复循环边采样边验证）
GPU_UTIL=0.55 MODEL=/raid/.../model-E MOUNTS="-v /raid/tmp:/raid/tmp:ro" \
    kernelbench/scripts/serve_qwen.sh

# 每题 4 个起始样本，最多 3 轮编译反馈修复
for L in 1 2; do
  kernelbench/scripts/in_container.sh "python3 repair/repair_loop.py \
      --level $L --samples 4 --max-rounds 3 --out /ws/runs/repairE_l$L"
done

# 用官方 harness 重验（修复循环内部走的是快速验证器）
cd kernelbench && DETACH=1 scripts/run_eval.sh repairE_l1 1 4

# 对比
python3 train/compare_partial.py --level 1,2 \
    --baseline results/level1_per_sample.json,results/level2_per_sample.json \
    --run "single-shot":../runs/E_l1,../runs/E_l2 \
    --run "with feedback":../runs/repairE_l1,../runs/repairE_l2 --by-category
```

### 第五阶段：速度课程

```bash
# 大形状融合任务，并确认它们大到能测出时间
python3 taskgen/generate_tasks.py --level 94 --tier 6 --count 200 --clean --seed 606
kernelbench/scripts/in_container.sh "python3 taskgen/audit_timing.py --level 94"

# 采样后连同计时一起验证（第二段会独占 GPU）
kernelbench/scripts/in_container.sh "python3 verify/fast_verify.py \
    --kernel-dir /ws/runs/repair_l94 --level 94 --measure-time \
    --out /ws/runs/repair_l94_verified.jsonl"

# 看模型在哪些融合模式上真能赢
python3 verify/speed_report.py --verified ../runs/repair_l94_verified.jsonl --level 94

# 只把跑得比 torch 快的拿去训练
kernelbench/scripts/in_container.sh "python3 train/build_sft_dataset.py \
    --run 94:/ws/runs/repair_l94:/ws/runs/repair_l94_verified.jsonl \
    --min-speedup 1.0 --out /ws/runs/sft_speed_only.jsonl"
```

硬件要求：cuTile 需要 Blackwell 或 Ampere/Ada、CUDA 13.1+、driver r580+。
本次基线跑在 GB200（sm_100）上。

---

## 四个方法学要点

评测这类任务时，这四点不处理会让数字完全失真，细节见 WORKLOG。

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

**测速度要独占 GPU，而且任务得够大。** 正确性筛查为了吞吐会把每张卡超卖到 4 个 worker，
但在共享 GPU 上测出来的时间被邻居污染，加速比毫无意义——所以流程拆成两段：并行筛正确性，
再让存活者独占 GPU 计时。另一半同样重要：训练任务的形状必须大到 kernel 不被启动延迟
主导，否则两个实现测出来一样快，加速比是噪声。`taskgen/audit_timing.py` 就是这个闸门，
先前 tier 2 的 `(2, 4, 16, 16)` 只有 2048 个元素，完全不合格。

**为速度重写的验证器必须跟官方 harness 对表。** 拒绝采样的吞吐要求让我另写了一个只查
正确性的快速验证器，它复现了 KernelBench 的协议——但漏了一处：构造参考模型和候选模型
前要**各重设一次随机种子**，只设一次的话参考模型建 `nn.Conv2d` 时就把 RNG 推进了，
候选模型拿到另一组权重。后果是**凡是自带可学习参数的任务永远判失败**，而这不会报错，
只会让数据集悄悄缺掉一整类，再被拒绝采样原样放大到模型能力里——我们据此得出过
"模型不会写卷积"的错误结论。`verify/cross_check.py` 就是为此存在的：拿官方 evaluator
复核抽样结果，目前 40/40 一致。

---

## 出处与许可

`overlay/` 与 `patches/` 是针对
[ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
（MIT）的扩展，上游 commit 钉在 `upstream.lock`。`scripts/setup_kernelbench.sh` 会把
上游连同其 LICENSE 一并 clone 下来。本仓库自身代码按 [LICENSE](LICENSE) 授权。

cuTile 文档包（`overlay/src/kernelbench/prompts/cutile_api_reference.md`）由
`scripts/build_cutile_docs.py` 从已安装的 `cuda-tile` 包 introspect 生成，
以保证与实际编译所用版本一致。
