# cuTileForge

提升语言模型生成 **cuTile**（NVIDIA `cuda.tile` Python DSL）kernel 的能力。

cuTile 比大多数模型的训练数据都新，所以值得问的不是"模型记没记住这个 DSL"，而是
"给了文档之后能不能用起来、以及怎么让它用得更好"。

## 最好的结果

held-out KernelBench Level 1+2（200 题），判据是**数值正确且完全用 cuTile**，全部用官方
harness 评测，单次生成：

| | prompt 长度 | 解出题数 | 其中快过 torch |
| --- | ---: | ---: | ---: |
| 基座模型 | 14,891 token | 47/200 | 5 |
| 微调六轮后 | 14,891 token | 51/200 | 11 |
| 第八阶段（给概念导读） | 2,393 token | 84/200 | 17 |
| 第八阶段（不给任何文档） | **939 token** | 77/200 | 19 |
| 第十阶段（自蒸馏 + GRPO） | 2,393 token | 127/200 | 48 |
| 第十一阶段（+ 类别平衡的 frontier） | 2,393 token | 134/200 | 45 |
| 第十二阶段（+ 数值 reward 分级） | 2,393 token | 144/200 | 41 |
| **第十四阶段（+ 纯度分级与链尾任务）** | 2,393 token | **150/200** | 43 |

**当前最好是 150/200，pass@1 从基座的 12.9% 涨到 51.6%——首次过半。**
卷积占 200 题的一半，从基线的 5 道涨到 **75 道**。

上表"快过"一栏是对 **`torch.compile`** 量的，不是 eager。这个口径是第十三阶段才换的，
换之前所有速度数字都偏乐观：L 对 eager 有 60 道更快、中位 1.00x，对 compile 只有 41 道、
**中位 0.92x——中位的 kernel 比 `torch.compile` 慢 8%，并没有追平**。
详见[第十三阶段](#第十三阶段把速度尺换成-torchcompile)。

一路上两件事值得单独讲：第八阶段证明了**不给任何文档、prompt 短 16 倍也能解出 77 道**，
而此前捧着 55k 字符完整 API 手册的模型只解出 51 道——模型现在是会 cuTile，不是在查
cuTile。第十阶段则第一次让 RL 生效，并发现它和自蒸馏**吃的是不同的类别**。

封存的 final test（KernelBench Level 3 的 50 道整网，只开封一次，第八阶段用掉）：
**pass@1 5.0% → 20.0%**。

Level 2（融合算子链，前六轮推不动的那一半）从 16 道涨到 46 道。

放宽采样预算到 k=16（同为 concepts 档、单次生成、不给任何反馈），第八阶段的 F 解出
129/200，第十阶段的 H 解出 **145/200**。这和上表不是一个预算，不能并排比，但它给出
可靠性还差多远：**H 的 pass@1 是 28.5%，pass@16 是 72.5%**。

---

当前进度：

- [x] **一：基线评测**——给 KernelBench 加 `cutile` backend，量出 Qwen3-Coder-Next 的基线能力
- [x] **二：拒绝采样 SFT**——程序化生成任务、编译器验证、LoRA 微调
- [x] **三：多轮编译反馈修复**——把编译器报错回灌给模型，让它改自己的 kernel
- [x] **四：第二轮 SFT**——卷积解出题数 5 → 14，pass@4 23.5% → 26.0%
- [x] **五：让"快"成为训练目标**——判据加入性能；Level 1 pass@1 +4.0pp，但 Level 2 未受益
- [x] **六：GRPO**——跑通了，但指标没动；顺带查出前几轮一直在全量微调 MoE 专家
- [x] **七：把修复循环用到基准上**——解出题数 51 → 76/200（**第九阶段已撤回**）
- [x] **八：撤掉训练时的文档包**——单次解出 **84/200**，不给文档也有 77/200；final test pass@1 5.0% → **20.0%**
- [x] **九：修复循环的等预算对照**——**否定结果**：单纯重采（129/200）胜过修复循环（107/200），七阶段结论撤回
- [x] **十：把覆盖面转成可靠性**——自蒸馏与 GRPO 互补且可叠加，解出 127/200，pass@1 39.5%，中位加速 **1.00x**
- [x] **十一：frontier 的类别构成**——补上缺失的激活与损失两族并加类别配额，解出 134/200，pass@1 42.8%
- [x] **十二：给“数值错”分级**——解出 **144/200**，pass@1 **44.6%**，且策略漂移只有 1/54
- [x] **十三：把速度尺换成 `torch.compile`**——正确性不变，但“中位追平 PyTorch”的说法不成立
- [x] **十四：纯度分级 + 链尾激活任务**——解出 **150/200**，pass@1 **51.6%**，激活族 22 → 25
- [x] **十五：纯度上限提到 0.3**——**否定结果**，解出掉到 147 且靶子变大；速度判定实验通了
- [x] **十六：以速度为目标**——修好了奖励死区，但训练分布选错，kernel 逐题速度比 1.000x

## 十六个阶段一览

| | 做了什么 | 手段作用在 | 主指标 |
| --- | --- | --- | --- |
| 一：基线评测 | 加 `cutile` backend，量基线 | —— | pass@1 **12.6%** |
| 二：拒绝采样 SFT | 自造数据 + LoRA 微调 | 训练时 | pass@1 12.9% → **13.9%** |
| 三：编译反馈修复 | 把报错回灌，让模型改自己的 kernel | **推理时**（不改权重） | 合成任务通过率 23.6% → **42.7%** |
| 四：第二轮 SFT | 把修复循环产出的正样本喂回训练 | 训练时 | pass@4 23.5% → **26.0%** |
| 五：让"快"成为目标 | 判据加性能，造大形状融合任务 | 训练时 | Level 1 pass@1 **+4.0pp**，Level 2 退步 |
| 六：GRPO | 组内归一化 advantage + 分档 reward | 训练时 | **没动**（全在噪声内） |
| 七：修复循环上基准 | 把编译报错回灌，最多 3 轮 | **推理时**（不改权重） | 解出题数 51 → 76/200（**九阶段撤回**） |
| 八：撤掉文档包 | 三档 prompt + 任务多样性提 22 倍 | 训练时 | 解出题数 51 → **84/200**；不给文档 77/200 |
| 九：等预算对照 | 拿 k=16 单次重采对照修复循环 | —— | **否定结果**：重采 **129/200** > 修复 107/200 |
| 十：覆盖面转可靠性 | 自蒸馏与 GRPO 同起点对照，再叠加 | 训练时 | 解出 79 → 127/200，pass@1 39.5% |
| 十一：frontier 类别构成 | 补齐激活/损失两族 + 类别配额 | 训练时 | 解出 134/200，pass@1 42.8% |
| 十二：数值 reward 分级 | 按相对偏差给“能跑但数值错”打分 | 训练时 | 解出 **144/200**，pass@1 **44.6%** |
| 十三：换 `torch.compile` 基线 | 速度改跟 inductor 比，含 RL 的速度奖励 | 口径修正 | 中位 1.00x → **0.92x**，快过 60 → **41** |
| 十四：纯度分级 + 链尾任务 | 按剩余 torch 算子给纯度失败打分 | 训练时 | 解出 **150/200**，pass@1 **51.6%** |
| 十五：纯度上限 0.18 → 0.30 | 让不纯但算对的解排在“编译不过”之前 | 训练时 | **否定结果**：解出 150 → 147，靶子 25 → 31 |
| 十六：以速度为目标 | 补上 1.0x 以下的奖励梯度 + 稳定解出的 frontier | 训练时 | **否定结果**：逐题速度比 **1.000x**，训练分布选错 |

**哪些能比、哪些不能比**：第一、二、四、五、六、八阶段同题集（KernelBench 200）、同判据、
k 对齐到 4，可以直接比。**第三阶段不能和它们比**——它换了题集（合成题）、换了验证器，
跑的也是基座模型。它验证的是一个独立于微调的推理时技术，产出的数据才进入第四阶段。
**第七、九阶段的多轮/多样本口径也不能直接和单次 k=4 比**，这正是第九阶段要说明的事。

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

> **第九阶段补做了这个等算力对照，结论是这一节的增益不成立**：同等采样预算下，
> 不给任何反馈的单纯重采比修复循环解得更多。详见
> [第九阶段](#第九阶段修复循环的收益是假的)。

详见 [results/phase7_feedback_comparison.txt](results/phase7_feedback_comparison.txt)。

---

## 第八阶段：撤掉训练时的文档包

前七轮每轮只训得起约 500 条样本、44 个优化步。原因是机械的：

| | 中位长度 | 占训练序列 |
| --- | ---: | ---: |
| cuTile API 参考 | 55,663 字符 | **91%** |
| 要学的 completion | 1,881 字符 | **3.1%** |

**对 15k token 做完整前向反向，只为学其中 3%。** 而概念上，我们每次都把手册摊开放在它
面前，它就永远不需要记住任何东西——高手是不翻手册的。

于是分三档 prompt：完整文档（14,891 token）、只给概念导读（2,393）、什么都不给（939）。
先在未训练的模型上测闸门：concepts 档用 16% 的成本保住 78% 的性能，作为训练口径；
nodocs 档 5.5% 太低但非零，且纯度反而最高（没文档时它仍写 cuTile 而不是退回 PyTorch），
所以按 15% 的比例混进训练集。

同时把任务多样性提上去：`torch.nn` 完整卷积面（转置、空洞、深度可分离、分组、1D/3D、
非对称）、组合链（Level 2 的本质，此前没有任何任务是这个形状）、以及**让模型自己出题**
（它对 PyTorch 极熟，不会的只有 cuTile；用执行验证合法性）。

| | 二至七阶段 | 第八阶段 |
| --- | ---: | ---: |
| 任务定义 | 1,480 条（累计） | 3,200 |
| 不同算子形态 | 29 个 builder | **651** |
| 训练样本 | 349–642 | 2,443 |
| 优化步 | 44 | **306** |
| 每 epoch token | 5.6M | 6.7M |

**7 倍样本只花 1.2 倍 token。** 一个没预料到的副产品：非有限 loss 的丢弃率从 **30.5%
降到 0.1%**——第五阶段花了很长时间调查那些 NaN 并归因于梯度检查点，实际上是序列长度，
序列缩到 2.4k 就不再发生。当时建的容错机制是在治症状。

### 顺带证明：那个 MoE 全量微调从来没必要

第二阶段起每轮都在训 6.88B 参数，其中 4.83B 是**误打误撞的 MoE 专家全量微调**。这一轮用
同样数据训了替代方案（仅注意力与 DeltaNet，rank 提到 128）：

| | 可训练 | Level 1 解出 | 时长 | 峰值显存 | adapter |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全量专家 r=32 | 6.88B | 46/100 | 87 min | 83.9 GB | 26 GB |
| **仅注意力 r=128** | **137M** | 45/100 | **31 min** | **46.5 GB** | **537 MB** |

**用 1/50 的参数打平。** 此前所有的显存紧张、NaN 绕行、27.5GB 存档、以及第六阶段为了
存档把专家冻掉（只剩 34M，这正是 RL 没动的首要嫌疑）——全都是可以避免的。

### 修复循环这一轮失效了

5,020 次修复尝试只成功 23 次，而第三阶段是 744 次里 100 次。两个原因大概都成立：
model-E 经过三轮微调加 RL 后倾向一次给出最好答案、见到报错也只是重复；新增的算子族
（转置、空洞、3D）本身远超它的能力，给报错也没用。后果是本轮计划的轨迹数据集几乎没有
素材，agentic SFT 这条线是**未验证**而不是被否证。

详见 [results/REPORT_PHASE8.md](results/REPORT_PHASE8.md)。

---

## 第九阶段：修复循环的收益是假的

把修复循环加到 model-F 上，200 题从 84 道涨到 **107 道**（卷积 33 → 43，fast_1.0 17 → 26），
看起来是本项目最大的一次提升。**但它经不起等预算对照。**

修复循环每题最多 16 次尝试（4 样本 × 3 轮修复），而对照的单次生成只有 4 次。补上等预算
对照——同一个模型、同一档 prompt，**不给任何反馈**，独立采 16 个样本：

| | 样本数 | 解出 | 快过 torch |
| --- | ---: | ---: | ---: |
| F 单次 k=4 | 800 | 84/200 | 17 |
| 修复循环（4 样本 × 4 轮） | 2,701 | 107/200 | 26 |
| 单纯重采 k=12 | 2,400 | 120/200 | — |
| **单纯重采 k=16** | 3,200 | **129/200** | **53** |

**用更少的样本（2,400 对 2,701），什么反馈都不给就多解出 13 道。** 性能维度是压倒性的：
重采 53 道快过 torch，修复循环只有 26 道——修复对话从第一次尝试的结构出发打补丁，
会继承那个结构的性能缺陷，而独立重采每次从头构思，有机会撞上更快的写法。

轨迹数据也支持这个解释：单次修复尝试只有 2-3% 真的修好，50% 以上是把同样的错误重复一遍。
题目层面看着涨，是 16 次尝试复合出来的，不是反馈起了作用。

**第七阶段的结论一并撤回。** 那一节当时就标注了"缺等算力对照"，现在补上了，结论是不成立。
修复循环作为**数据采集器**仍然有效（第三、四阶段靠它把弱类别正样本从 1.2% 提到 21.2%），
失效的是"推理时提分"这个用法。

这一轮还顺带否掉了第八阶段的一个猜测：当时修复循环在新合成任务上几乎不产出，我归因于
"F 微调后一次就给出最好答案、不再响应反馈"。**不对**——F 在 200 题上照样能被反馈推动，
那次失效是任务集太难。

### 官方 harness 会静默漏验

k=16 那次跑完，1,600 个样本里只有 593 个真的被验证，其余标成 `not_evaluated`，
而 pass@k 直接把它们当失败——**第一次算出来 21/100，补完是 61/100，差三倍**。
原因是有样本会挂死 GPU，连带 worker 一起退出，整个 run 提前结束但不报错。

已加 `scripts/eval_until_complete.sh` 反复重跑直到每个 (problem_id, sample_id) 都有结果，
`scripts/passk_curve.py` 遇到未验证样本会显式告警。**此前所有 run 的数字都该按这个标准
复查**——凡是没跑满的，报出来的都偏低。

真正的瓶颈是 `timeout=300`：Level 1 补验一度慢到 30 分钟推进 4 个样本，超时降到 60 秒后
变成 25 分钟推进 428 个。挂死的样本在 300 秒超时下把整个 pass 的时间全占了。

### 顺带：pass@1 与 pass@16 的差值就是 RL 的余量

| | pass@1 | pass@16 |
| --- | ---: | ---: |
| Level 1 | 32% | **68%** |
| Level 2 | 11% | **61%** |

覆盖面远大于可靠性，而且差值比第八阶段结束时估的 20pp 大得多。**这是下一步该吃的东西**：
模型已经知道怎么写对，只是十六次里才稳定写对一次。

详见 [results/REPORT_PHASE9.md](results/REPORT_PHASE9.md)。

---

## 第十阶段：把覆盖面转成可靠性

第九阶段量出 pass@1 21.5% 对 pass@16 64.5%——模型知道怎么写对，只是十六次才稳一次。
这一阶段从**同一个起点（model F）**出发跑两条臂：把模型自己答对的解拿回去做 SFT（自蒸馏），
以及 GRPO。设两条臂是因为本项目的历史是 SFT 每次都推动指标、RL 没有，没有对照就没法
解释 RL 涨的那几个点。

数据是同一份：训练题里随机 1000 道，concepts 档 k=16 采样，15,757 个样本。
这份采集本身就说明靶子有多大——**654 道"有时答对"**（可靠性前沿），174 道从没对过，
172 道次次对。

| | pass@1 | pass@4 | 解出 | 快过 torch | 中位加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 基座 | 12.9% | 23.5% | 47/200 | 5 | 0.12x |
| F（起点） | 21.4% | 39.5% | 79/200 | 21 | 0.55x |
| I（F + GRPO） | 24.6% | 49.0% | 98/200 | 31 | 0.90x |
| H（F + 自蒸馏） | 28.5% | 54.0% | 108/200 | 38 | 0.95x |
| **J（H + GRPO）** | **39.5%** | **63.5%** | **127/200** | **48** | **1.00x** |

**两条臂按类别互补，而且可以叠加。** 自蒸馏在卷积上碾压（31 → **50**/98），
但池化、归约、损失函数一道没多解；GRPO 卷积只 +7，却把**池化 1 → 3、归约 3 → 5**——
这些是前九轮一直推不动的族。自蒸馏的数据分布由"模型已经能答对什么"决定，卷积占了
SFT 集的 29.7%，弱类别本来就没多少正样本可蒸；GRPO 的 frontier 按"有时对"筛，
弱类别偶尔蒙对一次就进训练，组内归一化再把它放大。**一个吃厚尾，一个吃长尾。**

自蒸馏**没有拿覆盖面换可靠性**（这是我下场前最担心的）：pass@16 也从 129/200 涨到
**145/200**，三个口径同时涨。成本是 1534 条样本、24 分钟、零 NaN 丢弃。

### 叠加：在自蒸馏之上再跑 RL

配方和 I 完全一样，只把起点从 F 换成 merged H，**frontier 用 H 重新筛**（拿 F 的直接用会
浪费大半 rollout——H 稳定解出的题组内没有梯度；重筛后"次次答对"从 172 道涨到 326 道）。

结果是 **127/200，pass@1 39.5%，48 道快过 torch，中位加速比 1.00x**。卷积到
**69/98，其中 35 道快过 torch**。两个阶跃幅度几乎一样（F → I 是 +19，H → J 也是 +19），
**GRPO 在自蒸馏之上的增益没被吃掉**。

**但不是无损的**：矩阵乘退了 3 道、激活退了 2 道——这两族本就接近天花板（打不过 cuBLAS），
RL 把概率质量挪向 frontier 时把它们挤掉了一点。更值得注意的是**池化：从 F 起跑的 RL
推到 3 道，从 H 起跑的却停在 1 道**，两次的 frontier 不同，弱类别不会因为叠加自动被照顾到，
需要显式配额。

### RL 第一次生效，以及它中途崩过一次

第六阶段 RL 纹丝不动，这次动了 +19 道。差别在四处硬伤：可训练参数 34M → **137M**、
存档 27.5GB → **526MB**、序列 17k → 8k、每轮 954 秒 → **145-280 秒**。最关键的是容量——
第六阶段是在 r=32 的 adapter 上冻掉专家，这次改成在**已合并的** model-F 上挂全新的
仅注意力 r=128 adapter，B 零初始化所以起点策略严格等于 F。

第一次跑（lr 1e-5、无 KL）在第 22 到 28 轮之间**彻底崩溃**：无代码输出从 1.6% 涨到
**99.2%**，模型不写代码了。机制是多数 rollout 失败、多数 advantage 为负，而一次性压低
所有失败输出的最省力办法就是不输出代码；`inner_epochs=1` 时 ratio 恒为 1、裁剪从不生效，
没有东西拦得住。

**修法是 KL 锚，而且它是免费的——这点我之前判断错了。** 代码注释原本写着参考策略要么
抵消监督训练、要么再花 27.5GB。在 `--fresh-lora` 下两句都不成立：adapter 挂在已合并的
监督权重上，**关掉 adapter 得到的就是 SFT 策略本身**，`disable_adapter()` 免费给出精确的
参考 logprob，锚过去是拉向 SFT 而不是拉离它。

### 训练 reward 看不出效果，别信它

40 轮的训练 reward 是平的（前半 0.5247、后半 0.5244），照这个读数会判定 RL 又没用，
**但 dev 200 上实打实多解出 19 道**。每轮只从 818 道 frontier 抽 16 道，任务抽样噪声就有
±0.1，把改进完全盖住。**第六阶段"reward 没动所以 RL 没用"，其测量方式本身就撑不起
那个结论。** 判据只能是 held-out 评测。

详见 [results/REPORT_PHASE10.md](results/REPORT_PHASE10.md)。

---

## 第十一阶段：frontier 的类别构成决定 RL 学到什么

第十阶段的叠加净赚 19 道，但矩阵乘退 3 道、激活退 2 道、池化纹丝不动。我以为是弱类别的
frontier 素材不够，**量了发现不是**：池化在 frontier 里占 22.4%，是它 dev 占比的四倍多，
中位通过率 0.250 全场最低——素材不缺，是真的难。

真正的洞在别处：**frontier 里没有 activation，也没有 loss**。任务生成器压根没有这两族的
builder，最接近的只有一个八算子、只在 2D 上的 `elementwise`。而它们是 dev 200 的
**35 道题（17.5%）**，前十轮从没有人给它们造过任务。

补上之后（按 `torch` API 面枚举 24 个激活、7 个两输入损失，激活在 2D/3D/4D 都出），
再给 `select_frontier.py` 加类别配额，把 matmul 从 10 道提回 35 道。J 和 K
**起点相同、配方完全相同，只有 frontier 的选法不同**：

| | pass@1 | pass@4 | 解出 | 快过 torch |
| --- | ---: | ---: | ---: | ---: |
| H（起点） | 28.5% | 54.0% | 108/200 | 38 |
| J（按 spread 排序） | 39.5% | 63.5% | 127/200 | 48 |
| **K（类别平衡）** | **42.8%** | **67.0%** | **134/200** | **49** |

**修好的**：矩阵乘 10 → 11，归约 5 → 7，池化第一次到 2 道，卷积再涨 5 道到 74/98。

**没修好的**：**激活仍停在 21，比起点 H 的 23 还低，尽管 frontier 给了它 52 道题**；
损失函数 1 道，也低于 H 的 2 道。给了配额还是退步，说明问题不在"有没有任务"——
可能是合成的单算子激活任务和基准的分布不同，也可能是这两族本就接近天花板、
RL 在别处拿分时把它们挤掉了。**这一条没有定论。**

一个要盯的信号：同样的 KL 系数，K 后半程的 KL 是 0.5522，**是 J 的 70 倍**。策略离 H 远得多，
这次结果是好的，但 0.55 已经不算"锚住"，激活和损失的退步很可能就是这个漂移的代价。

### 开封封存的 held-out：没有过拟合 dev 200

dev 200 已经被用于十一轮选型决策——从没在上面训练，但反复用它挑模型，这是选择性过拟合。
把第九阶段封存的 500 道合成题开封一次（同一个纯度门、同一批测三个模型）：

| | level 97（程序化） | level 98（模型自撰） | 合计 | pass@1 |
| --- | ---: | ---: | ---: | ---: |
| 基座 | 53/300 | 24/200 | 77/500 15.4% | 6.4% |
| F | 196/300 | 104/200 | 300/500 60.0% | 35.8% |
| **K** | **269/300** | **159/200** | **428/500 85.6%** | **64.3%** |

关键不是绝对值（合成题比 KernelBench 容易得多），而是**相对增益：held-out 上 K 是基座的
5.56 倍，dev 上是 2.85 倍——倍数更大**。如果 dev 的增益是选择性过拟合的产物，这里应该更小。
二阶检验也过：level 98 分布和 builder 不同，F 的 97-98 差距 13.3pp、K 是 10.2pp，**收窄了**。

**同时这也量出了分布迁移的代价**：同一个模型在熟悉算子的新形状上 85.6%，在独立的人写题目上
67.0%。这 19pp 说明模型把训练分布学得比"cuTile 本身"透得多，dev 200 仍然是那个保守数字。

详见 [results/REPORT_PHASE11.md](results/REPORT_PHASE11.md)。

---

## 第十二阶段：给"数值错"分级，以及一个把我带偏的诊断 bug

诊断 K 剩下的 66 道未解出题时，我用 `compiled and not numerically_correct` 判定"能跑但
数值错"，报出 92.4%。**这是错的**，而 `analyze_cutile_run.py` 自己的注释早就写了为什么：
`compiled` 只表示模块导入成功，而 cuTile 是**首次 launch 才 JIT 编译**的，所以编译失败的
kernel 照样 `compiled=True`。那个判据把全部 cuTile 编译错误算成了数值错。

用正确的 `failure_stage` 重算，最大的一块根本不是数值：

| | 数量 | 占比 |
| --- | ---: | ---: |
| **数值正确但没过纯度门** | 34 | **51.5%** |
| 能跑但数值错 | 22 | 33.3% |
| cuTile API 误用 | 8 | 12.1% |

一个判据用错，把最大的靶子藏了起来，还顺手造出一个假的。

### 分级 reward 仍然成立，而且效果不小

好在这轮的改动不受影响——`reward.py` 判定数值错走的是验证器的错误文本，不看 `compiled`。
原来的问题是真的：这一档给**恒定 0.6**，从"最后一位不同"到"差六个数量级"同分；对 GRPO
更糟，组内八个样本全落在这档时 advantage 精确为零，整组白跑。改成按相对偏差在
0.6 → 0.3 之间按对数刻度分级（下限仍高于"编译不过"的 0.2，档位序不变）：

| | pass@1 | pass@4 | 解出 | 快过 torch | 末段 KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| H（起点） | 28.5% | 54.0% | 108/200 | 38 | — |
| K（恒定 0.6） | 42.8% | 67.0% | 134/200 | 49 | 0.5522 |
| **L（分级）** | **44.6%** | **72.0%** | **144/200** | **60** | **0.0102** |

**最值得注意的不是 +10 道，是 KL。** 两者到达几乎相同的训练 reward 和 pass，但 K 是靠离开
起点 54 倍的距离换来的。这解释了第十一阶段那个"KL 漂了 70 倍"：**组内没有真实的接近度
排序时，策略只能靠大幅偏移撞运气**；给了带内梯度之后，同样的收益能在起点附近拿到。
K 的矩阵乘退步（13 → 11）大概就是漂移的代价，L 把它拿回到 13。

### 下一个靶子：链尾的 torch 激活

L 剩下 56 道未解出题里，**29 道（52%）是"数值正确但没过纯度门"**，Level 2 上占 75%。
看被挡的原因，73 个样本里：`torch.relu` 17 次、`torch.softmax` 12、`torch.sigmoid` 11、
`torch.logsumexp` 9、`F.gelu` 8、`torch.tanh` 6——**全是逐点激活**。

模型把难的部分（卷积、GEMM、归一化）成功搬进了 cuTile，然后在链尾顺手写个
`torch.relu(...)` 交卷。这也解释了激活族为什么一直"退步"：它从来没稳过，
**模型在链条语境里不会用 cuTile 写激活**，而第十一阶段补的是独立单算子任务，没补到点上。

而 `reward.py` 给纯度失败 **0.0，和空白卷同分**——一个只差最后一个 `torch.relu` 没搬的
正确解，收到的信号和什么都没写一样多。

详见 [results/REPORT_PHASE12.md](results/REPORT_PHASE12.md)。

---

## 第十三阶段：把速度尺换成 `torch.compile`

前十二轮所有的加速比都是对 **eager** PyTorch 量的。这个基线偏弱，而且偏得不随机：
基准的一半是融合链，我们在那里的胜绩几乎全部来自"PyTorch 会物化中间结果"——
**而这正是 inductor 免费消除的东西**。用户在动手写 kernel 之前会先试 `torch.compile`，
所以对 eager 的加速比不是他会关心的数字。

重新量了 `torch.compile`（inductor、fp32、同一块 GB200）的基线，正确性一栏完全不变，
速度一栏缩水：

| | 解出 | 快过 eager | 快过 **compile** | 中位（eager） | 中位（**compile**） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 基座 | 47/200 | 5 | 3 | 0.12x | 0.12x |
| H | 108/200 | 38 | 24 | 0.95x | 0.84x |
| K | 134/200 | 49 | **45** | 1.00x | 0.93x |
| L | 144/200 | 60 | 41 | 1.00x | **0.92x** |

**"中位追平 PyTorch"这个说法不成立**——对 compile 是 0.92x，中位的 kernel 慢 8%。

更难看的一点：按 eager 口径 L（60）明显好过 K（49）；按 compile 口径**反过来了**，
K 45、L 41。L 相对 K 多出来的那批"快"，主要赢在 inductor 本来就能消掉的地方。
`torch.compile` 相对 eager 的中位只有 0.99x / 1.03x，但 **Level 2 上有 44 道题它明显更快**,
所以中位没怎么动、题数掉得多——恰好符合"我们赢的正是 compile 也能赢的"。

所以改的不只是报告口径，**`rl/reward.py` 的速度奖励也换了基线**（`verify/worker.py` 现在
默认把参考模型放进 `torch.compile` 再计时，按任务缓存编译结果，并把 `ref_mode` 记进
每条记录，编译失败会显式退回 eager 并标注）。此前那个奖励是在训练模型去赢一个偏弱的
对手，这大概就是 L 在 compile 口径下反而不如 K 的原因。

两套基线各存一份、不混用：`compare_partial.py --analysis-name analysis_compile.json`
读 compile 口径，默认仍读 eager 以便复现历史数字。

详见 [results/phase13_compile_baseline.txt](results/phase13_compile_baseline.txt)。

---

## 第十四阶段：给纯度失败分级，并补上"链尾激活"这个形态

第十二阶段查出最大的一块是**"数值正确但没过纯度门"**——L 剩下 56 道未解出题里占 29 道，
Level 2 上占 75%。被挡的原因高度集中：73 个样本里 `torch.relu` 17 次、`torch.softmax` 12、
`torch.sigmoid` 11、`torch.logsumexp` 9、`F.gelu` 8——**全是逐点激活**。模型把卷积、GEMM、
归一化都搬进了 cuTile，链尾顺手写个 `torch.relu(...)` 交卷。

两处改动：

**reward 给纯度失败分级。** 原来是 0.0，和空白卷同分——一个只差最后一个激活没搬的正确解，
收到的信号和什么都没写一样多。现在按剩余 torch 算子数在 0.18 → 0.04 之间分级，
**上限严格低于"纯 cuTile 但编译不过"的 0.20**，所以任何真正纯的尝试都不低于任何不纯的，
甩回 PyTorch 永远不划算——这是构造上的保证。没有真实 `ct.kernel`、或用了 `torch.nn`
计算层的，仍然硬零。

**任务补上缺的形态。** `FUSION_TAILS` 原来只有 5 个尾巴，softmax / logsumexp / gelu
一个都没有。现在 13 个逐点尾 + 5 个归约尾，并新增 `anchor_with_tail`（重算子锚 +
1-2 个激活尾），权重全表最高。**这是第十一阶段那次补漏没打中的部分**——那次加的是
独立单算子激活，而模型在独立激活上本来就没问题，它错在**链条语境**里。缺的不是算子，是位置。

| | pass@1 | pass@4 | 解出 | 快过 compile |
| --- | ---: | ---: | ---: | ---: |
| L | 44.6% | 72.0% | 144/200 | 41 |
| **M** | **51.6%** | **75.0%** | **150/200** | 43 |

**激活族 22 → 25，是涨幅最大的一族**，正是瞄准的靶子；归约 5 → 7，归一化 18 → 19，
池化和损失各 +1。训练侧一致：纯度率 0.930 → 0.957，而 KL 只有 0.0045。
卷积退了 2 道，是 frontier 配额让位给新形态的预期代价。

**但只是缓解，没有解决**：Level 1 的纯度失败清零了，**Level 2 仍有 25 道**，占比几乎没变
（75% → 71%）。容易的链尾修好了，难的没动。可能是分级上限压得太低——0.18 对 0.20，
"差一个激活"和"编译不过"几乎无差别，拉力不够。

详见 [results/REPORT_PHASE14.md](results/REPORT_PHASE14.md)。

---

## 第十五阶段：把纯度上限提到 0.3 是个错误；速度那条路通了

上一轮 Level 2 的纯度失败只从 75% 降到 71%，当时的猜测是**分级上限压得太低**——0.18 对
"编译不过"的 0.20，"差一个激活"和"根本跑不起来"几乎无差别。于是这一轮把"数值已正确"的
纯度失败提到 0.20-0.30（需要验证器对不纯候选也跑数值检查，冒烟测试里 65 个纯度失败中
有 29 个数值完全正确，目标人群确实存在）。

代价当时就写在注释里：**这打破了"任何纯的尝试 ≥ 任何不纯的尝试"的结构性保证**，
标注的风险是"在做不对的题上 0.3 会变成舒服的局部最优"。

**风险实现了：**

| | pass@1 | pass@4 | 解出 | Level 2 纯度失败 |
| --- | ---: | ---: | ---: | ---: |
| **M** | 51.6% | **75.0%** | **150/200** | 25/35（71%） |
| N | **55.8%** | 73.5% | 147/200 | **31/38（82%）** |

pass@1 涨了 4.2pp，但**解出题数掉了 3 道，而且瞄准的那个 bucket 反而变大了**。
N 相对 M 在 Level 2 上丢的 11 道里，**9 道的失败模式正是"数值正确但没过纯度门"**——
M 本来能写出完全纯的通过解，N 改成了"算对但留个 torch 尾巴"。

**训练指标完全看不出来**：纯度率 0.9706 → 0.9710 持平、pass 还在涨。因为合成任务的融合链
比基准短得多，纯度率已在 0.97 的天花板，那个局部最优在训练分布上不构成诱惑。
我中途汇报时据此说"排除了该失败模式"，那个判断是错的。

结论：**那个不变式在做实事，不是在浪费分辨率。** 上限已退回 0.19（仍在 0.20 以下，
但保留"算对"与"算错"的区分）。Level 2 剩的 25 道不是奖励刻度问题，得换思路。

### 顺带：速度是探索问题，不是知识缺口

判据在跑之前写死在 [verify/speed_probe.py](verify/speed_probe.py) 里。用 M 在 16 道稳定
解出的合成题上高温采样（T=1.2，平时 1.0），看**正确样本之间**的速度分散：

| | |
| --- | --- |
| 内部差异 ≥1.3x 的题 | **8/16（50%，门槛 1/3）** |
| 中位 best/median | 1.26x |
| **用了多于一组 tile size 的题** | **9/16** |

个别题差得很大（6.46x → 11.05x、5.82x → 10.46x）。而 dev 上常规温度只有 3/39（L1）、
0/15（L2）有这么大差异——**不是模型只会一种实现，是温度把变异压掉了**。

所以下一步很具体：在模型**已经稳定解出**的题上、用更高的 rollout 温度、只按速度给
group-relative advantage。而那批题正是 `select_frontier.py` 现在当"无梯度"丢掉的
（M 的筛选里 326 道"次次答对"被扣掉）——**素材一直都在，只是被扔了。**

详见 [results/REPORT_PHASE15.md](results/REPORT_PHASE15.md)。

---

## 第十六阶段：修好了速度奖励的死区，但训练分布选错了

先说找到的缺陷，它是真的：

```python
def speed_bonus(speedup) -> float:
    if not speedup or speedup <= 1.0:
        return 0.0          # 0.1x 和 0.99x 同分
```

**1.0x 以下一律返回 0**，而最好模型对 `torch.compile` 的中位就是 0.92x——**死区正好盖住了
几乎所有基准 kernel 所在的区间**。十五轮里速度维度在最需要梯度的地方根本没有梯度。改成
对数空间对称分级后，pass 的区间是 0.70–1.30，仍永远高于失败上限 0.60，正确性绝对优先。

frontier 也换成了"稳定解出"的题（`--mode solid`）：这批题上正确性是常数、零 advantage，
速度项自动承担全部梯度。温度提到 1.2。

**但这一轮没有让 kernel 变快。**

| | pass@4 | 解出 | 快过 compile | 中位 |
| --- | ---: | ---: | ---: | ---: |
| **M** | **75.0%** | **150/200** | 43 | 0.92x |
| O | 73.5% | 147/200 | 46 | 0.94x |

表面上 43 → 46、0.92x → 0.94x，但两者解出的题不是同一批。**在都解出的 136 道上逐题对比，
速度比的中位是 1.000x，120/136 变化在 ±5% 以内**（10 快、6 慢）。涨幅全是构成效应。

**原因**：训练日志里 `fast_rate` 中位 **0.70**——七成 rollout 已经快过参考；而 dev 上只有
23%。判定实验也显示那些合成题上模型能拿 6-11x。**训练是在"已经赢很多"的区间里教它赢更多，
而基准的困难在"0.92x、打不过 cuBLAS 和 inductor"那一段。**

**顺带一个流程错误**：建 frontier 用的两次采集跑的是 `fast_verify.py` 不带
`--measure-time`，所以 verified.jsonl 里一条计时都没有——我筛一个**以速度为目标**的训练集，
却对它的速度分布毫无信息，事后回查才发现"frontier 里有计时的题：0"。这本该在筛之前
看一眼分布就发现。

修法很直接：speed frontier 应该按**"稳定解出但比参考慢"**（`0 < speedup < 1.0`）来筛，
那才是基准所处的区间。死区修复、solid 模式、温度 1.2 都保留，错的只是筛选条件缺了速度这维。

详见 [results/REPORT_PHASE16.md](results/REPORT_PHASE16.md)。

---

## 测试集划分

| | 用途 | 状态 |
| --- | --- | --- |
| KernelBench Level 1+2（200 题） | **dev set** | 已用于十一轮迭代决策，侵蚀了；第十一阶段用封存的合成 held-out 验过，**未见选择性过拟合** |
| 合成 held-out（level 97/98，500 题） | 干净的内部参照 | 已开封（第十一阶段）：K 解出 **428/500**，相对增益比 dev 上更大。此后不再用于调优决策，见 [tasks/HELDOUT.md](tasks/HELDOUT.md) |
| KernelBench Level 3（50 题） | **final test** | 已开封（第八阶段）：pass@1 5.0% → **20.0%**，fast_1.0 2/50 → 4/50。此后不应再用于任何调优决策 |
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
patches/     对 KernelBench 自身文件的修改（8 个文件，约 330 行）
golden/      手写的 cuTile 参考解，用于验证题目可解性
rl/          GRPO：分档 reward、可学边界筛选、循环驱动
docker/      评测容器
results/     各轮评测的汇总产物（约 4.7 MB，让报告里的数字可核验）
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
