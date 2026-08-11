# 封存的合成 held-out 集（level 97 / 98）

生成于第九阶段，**未参与任何训练或选数**。

> **已于第十一阶段开封一次**，用途是回答"dev 200 被用于十一轮选型决策，那 47 → 134 的
> 增益有多少是选择性过拟合"。结论：**没有过拟合的迹象**——held-out 上的相对增益比
> dev 上更大（K 是基座的 5.56 倍，dev 上是 2.85 倍）。如果 dev 的增益来自选择性过拟合，
> held-out 的倍数应该更小。结果与边界见
> [results/phase11_heldout.txt](../results/phase11_heldout.txt) 与
> [REPORT_PHASE11.md](../results/REPORT_PHASE11.md)。
>
> 按纪律，**此后不应再用它做任何调优决策**。

| level | 题数 | 来源 | 种子 |
| --- | ---: | --- | --- |
| 97 | 300 | 程序化 builder（curriculum，tier 1-5） | 970001 |
| 98 | 200 | 模型自撰（F 生成，经沙箱验证） | 980001 |

题目本身存在 `tasks/heldout/level97/` 和 `tasks/heldout/level98/`（共 2.1 MB）。
**必须存原文，不能只存种子和命令**：builder 在后续阶段还会改，改了之后同一个种子
生成的就不是同一批题，封存也就失效了。用的时候拷进 `kernelbench/KernelBench/`。

当初的生成命令（留作记录，不保证未来还能复现出同样的题）：

```bash
python3 taskgen/generate_tasks.py --level 97 --count 300 --curriculum \
    --seed 970001 --exclude-levels 90,91,92,93,94,95,96 --clean
python3 taskgen/model_tasks.py --level 98 --count 200 --seed 980001 \
    --exclude-levels 90,91,92,93,94,95,96,97 --clean
python3 taskgen/check_holdout.py --holdout 97,98 --train 90,91,92,93,94,95,96
```

`check_holdout.py` 的结论是 `clean`：500 道题内部无重复，与 2415 道训练题无交集。

## 为什么不能只换种子

生成 level 97 时，**1481 个候选里有 1181 个（80%）与训练集逐字重复**，被哈希去重挡掉了。
形状阶梯本身很短，同一个算子在同一形状上会反复出现。所以"换个种子重新生成一批"
得到的并不是 held-out 集，而是 80% 的训练集副本——这一步以前如果省掉，测出来的数字
会严重虚高。

哈希是规范化之后算的（去注释、去空行、压空格），所以只改格式骗不过去。

## 这个集合能证明什么，不能证明什么

**能**：模型在**新形状、新算子组合**上的泛化。level 98 更进一步——题目由模型自己撰写，
分布和 builder 不同，能测到程序化生成覆盖不到的写法。

**不能**：这不是对 cuTile **API 面**泛化的检验。两个 level 用的都是训练期见过的
torch 算子词表，只是形状和组合是新的。要测 API 泛化，需要一个词表本身就没见过的题集，
目前没有。

也不能替代 KernelBench 200 题：那是人写的、分布完全独立的题目，仍然是唯一的外部标尺。
这个集合的作用是在 200 题被反复使用（已经用了八轮，存在过拟合风险）之后，
提供一个干净的内部参照。

## 使用纪律

- 只在需要判断"是否过拟合了 dev 200 题"时开封，不要用来调参、选数、挑 checkpoint。
- 用过一次就要在 `docs/WORKLOG.md` 里记下用途和结果。
- Level 3（50 题）和 Level 4（20 题）是另一套封存的 final test，规则同上，且只能用一次。
