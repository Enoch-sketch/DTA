# DTA training optimizer

这是一个针对 PMHGT、PairSel-DTA 和 MMCLKin 的项目级 Codex skill。它基于
[`nnu-sky/gpu-training-infra`](https://github.com/nnu-sky/gpu-training-infra) 的“基线—优化—验证”思路，
增加了 DTA 模型的具体热点、论文边界和可执行的只读审计/微基准工具。

Skill 位于：

```text
.agents/skills/dta-training-optimizer/
```

在 Codex 打开本仓库后可以直接说：

```text
使用 $dta-training-optimizer 审计并优化我的 PMHGT（或 PairSel/MMCLKin）训练。
```

也可以用 GitHub 路径安装：

```text
https://github.com/Enoch-sketch/DTA/tree/main/.agents/skills/dta-training-optimizer
```

## 当前审计结论

- 上游 skill 不是“图像模型专用”，而是通用 PyTorch/CUDA 训练优化流程；它同样适用于分子图、蛋白序列和 3D 结构模型。
- PMHGT 的首要候选是 GPU 上向量化 mask、避免每轮清空 CUDA allocator/强制 GC、异步数据传输，以及经过验证的 BF16。
- MMCLKin 的首要候选是把 collate 中循环内反复 `torch.cat` 改成列表累积后一次拼接，并评估 `.pt` 特征的有界缓存。
- PairSel 继承 MMCLKin 的 I/O/collate 热点；优化时必须保持双流随机种子、采样权重与顺序反传语义。

运行静态审计：

```bash
python .agents/skills/dta-training-optimizer/scripts/audit_dta_training.py \
  --project pmhgt=/path/to/PMHGT-DTA-main \
  --project pairsel=/path/to/PairSel-DTA \
  --project mmclkin=/path/to/mmclk
```

运行不依赖数据集的机制微基准：

```bash
python .agents/skills/dta-training-optimizer/scripts/benchmark_hotspots.py --device auto
```

微基准只证明局部改写机制，不能代替真实 GPU 训练前后对照。

对真实 MMCLKin 特征做 collate 等价性与计时检查：

```bash
python .agents/skills/dta-training-optimizer/scripts/optimized_mmclkin_collate.py \
  --original-loader /path/to/mmclk/dataset_loader.py \
  --feature-dir /path/to/mmclk/3dkdavis/3dkdavis_gra_seq_pts \
  --batch-size 8 --repeats 3
```

## 论文边界

这个 skill 本身不属于论文模型方法，也无需作为模型创新写入论文。实际训练仍应披露硬件、框架/CUDA、精度、有效 batch size、随机性与分布式设置。任何改变 loss、采样、batch 语义、调度、近似算法或最终结果的“优化”，都必须作为方法/实验设置说明并重新做对照。
