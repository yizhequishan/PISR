# 硬边界流函数基线

本实验隔离 2026-09-23 暴露的软壁面漏流问题。架构为 `hard_streamfunction`，构造 `ψ=ψ_bg+D Nθ`，速度仍由 `u=ψ_y, v=−ψ_x` 得到，压力由网络的另一输出给出。入口、上下壁面与圆柱的速度边界由解析构造保证，出口 Cauchy 牵引仍使用原来的软损失。没有使用 SR、CFD 数据或新的动量算子。

默认 1500 Adam＋300 L-BFGS 配置用于复现第一阶段，不能将这一配置直接标成已通过完整门槛。结果应按下述完整阶段、统一审计和首轮对照保护条件判断。

实际结果见 [2026-10-13 试验报告](../reports/pinn_hard_boundary_2026-10-13.md)，包含首轮未通过、完整背景消融及最终全部相对验收条件通过的记录。

## 解析构造

坐标沿用原项目的直径无量纲坐标。通道 `[0,L]×[0,H]`，圆心 `(cx,cy)`，半径 `R=0.5`，入口平均速度为 1。定义

```text
s = y/H
P(y) = H s²(3−2s)
P'(y) = 6y(H−y)/H²
A = [x²/(x²+ℓ_in²)] [4s(1−s)]²
φ_c = [(x−cx)²+(y−cy)²−R²]/(2R)
B = (φ_c/ℓ_c)²
w = A/(A+B)
D = AB/(A+B)
ψ_bg = P(y) + w[C−P(y)]
```

默认 `ℓ_in=ℓ_c=1`，未根据测试结果调参。`φ_c` 是光滑的圆柱定义函数，圆柱表面法向导数为 1；它不是全域精确欧氏距离。这里的 `D` 是距离构造的二阶消失因子，不使用最近边界距离的 `min`，也不是要求 `|∇D|=1` 的距离函数。

圆柱严格位于通道内部，因此流体闭包内 `A+B>0`。这些有理式在该区域为 C∞，动量残差需要的流函数三阶导数存在；没有数值 epsilon 改动边界值。各边界满足：

| 边界 | 解析恒等式 | 推出的速度条件 |
|---|---|---|
| 入口 `x=0` | `A=∇A=0, w=∇w=0, D=∇D=0` | `u=P'(y), v=0` |
| 下壁 `y=0` | 同上，`P=0, P'=0` | `u=v=0, ψ=0` |
| 上壁 `y=H` | 同上，`P=H, P'=0` | `u=v=0, ψ=H` |
| 圆柱 | `B=∇B=0, w=1, ∇w=0, D=∇D=0` | `u=v=0, ψ=C` |

仅仅 `D=0` 不足以保证无滑移，必须同时有 `∇D=0`。网络在这些边界上的任意有限取值和导数都不会改变预设速度。

圆柱流函数常数 `C` 不是全局 gauge：它决定圆柱下方的通量，固定为一个任意值会额外约束上下流量分配。默认从 `P(cy)` 初始化并通过同一动量/边界目标学习一个标量，不增加边界自由度中的速度误差。`δC·w` 本身也是满足齐次速度边界的解析模式。`--fixed-cylinder-constant` 可固定该常数，配置字段为 `hard_train_cylinder_constant=false`。神经网络仍只学习 `D Nθ`，修正组比纯神经网络多 1 个标量参数。

不穿圆柱的截面通量为 `ψ(x,H)−ψ(x,0)=H`。穿圆柱的两个流体区间分别贡献 `C−0` 与 `H−C`，总通量仍为 `H`。因此流量守恒在训练前即成立；是否有可用的动量解必须另行检验。

## 同一个 D 的背景消融

| 实验标签 | 背景 | 边界问题是否可满足 |
|---|---|---|
| `soft` | 原纯神经流函数，无 D | 可满足；依靠软边界损失 |
| `zero` | `0` | 不可：全部速度边界为零，通量为零，入口损失有不可消除的下界 |
| `poiseuille` | 原始 `P(y)` | 不可：圆柱速度仍为 `P'(y)`，网络在圆柱上的速度修正为零 |
| `cylinder_poiseuille` | `P(y)+w[C−P(y)]` | 可满足：所有预设速度边界精确成立 |

三个硬因子组的 `D` 完全相同。零背景和原始 Poiseuille 组是字面意义的负对照，不能把它们的低动量残差解释成有效圆柱流解。若零背景另外加入非齐次边界 lifting，它在数学上已经不是 `ψ_bg=0` 的消融。修正后的 Poiseuille lifting 属于手工解析背景，不是 SR 发现的先验。

对默认几何、均匀连续边界积分，零背景的入口分量 RMS 固定为 `sqrt(0.6)=0.7745967`，仅入口项就给总损失贡献 6；原始 Poiseuille 的圆柱分量 RMS 固定为 `1.0287613`，仅圆柱项贡献 `10.5834986`。有限随机边界批次会在这些连续积分值附近波动，训练不能消除相应误差。圆柱净穿透通量可能因正负抵消而接近零，因此必须检查局部速度，不能只看净泄漏。

## 实验协议与验收

默认复用 9-23 的 Re=20、seed=42、4×64 tanh、float64、2048 内部点、每段 256 边界点、1500 Adam＋300 L-BFGS。每组从相同神经网络权重开始，采样序列、损失权重、重采样频率、学习率和 L-BFGS 设置相同；全部原有边界损失项保留，精确边界组的相应项自然为浮点零。更新次数相同不表示时间或 FLOPs 相同。

验收预先规定：修正背景组的最大截面通量误差 `<1%`，独立 20000 点上的 **两个** 动量残差 RMS 分别不高于同预算纯神经流函数，并确认全部预设速度边界最大绝对误差 `<1e-10`。不会因零背景的静止解具有低动量残差而判定成功。

复用 `audit_pinn_cylinder.py` 的 9-23 独立采样种子 `20260923` 和 4097 点截面积分，保留可比性。同时新增按圆柱精确交点切分流体区间的 64/128/256 点 Gauss–Legendre 积分，并与流函数端点差核对，防止跨固体区域的掩膜积分伪差。边界诊断与截面通量在训练前、训练后各记录一次。

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' scripts/run_hard_boundary_trial.py `
  --out-dir outputs_pinn/cylinder/new_hard_boundary_trial --device cuda `
  --adam-steps 1500 --lbfgs-steps 300

& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' scripts/plot_hard_boundary_trial.py `
  --trial-dir outputs_pinn/cylinder/new_hard_boundary_trial

& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' train_pinn.py `
  --config configs/pinn/cylinder_steady_hard.json --device cuda --no-plots

python -m pytest tests/test_pinn.py tests/test_pinn_streamfunction.py tests/test_pinn_hard_boundary.py -q
```

输出目录必须为空。每组保存配置、环境、训练进度、checkpoint、完整损失历史、场数据、常规密集审计和分段积分。试验根目录保存源文件快照和哈希、`comparison.json`、判据分项及所有请求组是否完成的标志。旧架构和 checkpoint 继续通过原入口加载。

默认串行运行。在 CPU 更合适的机器上可使用 `--device cpu --threads 1 --workers 2`；每个进程使用独立目录并重新设置相同随机种子，汇总仅由父进程写入。并行模式只允许 CPU，避免多个训练任务争用同一 GPU。实际设备、线程数和进程数写入协议；比较时应为所有组选择同一设置。

## 对称的圆柱邻域精修

2026-10-13 的首次同预算试验解决了漏流，但修正背景组的独立 y 动量 RMS 为 `0.181905`，高于同轮软边界对照的 `0.121352`，因此首轮没有通过完整验收。训练批次的 y 动量 MSE 为 `0.003006`，独立均匀点为 `0.033089`；网格检查将峰值定位到圆柱周围，不能仅凭较低训练损失宣称成功。

后续精修必须同时应用于两个有效模型：从各自完成的首轮 checkpoint 开始，再做 500 Adam＋300 L-BFGS；不继承优化器状态。内部点总数仍为 2048，1024 个全域均匀点，1024 个在圆柱中心半径 `[0.5,1.5]` 环带内按半径和角度均匀抽样，超出通道的点拒绝采样。两组使用同一新采样序列、设备、线程数和损失权重。该阶段刻意提高了近圆柱区域的训练权重，不属于单纯增加硬边界条件的原预算实验。

`FlowProblem.interior()` 始终是全域均匀采样，只有训练/验证批次所用的 `collocation()` 接受 `cylinder_refine_fraction`。因此原 20000 点全域审计的空间权重不变。由于首轮审计已用于定位问题，额外使用预先固定的种子 `20261014` 再生成 20000 个均匀点；两套审计都要求两个动量分量不劣于同阶段纯神经对照。

```powershell
& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' scripts/run_hard_boundary_trial.py `
  --out-dir outputs_pinn/cylinder/new_hard_boundary_refinement `
  --init-trial outputs_pinn/cylinder/new_hard_boundary_trial `
  --variants soft cylinder_poiseuille --device cpu --threads 1 --workers 2 `
  --adam-steps 500 --lbfgs-steps 300 --cylinder-refine-fraction 0.5 `
  --extra-audit-seed 20261014
```

源 checkpoint 必须已完成。结果记录原 checkpoint 的哈希、原预算和优化器重置状态；两阶段合计每组 2000 Adam＋600 L-BFGS，但不能把这种分阶段 warm start 描述成一次连续 2000＋600 的训练，也不能用额外精修的硬边界模型与未精修软边界模型宣称同预算优势。

零背景和原始 Poiseuille 的固定边界缺陷与采样无关；它们保留在首次四组同预算表中，不将其首轮结果混入两组精修的同预算对照。后续 SR 试验应继承相同 lifting、D、自由常数处理和完整采样/优化协议，再比较额外符号项。

## 入口与圆柱联合精修及原始对照保护

仅加密圆柱的第二阶段降低了近圆柱残差，但两组在入口附近出现较高 y 动量峰值：硬边界全域 RMS 为 `0.167152`，软边界为 `0.370570`。硬边界虽优于这一阶段的软边界，仍劣于首轮软边界的 `0.121352`，不能据此宣称原始门槛已通过。

第三阶段从两组各自的第二阶段 checkpoint 继续，均增加 300 Adam＋300 L-BFGS。2048 个点分为 1024 全域均匀、512 圆柱环带、512 入口条带 `[0,1]×[0,H]`；训练损失仍不加重要性权重，物理算子和 D 不变。两组都使用 CPU、单线程，并行运行不共享优化器。额外审计使用新种子 `20261015`。`--reference-trial` 还要求硬边界在两套均匀审计上均不劣于首轮纯神经模型，防止对照退化造成虚假的成功。

```powershell
& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' scripts/run_hard_boundary_trial.py `
  --out-dir outputs_pinn/cylinder/new_hard_boundary_inlet_refinement `
  --init-trial outputs_pinn/cylinder/new_hard_boundary_refinement `
  --reference-trial outputs_pinn/cylinder/new_hard_boundary_trial `
  --variants soft cylinder_poiseuille --device cpu --threads 1 --workers 2 `
  --adam-steps 300 --lbfgs-steps 300 `
  --cylinder-refine-fraction 0.25 --inlet-refine-fraction 0.25 --inlet-refine-width 1 `
  --extra-audit-seed 20261015
```

完整三阶段每个有效模型合计 **2300 Adam＋900 L-BFGS**，含两次优化器重置。跨阶段同时增加了预算并改变了采样，不能把跨阶段改善全部归因于采样加密；硬/软边界收益的有效对照是同一完整协议下的两组结果。首轮四组、第二阶段结果和每次源 checkpoint 的哈希均保留。

本轮范围为稳态圆柱；瞬态模式会明确报错。该基线验收不同于原项目的更严格筛查（例如各 PDE 分量 RMS `<0.01`），也不替代独立 CFD 场和升阻力精度验证。
