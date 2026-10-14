# 圆柱绕流 PINN 基线

入口：[train_pinn.py](../train_pinn.py)。本目录提供可独立运行的二维不可压缩 Navier–Stokes PINN，优先用于建立后续改进的对照组。主程序不依赖 PySR、Julia 或 COMSOL Server。

新增可选架构与三组对照试验见 [符号流函数＋神经残差](pinn_hybrid_streamfunction.md)；默认 `uvp` 与既有检查点保持兼容。

## 1. 直接运行

依赖：Python 3.8+、PyTorch 2.4+、NumPy、Matplotlib。当前机器已验证 `D:\keyan\Anaconda\envs\deeplearning\python.exe`，支持 RTX 4060 Laptop GPU；普通 `python` 环境使用 CPU。

```powershell
# 新环境安装；本机已有依赖时无需执行
python -m pip install -r requirements-pinn.txt

# 快速检查：3 步 Adam + 2 步 L-BFGS，会生成流场图和检查点
python train_pinn.py --smoke --device cpu

# 稳态低 Re 圆柱绕流；默认也是此配置
& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' train_pinn.py `
  --config configs/pinn/cylinder_steady.json --device cuda

# 非稳态圆柱绕流
& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' train_pinn.py `
  --config configs/pinn/cylinder_unsteady.json --device cuda
```

输出自动写入 `outputs_pinn/<case>/<时间戳_随机标识>/`。可用 `--out-dir` 指定新的或空的目录；程序拒绝覆盖已有结果。`--seed`、`--adam-steps`、`--lbfgs-steps` 可覆盖配置。`--no-plots` 跳过绘图。

默认配置是**参数明确的独立标准算例**，没有声称与现有 COMSOL 模型完全一致：导出文件未提供 Re，实际入口、初值、出口条件也必须从原仿真设置核对。

| 配置 | Re | 输入 | 初值与训练预算 |
|---|---:|---|---|
| `configs/pinn/cylinder_steady.json` | 20 | `(x,y)` | 10,000 Adam + 1,000 L-BFGS |
| `configs/pinn/cylinder_unsteady.json` | 100 | `(x,y,t)` | 初始静止，`t*=0..20`，20,000 Adam + 2,000 L-BFGS |

这些是起始实验预算，不是精度保证。尤其 Re=100 的长时间涡脱落，普通全时域 PINN 可能训练困难，需参考场、延长时间范围和多随机种子评估。`--smoke` 只用于检查执行。

## 2. 方程、量纲和边界

以圆柱直径 `D`、参考截面平均入口速度 `Uref`、密度 `rho` 无量纲化：

```text
x* = x/D, y* = y/D, t* = t Uref/D
u* = u/Uref, v* = v/Uref, p* = p/(rho Uref²)
Re = Uref D / nu
```

以下省略星号，网络输出 `(u,v,p)`，残差为：

```text
r_c = u_x + v_y
r_u = u_t + u u_x + v u_y + p_x - (u_xx + u_yy)/Re
r_v = v_t + u v_x + v v_y + p_y - (v_xx + v_yy)/Re
```

稳态移除时间导数。网络内部把输入映射到 `[-1,1]`，自动微分仍针对原无量纲坐标，链式缩放保留在计算图中。激活函数使用 `tanh`，适合二阶求导。

几何为 `[0,22] × [0,4.1]` 通道，圆心 `(2,2)`、半径 `0.5`。长度、通道高度和圆心可由 JSON 修改；直径始终为一个无量纲单位。

- 入口：`u=6 y(H-y)/H²`、`v=0`，截面平均速度为 1，最大速度为 1.5。
- 非稳态入口额外乘 `1-exp[-(t/ramp_time)²]`，默认 `ramp_time=2`。
- 上下壁面、圆柱表面：`u=v=0`。
- 右出口：零 Cauchy 牵引，`-p+2u_x/Re=0`、`(u_y+v_x)/Re=0`。
- 非稳态初值：整个流体域 `u=v=0`。入口在 `t=0` 同样为零。

出口牵引固定了压力基准，因此圆柱算例不另加压力锚点，也没有同时施加冗余的 `p=0`、`u_x=v_x=0`。此边界形式必须与外部参考仿真一致后才能比较。默认启动函数不同于一些公开圆柱基准的正弦驱动，不应直接引用那些基准的升阻力值。

损失为各组分项 MSE 之和乘组权重：`weight_pde`、`weight_boundary`、`weight_initial`、`weight_gauge`、`weight_data`。圆柱算例不使用 `gauge` 项。边界各段等权，而非按弧长加权；`n_boundary` 表示**每个边界段**的点数。连续性、两个动量方程及入口/两壁/圆柱/出口分别记入日志。

## 3. 训练、保存与评估

Adam 阶段每 `resample_every` 步重新均匀采样，流体内部采用拒绝采样排除圆柱。L-BFGS 整个阶段固定内部点、边界点及观测子集，避免线搜索期间目标函数变化。`lbfgs_steps` 统计外层优化器更新次数，实际闭包调用数单独记录。

每次运行包含：

| 文件 | 内容 |
|---|---|
| `config.json`、`environment.json` | 最终配置、随机种子、设备、依赖版本、数据哈希、运行类型 |
| `history.csv`、`training_loss.png` | 各损失分项及总损失；采样变化时不同日志点的目标集合不同 |
| `checkpoint_last.pt` | 网络权重、配置、进度；定期写入，训练结束或异常时再次保存 |
| `training_status.json` | 完成步数和是否正常完成训练；不表示 CFD 精度已收敛 |
| `evaluation/metrics.json` | 独立采样点的 PDE/边界/初值 MSE，流量平衡，参考误差（如提供） |
| `evaluation/fields.npz` | 流体域坐标、`uvp`、涡量、三个方程残差；非稳态包含多个时刻 |
| `evaluation/fields.png` | 稳态场或非稳态末时刻的 `u,v,p` 和残差图 |
| `evaluation/forces.csv` | 由表面压力与黏性应力积分得到的 `Cd,Cl` 及各自分量 |

系数采用 `Cd=Fx/(0.5 rho Uref² D)`、`Cl=Fy/(0.5 rho Uref² D)`，每单位展向长度，法线从圆柱指向流体。均匀角度中点积分，不重复首尾节点。瞬态输出至少 21 个时刻，供诊断使用；此时间分辨率和短启动窗口不适合直接提取 Strouhal 数。

独立重载评估：

```powershell
python train_pinn.py --evaluate outputs_pinn/my_run/checkpoint_last.pt `
  --out-dir outputs_pinn/my_run_evaluation --device cpu
```

从已有权重继续优化：

```powershell
python train_pinn.py --config outputs_pinn/my_run/config.json `
  --init-from outputs_pinn/my_run/checkpoint_last.pt `
  --adam-steps 1000 --lbfgs-steps 200
```

`--init-from` 是**权重热启动**：重置优化器和采样器，不是精确恢复断点。网络、Re、几何和时间域必须一致。`--evaluate` 不训练；可在 CPU 重载 GPU 生成的检查点。

较小的损失、流量差或训练完成标记都不能单独证明流场正确。升阻力尤其依赖壁面导数，必须结合独立 CFD 参考和表面/时间积分加密检查。

## 4. 接入 COMSOL 瞬态数据

`--data <npz>` 增加观测损失；`--reference <npz>` 仅在训练完成后评估。两者均可省略，默认是纯物理 PINN。训练数据小批次随 Adam 采样周期更新，L-BFGS 使用固定子集。

NPZ 契约如下：

| 键 | 形状/取值 |
|---|---|
| `coords` | 稳态 `(N,2)`，非稳态 `(N,3)`，列顺序 `x,y[,t]` |
| `velocity` | `(N,2)`，列顺序 `u,v` |
| `pressure` | 可选 `(N,1)` 或 `(N,)`；必须使用相同压力基准 |
| `field_kind` | 稳态 `steady`；非稳态 `instantaneous`，标量字符串 |
| `coordinate_system` | `dimensionless`，标量字符串 |
| `reynolds` | 可选标量，存在时必须与训练配置一致 |

加载时检查有限值、形状、时间/空间域和固体内部点。程序拒绝把 `time_mean` 数据作为稳态 NS 数据；已有 `outputs_mgn/comsol/eta_sym` 的时均尾流不能直接喂给这个物理损失。涡脱落的时均方程包含雷诺应力项。

[scripts/prepare_pinn_comsol.py](../scripts/prepare_pinn_comsol.py) 读取本项目的宽表 `x y [u,v,p,spf.U,x,y] × 时间`，选取瞬态快照并无量纲化，不做时均。需提供仿真的真实物性参数，例如：

```powershell
# 下列参数仅示范用法，必须替换为目标仿真的真实设置
python scripts/prepare_pinn_comsol.py --source cylinder_flow_result `
  --output outputs_pinn/data/instantaneous.npz `
  --length-ref 0.1 --velocity-ref 1.0 --density 1000 --reynolds 100 `
  --time-start 0 --time-end 2 --time-stride 20 --node-stride 4

python train_pinn.py --config configs/pinn/cylinder_unsteady.json `
  --data outputs_pinn/data/train.npz --reference outputs_pinn/data/test.npz
```

转换脚本不自动生成 `train.npz/test.npz`。按时段或空间块划分独立验证集，保留字符串元数据及 Re，避免同一数据同时训练和宣称独立验证。`reference_error` 只表示给定参考集上的误差，程序不推断其独立性。压力误差不自动去均值。

转换时 `t*=t Uref/D`，不重设时间零点；参考数据的最大无量纲时间须不超过配置 `t_end`。`--origin x0 y0` 用于坐标原点转换，`--pressure-offset` 以原始压力单位指定常数偏移。先核对几何、入口剖面/时间函数、初值、Re 和出口条件，再使用观测损失。若截取的是已经发展的流动窗口，应实现该窗口的初值，不能仍使用默认静止初值。

## 5. 后续改进接口

| 改进目标 | 修改位置 |
|---|---|
| Fourier 特征、不同网络、流函数输出 | `pinn/model.py`；保持 `(x,y[,t]) -> (u,v,p)` 约定或同步修改残差 |
| gPINN、额外约束、数据导数损失 | `pinn/losses.py:loss_terms` |
| 自适应损失权重 | `pinn/losses.py:weighted_loss` 及训练状态保存 |
| 残差自适应采样、近壁加密 | `pinn/problems.py:FlowProblem.sample/interior` |
| 时间分段、因果训练、调度器 | `pinn/train.py:train` |
| 方程/湍流闭合 | `pinn/physics.py:navier_stokes` |
| 升阻力、独立参考、误差指标 | `pinn/evaluation.py` |

网络应逐点运算，避免 BatchNorm 或跨样本耦合层改变自动微分的含义。比较改进时固定物理问题、测试集、网络参数量/预算，并运行多随机种子；保存独立目录。建议先在 Re=20 上验证边界和速度场，再做 Re=100 非稳态；解析 Kovasznay 和 Taylor–Green 仅作为现有方程实现的验证工具。

## 6. 验证

```powershell
python -m pip install pytest
python -m pytest tests/test_pinn.py -q
python -m pytest tests -q
```

测试包含稳态与非稳态解析 NS 残差、Poiseuille 压力/黏性项符号、输入缩放链式求导、真实反向传播、圆柱几何采样、兼容初值、出口牵引、应力积分及其归一化、COMSOL 瞬态转换、错误数据拒绝，以及真实 Adam/L-BFGS 训练—保存—重载—评估。这些验证针对程序正确性，不替代真实圆柱流场精度评估。
### 本次验证记录（2026-09-22）

- PINN 专项：在 `deeplearning`（Python 3.8.20、PyTorch 2.4.1+cu121）运行 `tests/test_pinn.py`，22 项通过。
- 完整回归：在 `PythonProject`（Python 3.12.11、NumPy 2.0.1、PyTorch 2.7.1 CPU）运行全部测试，40 项通过。该环境借用 base 环境的 pytest，关闭第三方 pytest 插件自动加载，并在测试进程中设置 `MKL_THREADING_LAYER=SEQUENTIAL`；没有安装或修改全局依赖。
- GPU 实跑：RTX 4060 Laptop GPU，稳态短程 300 Adam + 100 L-BFGS；另完成非稳态冒烟训练、独立重载评估及绘图。
- 稳态短程训练总损失约从 `7.280` 降至 `0.101`，独立采样损失约 `0.0902`；但入口流量约 `4.024`、出口约 `-0.0678`，仍有明显守恒误差，**未达到物理收敛**，其升阻力不可用作基准结果。
- 验证产物位于 `outputs_pinn/verification/cylinder_steady_short`、`cylinder_steady_reloaded`、`cylinder_unsteady_complete`。配置保存在各自运行目录中。
- 已实际读取原始 COMSOL 时间表头，识别出 `0..8 s` 的 801 个快照；数值转换和量纲测试使用有已知答案的合成宽表。尚未将真实 COMSOL 数据用于训练，因为其 Re 和完整边界条件未确认。

本机默认 Python 的 NumPy/Pandas 二进制版本不兼容，`deeplearning` 的 Python 3.8 又不满足旧 PISR 全部测试需要的 Python 3.10+；因此推荐用 `deeplearning` 运行这里的 PINN，完整旧项目回归使用版本匹配的 Python 3.10+ 科学计算环境。

绘图在独立 Python 进程中完成，避免此机器上 PyTorch 与绘图库同时加载冲突的 OpenMP 运行库；未使用 `KMP_DUPLICATE_LIB_OK`。若已有数值输出需要重新画图，可运行：

```powershell
python scripts/plot_pinn.py --evaluation-dir outputs_pinn/my_run/evaluation
```
## 7. 完整训练结果

2026-09-22 已完成 Re=20 稳态的原配置全量训练（10,000 Adam + 1,000 L-BFGS），以及加密采样续训（2,000 Adam + 5,000 L-BFGS）。最终训练损失约 0.001853，但最大截面流量相对误差仍为 81.3%，未通过物理收敛筛查。结果用于保留普通 u-v-p PINN 的完整训练基线，不能作为可靠流场或升阻力基准。

配置、权重位置、两轮指标和图表见 [完整训练报告](../reports/pinn_full_training_2026-09-22.md)。独立诊断脚本为 `scripts/audit_pinn_cylinder.py`，可重载稳态圆柱权重，检查共同独立内部点、密集边界、多个截面流量及升阻力积分加密结果。
