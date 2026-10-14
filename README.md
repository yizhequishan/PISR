# Cylinder Flow PISR

基于 COMSOL 圆柱绕流数据的物理启发符号回归（PISR）实验。项目先从时均速度场构造无量纲尾流数据，再拟合速度或流函数，并用 Euler 几何算子和多算例有限差分检验半径、圆心位置等几何参数的灵敏度。

核心分解为

$$
T_p = K_p + M_p,
$$

其中 `T_p` 是固定物理坐标下的总响应，`K_p` 是坐标变换引起的输运项，`M_p` 是固定无量纲坐标下的内在响应。流函数路线使用

$$
u^*=\partial_\eta\psi^*,\qquad v^*=-\partial_\xi\psi^*,
$$

从结构上满足二维不可压缩约束。

## 圆柱绕流 PINN 基线

新增独立的标准 Navier–Stokes PINN，支持 Re=20 稳态和 Re=100 非稳态圆柱绕流、Adam + L-BFGS、模型保存与评估、流场/残差图、升阻力积分及 COMSOL 瞬态观测接口。网络、物理残差、采样和损失分开，便于后续方法改进。

```powershell
python train_pinn.py --smoke --device cpu
python train_pinn.py --config configs/pinn/cylinder_steady.json
python train_pinn.py --config configs/pinn/cylinder_unsteady.json
```

详细配置、量纲、边界条件和数据契约见 [docs/pinn_baseline.md](docs/pinn_baseline.md)。默认算例不假定与现有 COMSOL 设置相同；时均尾流数据不能直接作为标准稳态 NS 的解。

入口、上下壁面与圆柱的硬边界流函数基线及零背景/Poiseuille 消融见 [docs/pinn_hard_boundary.md](docs/pinn_hard_boundary.md)。解析 lifting 保证通量与速度边界，独立比较动量残差，用于隔离边界修正和后续 SR 先验的收益。[2026-10-13 试验报告](reports/pinn_hard_boundary_2026-10-13.md) 保留首轮失败与后续同预算精修的完整结果。

Re=20 稳态已完成两轮全量训练，结果及未收敛原因见 [完整训练报告](reports/pinn_full_training_2026-09-22.md)。

## 当前状态

以下数据与模型目录描述本地历史存档。GitHub 保留的核心结果及其证据见 [结果文档索引](reports/README.md)；完整运行产物需在本地保留或重新生成。

- 基准 COMSOL 尾流数据 `outputs_mgn/comsol/eta_sym` 已更新为 `dense_xi_v1`：回归窗 `ξ∈[3,5]` 上为 `33 × 64` 网格，`Δξ=0.0625`、`Δη≈0.0603`。
- `outputs_mgn/comsol/eta_sym_psi` 中现有符号流函数模型仍来自旧 `16 × 64` 网格，不能与新网格直接混用；应从 Step 2 重新训练。
- `outputs_mgn/comsol/eta_sym_psi_mlp` 已使用新 `33 × 64` 网格，并包含 3 个随机种子的聚合结果。
- `outputs_mgn/fd_check/phi_R_pysr` 中的响应流函数结果仍对应旧 `16 × 64` 网格；基准 `ψ` 重训后，Step 5–6 也需要重新运行。

以下为历史产物中的结果，尚未按本次修正协议重算；旧 P2、P1 稀疏均匀采样比较不应作为新协议的验证证据。

对现有结论的独立重算、P1/A0 重跑、P2 交叉选式与空间验证见 [逻辑与实验验证报告](reports/logic_validation_2026-09-08.md)。

| 实验 | 当前结论 |
|---|---|
| P0：`T=M+K` 闭合 | 精确离散路径的最大相对 L2 误差为 `1.70e-16`，说明坐标、符号和归一化记账闭合；现有符号梯度仍有明显误差。 |
| P1：微分观测可辨识性 | `u` 单分量观测秩为 `3/7`，加入 `v` 后为 `6/7`，剩余 1 维是流函数常数规范自由度。 |
| P2：解析导数损失 | 旧选式网格上的梯度 MSE 降低 `68.4%`，函数值 MSE 增加 `14.3%`；选式与评价复用同一网格，独立测试提升待重跑。 |
| P3：不同 `H/D` 的半径响应 | 当前数值分辨率下“无结论”：`±1%R` 与 `±2%R` 给出相反趋势，且步长误差约为 `75%–98%`。 |

更完整的 P0–P2 结果与修正后的结论见 [成果汇总报告](reports/圆柱绕流符号建模与灵敏度研究成果汇总报告_2026-09-20.md)，P3 判定及适用边界见 [outputs_hd_p3/P3_assessment.md](outputs_hd_p3/P3_assessment.md)。

## 项目结构

```text
.
├── step1_extract_wake_comsol.py   # COMSOL 导出数据预处理
├── step2_pisr.py                  # 直接拟合速度 u 的基线
├── step2_pisr_psi.py              # 拟合流函数 psi 的主流程
├── step2_eta_sym.py               # 对称窗口旧入口
├── step3_visualize.py             # 拟合结果可视化
├── step4_euler_sensitivity.py     # Euler 几何灵敏度
├── step5_fd_sensitivity_check.py  # 总响应/输运/内在响应分解闭合
├── step6_phi_R_analysis.py        # 响应流函数 phi_R 的选式与验证
├── diag_v_xi.py                   # 横向速度与 xi 依赖诊断
├── experiments/                   # P0–P2 与梯度审计脚本
├── scripts/                       # Windows 本地 Python/Julia 启动器
├── docs/                          # 理论、设计与实施记录
├── outputs/                       # P0–P2 结果与综合摘要
├── outputs_mgn/                   # 主流程数据、模型、图表和有限差分结果
├── outputs_hd_p3/                 # H/D=4、8 的 P3 诊断产物
└── archive/                       # 已退出主流程的历史实验
```

模块关系、文件契约和迁移顺序见 [docs/pipeline_contracts.md](docs/pipeline_contracts.md)，预备实验的逐项运行说明见 [experiments/README.md](experiments/README.md)。

## 环境

普通预处理、分析和绘图脚本需要 Python 3.10+ 以及以下依赖：

```powershell
python -m pip install numpy pandas scipy sympy matplotlib
```

符号回归还需要 PySR 和 Julia。本工作区的 PowerShell 启动器固定使用本机 `deeplearning` conda 环境与 Julia project；换机器后请先修改 [scripts/Invoke-DeeplearningPython.ps1](scripts/Invoke-DeeplearningPython.ps1) 中的路径。

```powershell
# 运行任意需要 PySR 的 Python 脚本
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\step2_pisr_psi.py --help
```

## 主流程

以下命令均在项目根目录执行。

### 1. 生成严格对称的密集尾流网格

```powershell
python .\step1_extract_wake_comsol.py `
  --source .\cylinder_flow_result `
  --out-dir .\outputs_mgn\comsol\eta_sym `
  --symmetric-y-window `
  --n-x-grid 97 `
  --min-wake-x-points 32
```

默认使用后半段时间步做时均。`--min-wake-x-points 32` 会阻止流程静默退回旧的 16 列粗网格。运行后应在 `meta.json` 中确认：

```text
grid_revision = dense_xi_v1
n_x_wake      = 33
n_y           = 64
```

### 2. 训练流函数 PISR

```powershell
.\scripts\Invoke-DeeplearningPython.ps1 `
  -ScriptPath .\step2_pisr_psi.py `
  --base-out-dir .\outputs_mgn `
  --input-subdir comsol/eta_sym `
  --out-subdir comsol/eta_sym_psi
```

默认同时运行 `A_psi`（只使用 `u` 损失）和 `B_psi`（联合 `u+v` 损失）。只运行推荐的联合路线可追加 `--skip-strategy-a`；可通过 `--lambda-v`、`--niterations` 和 `--maxsize` 调整搜索。

每次训练写入独立的 `eta_sym_psi/runs/<run_id>/`，成功后才更新 `current_run.json`。后续脚本接受上面的根目录并自动解析当前完成的运行；根目录旧的 A/B 文件保留为历史快照，不会混入新运行。所选表达式及指纹保存在 `selected_models.json`。

### 3. 可视化并计算 Euler 坐标输运项 K

```powershell
python .\step3_visualize.py --out-dir .\outputs_mgn\comsol\eta_sym_psi
python .\step4_euler_sensitivity.py --out-dir .\outputs_mgn\comsol\eta_sym_psi
```

### 4. 用七组 COMSOL 数据检查有限差分闭合

```powershell
python .\step5_fd_sensitivity_check.py
```

该步骤读取基准算例以及 `R±.txt`、`cx±.txt`、`cy±.txt`，计算固定物理坐标总响应、坐标输运项和内在响应，并生成 Step 6 所需的 `outputs_mgn/fd_check/phi_R_input`。

旧 FD 归档缺少完整字段和版本信息，必须先重跑 Step 5 才能用于新版 P0、Step 6 和梯度审计。可用 `--source-dir`、`--base-grid-dir`、`--psi-dir`、`--out-dir` 指定独立的数据与输出目录。

### 5. 拟合并验证半径响应流函数

```powershell
.\scripts\Run-PhiRPySrLocal.ps1
```

启动器先拟合 `φ_R`，再运行 `step6_phi_R_analysis.py`，输出位于 `outputs_mgn/fd_check/phi_R_pysr/runs/<run_id>/`。Step 6 在当前 FD 目标上重新计算候选误差，保存 `response_selection.json`；这属于当前目标上的响应重建，独立泛化误差需要另设测试数据。

## 预备实验与审计

不需要重新训练 PySR 的实验：

```powershell
python .\experiments\p0_derivative_closure.py
python .\experiments\p1_observability_nullspace.py
python .\experiments\psi_pareto_gradient_audit.py
python .\experiments\mlp_psi_gradient_audit.py
```

P2 会重新训练两个 PySR 模型。首次运行先安装并预编译 Julia 的 `Zygote` 扩展：

```powershell
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\scripts\setup_pysr_derivatives.py
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\experiments\p2_analytic_derivative_loss.py
python .\experiments\summarize_preliminary_experiments.py
```

`setup_pysr_derivatives.py` 会修改本机 Julia 环境，只需执行一次。

新版 P2 用 `--n-validation` 控制独立选式集，冻结表达式后才计算密集测试网格指标。P1 对每个观测预算重新在全域均匀选点。历史运行产物保留在本地，GitHub 仅维护结果索引中的核心报告及必要证据；汇总脚本会根据协议标记新旧结果。

接口回归测试：`python -m pytest tests -q`。合成测试执行真实的数据预处理、文件读写、选式和评价，仅用解析候选替代 PySR 搜索；它不代表真实 CFD 重训结果。

## 数据与产物

| 路径 | 用途 | 版本控制建议 |
|---|---|---|
| `cylinder_flow_result` | 基准 COMSOL 时序导出 | 大文件，仅保留在本地或外部存储 |
| `R±.txt`、`cx±.txt`、`cy±.txt` | 六组参数扰动算例 | 大文件，仅保留在本地或外部存储 |
| `H_D.txt` | `H/D=4,8` 的 P3 原始数据，约 12 GB | 大文件，仅保留在本地或外部存储 |
| `outputs_mgn/comsol/eta_sym` | 标准化密集尾流输入 | 用 `meta.json` 检查网格版本 |
| `outputs_mgn/fd_check` | 分解闭合、梯度审计与 `φ_R` 数据 | 保存关键表格、报告和必要复现产物 |
| `outputs` | 独立实验结果 | 仅跟踪核心报告引用的最终指标、协议及配图 |
| `outputs_hd_p3` | P3 判断报告及指标摘要 | 完整场数据保留在本地；尚无一键重跑脚本 |

不要依赖绝对路径复制运行产物；跨机器复现时，应从原始 COMSOL 数据重新执行 Step 1。提交前也请特别检查大型二进制文件、模型权重、缓存和长时间训练产物，避免误入 Git 历史。

GitHub 结果文档以 [reports/README.md](reports/README.md) 为入口。旧规划、重复 Word/PPT、冒烟运行、候选枚举、模型权重和场数组保留在本地；新增结果按 `.gitignore` 中的保留清单纳入版本控制。
