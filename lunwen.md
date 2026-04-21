 ▎ Physics-Informed Symbolic Regression for Single-Simulation Geometric Sensitivity Analysis in Fluid Device Design

  ▎ 基于物理约束符号回归的单次仿真流体器件几何敏感度分析方法

  ---
  1. 前提假设与适用条件 (Assumptions & Scope)

  本方法建立在以下两个核心假设之上，论文中需要明确声明：

  **假设 H1：固定 Reynolds 数条件 (Constant-Re Assumption)**

  参数变化在固定 Reynolds 数下讨论。即当几何参数 $R \to R + \delta R$ 时，
  入口条件（如 $U_{in}$）相应调整，使得 $Re = U_{in}D/\nu$ 保持不变。

  数学表述：
  $$\frac{\partial u^*}{\partial R}\bigg|_{Re=\text{const}} = -\frac{2}{D}\,\mathcal{E}[f]$$

  当 $Re$ 同时变化时（例如 $U_{in}$ 固定而 $D$ 改变），完整的敏感度表达为：
  $$\frac{\partial u^*}{\partial R}\bigg|_{U_{in}=\text{const}} = -\frac{2}{D}\,\mathcal{E}[f] \;+\; \frac{\partial u^*}{\partial Re}\cdot\frac{\partial Re}{\partial R}$$

  后一项为 Reynolds 数效应修正项，在 $\delta R/R \ll 1$ 时为高阶小量。
  本文聚焦于第一项（几何缩放效应），第二项作为已知近似误差在 Discussion 中讨论。

  这一近似与伴随灵敏度分析中广泛使用的"冻结湍流假设"(frozen turbulence assumption)
  类似——已知存在误差，但在工程实践中被接受为有效的一阶近似。

  **假设 H2：局部光滑性与空间代表性 (Smoothness Assumption)**

  在 SR 拟合域（本文为 3D~5D 尾流区）内，时均流场在空间上足够光滑，
  使得 SR 公式的空间梯度具有物理代表性，而非数值噪声或过拟合的产物。

  验证方式：
  - 不同复杂度公式的积分敏感度 $\bar{S}_R$ 在 complexity > 8 后收敛到约 30（Fig 09）
  - Strategy A（标准 SR）和 Strategy B（散度约束 SR）给出接近的 $\bar{S}_R$
  - 散度过滤确保公式满足质量守恒约束

  **方法适用范围**

  | 适用 | 不适用 |
  |------|--------|
  | 固定 Re 下的几何形状优化 | 跨流态变化（层流→湍流转捩） |
  | 小扰动范围 $\delta R/R \ll 1$ | 大幅度几何变化导致拓扑改变 |
  | 时均流场稳态或周期稳态 | 强非定常流（如湍流突变） |
  | 几何参数可映射为坐标缩放 | 参数与坐标关系非线性（如角度参数） |

  ---
  2. 核心逻辑链 (The "Single-Sim" Bridge)

  这是审稿人必问的问题： "只做一次仿真，你凭什么知道改了参数 H 后流场会变？"
  你的回答逻辑：
  1. 空间-参数等价性 (Spatial-Parameter Equivalence)：对于参数化设计的流体器件，几何参数（如高度 H）的变化往往体现在空间坐标（如 z 轴）的缩放或场分布的平移。**关键前提：此等价性在固定 Re 下严格成立（假设 H1），因为无量纲流场 $u^*(\xi,\eta;Re)$ 在 Re 不变时仅通过坐标缩放响应几何变化。**
  2. 局部梯度分析：单次仿真流场中的空间梯度 $\nabla v(x,y,z)$ 包含了边界层和流道变化的完整信息，这些信息在数学上与设计参数的灵敏度存在映射关系。
  3. PISR 约束：通过在符号回归中加入物理约束（如质量守恒），确保拟合出的公式不是随机凑出来的，而是符合物理趋势的解析解。

  **Euler 算子映射 (The Euler Operator Bridge)**

  对于无量纲化公式 $u^* = f(\xi, \eta)$，其中 $\xi = x/D$, $\eta = y/D$, $D = 2R$：

  Step 3a: 定义 Euler 齐次算子
  $$\mathcal{E}[f] = \xi\frac{\partial f}{\partial \xi} + \eta\frac{\partial f}{\partial \eta}$$

  Step 3b: 由链式法则推导几何敏感度
  $$\frac{\partial u^*}{\partial D} = \frac{\partial f}{\partial \xi}\cdot\left(-\frac{\xi}{D}\right) + \frac{\partial f}{\partial \eta}\cdot\left(-\frac{\eta}{D}\right) = -\frac{1}{D}\mathcal{E}[f]$$

  Step 3c: 映射到圆柱半径
  $$S_R = \frac{\partial u^*}{\partial R}\bigg|_{Re=\text{const}} = 2\frac{\partial u^*}{\partial D} = -\frac{2}{D}\mathcal{E}[f]$$

  Step 3d: 积分得到全局敏感度系数
  $$\bar{S}_R = \frac{1}{|\Omega|}\int_\Omega |S_R|\,d\Omega$$

  **特殊情况：当 SR 公式仅依赖 $\eta$（如圆柱尾流区）**

  此时 $\partial f/\partial\xi = 0$，Euler 算子简化为：
  $$\mathcal{E}[f] = \eta\frac{df}{d\eta}, \quad S_R = -\frac{2}{D}\eta\frac{df}{d\eta}$$

  物理含义：
  - $S_R = 0$ 在 $\eta = 0$（尾流中心线）：对称性要求，中心线速度对 R 不敏感
  - $|S_R|$ 峰值在中间 $\eta$（剪切层区域）：尾流宽度 $\propto D$，该区域对 R 最敏感

  ---
  3. 方法论步骤 (Methodology)
  Step 1: 关键流场点集提取 (Single CFD & Sampling)

  - 输入：进行一次基准设计 $(H_0, R_0, T_0, L_0)$ 的 CFD 仿真。
  - 输出：在目标表面/截面提取高密度点集 $P = {(x_i, y_i, z_i, v_i)}$。
  - 亮点：通过分析网格敏感度，确保单次仿真的点集足以覆盖流场特征。

  Step 2: 物理约束符号回归 (PISR) —— 这是你的 AI 贡献

  - 算法选择：基于 PySR 框架，自定义 Loss 函数。
  - 自定义 Loss：
  $$Loss = \text{MSE}{\text{data}} + \lambda_1 L{\text{phys}} + \lambda_2 L_{\text{dim}}$$
    - $L_{\text{phys}}$: 惩罚违反流体物理律（如散度 $\nabla \cdot \mathbf{v} \approx 0$）的公式。
    - $L_{\text{dim}}$: 惩罚量纲不匹配的组合（利用 Buckingham $\pi$ 定理）。
  - 结果：得到解析公式 $v = f(x, y, z; \mathbf{C})$，其中 $\mathbf{C}$ 是回归系数矩阵。

  Step 3: Euler 算子敏感度映射 (Euler Operator Sensitivity Mapping)

  - 直接参数映射：对 SR 公式应用 Euler 齐次算子 $\mathcal{E}[f]$，通过 SymPy 自动微分。
    - $S_R = -\frac{2}{D}\mathcal{E}[f]$（在假设 H1 下严格成立）
  - 间接参数映射 (针对非缩放类参数)：提取局部点集，分析公式中关于特定空间特征的权重。
  - 输出：参数影响系数矩阵 $\mathbf{S} = [S_H, S_R, S_T, S_L]$，并进行归一化排序。

  ---
  4. 实验与验证 (Validation)

  - 基准对比 (Ground Truth)：使用你手头已有的 LHS 基准数据（如 100 次仿真的 Sobol 指数）。
  - 内部一致性验证 (Internal Consistency)：
    a. 敏感度收敛性：不同复杂度公式的 $\bar{S}_R$ 是否收敛（Fig 09）。
    b. 策略一致性：Strategy A 与 B 的敏感度排序是否一致。
    c. 物理一致性：$S_R = 0$ 在中心线、$|S_R|$ 峰值在剪切层是否符合预期。
  - 评价指标：
    a. 排序一致性：计算新方法与 LHS 方法的 Spearman 秩相关系数。
    b. 物理一致性：展示 PISR 得到的公式是否比纯 SR 更符合流体力学基本规律。
    c. 效率提升倍数：计算时间对比（1 次 vs 100 次）。

  ---
  5. 给 AI 学生的可解释性加分项 (The "AI-Explainability" Story)

  你要在论文中强调：
  - By-design Interpretability：我们不是在训练黑箱，而是在自动发现描述该流场的物理定律。
  - White-box Ranking：相比 SHAP 或 LIME 等事后解释方法，我们的排序是直接基于解析公式的偏导数，具有数学上的精确性。

  ---
  6. 方法局限性与讨论要点 (Limitations & Discussion)

  **6.1 固定 Re 假设的影响**

  本方法在固定 Re 下严格成立。当 $U_{in}$ 固定而 $R$ 变化时，$Re$ 同步变化，
  引入额外的 $\partial u^*/\partial Re$ 项。对于圆柱绕流在 $Re \approx 100$ 附近：
  - $\pm 10\%$ 的 $R$ 变化对应 $\pm 10\%$ 的 $Re$ 变化
  - 在层流涡脱落稳定区间（$Re \in [50, 150]$），流动定性特征不变
  - 因此 Re 效应为高阶修正，不改变敏感度的定性排序

  **6.2 SR 公式仅依赖 y/D 的含义**

  在 3D~5D 尾流区，时均速度主要由横向剪切层结构主导，
  x 方向变化很弱（约 2D 跨度内）。这是物理现象而非拟合缺陷。
  但这意味着 Euler 算子的 $\xi\partial f/\partial\xi$ 项为零，
  方法仅捕捉了尾流横向结构对 R 的敏感度，未捕捉纵向扩展效应。

  **6.3 方法不适用的情况**

  - 跨流态变化：$R$ 变化导致 $Re$ 跨越临界值时（如层流→湍流转捩），
    流场发生质的改变，空间-参数等价性完全失效
  - 几何拓扑变化：当参数变化导致流道拓扑改变（如通道闭合/分叉）时，
    坐标缩放假设不再成立
  - 强非定常流：当流场的非定常特征（涡脱落频率、振幅）对参数高度敏感时，
    时均场的梯度可能不足以代表瞬态敏感度

