# B+g substitution 实验归档

## 动机

Strategy B 在 `eta_sym` 窗口下的最优公式（c=18, MSE=0.0021, div_rms=0.026）包含 `dvdy` 作为输入变量。这偏离了 PISR "单次仿真 → 纯闭式公式" 的原始主张。

本实验尝试补救：额外用 PySR 回归一个代理公式 `g(x_D, y_D) ≈ dvdy`，再把 g 代入 Strategy B 的公式里，得到一个形式上纯 `(x_D, y_D)` 的闭式表达式，希望同时保留精度和物理约束。

## 为什么被移出主流程

放开 Strategy A 的 `maxsize` 从 25 到 45 后，发现 A 直接就能找到 ops=13、MSE=8.5e-4、div_rms=0.048 的公式（`exp(cos(...))` 形式，含 `-2.48/x_D` 远尾流 1/x 项）。这揭示：

1. B+g 的"精度优势"来自 A 被 `maxsize=25` 人为卡住，不是物理约束红利
2. 同一数据集训练 + 同一数据集评估，MSE 存在过拟合偏差（c=24 代入后 MSE 反低于原始版）
3. 展开复杂度（`sp.count_ops`）B+g c=18 实际为 41，而 A max=45 ops=13 精度更高
4. 代入 g 后物理约束成为假约束——g 没有几何扰动下的重标度规则
5. 嵌套 SR 缺少单阶段 SR 的 Pareto 保证

详细分析见 `报告.md` 附录。

## 文件清单

- `experiment_dvdy_surrogate.py` — B+g 主实验脚本（回归 g + 代入评估）
- `verify_substitution.py` — 计算展开复杂度（sp.count_ops）与代入后 div_rms
- `step2_a_maxsize45.py` — Strategy A 放开 maxsize=45 的一次性基线探索（证伪 B+g 的关键脚本）
- `compare_a45_vs_bg.py` — 三条 Pareto 对照（A max=25 vs A max=45 vs B+g）
- `Run-DvdySurrogateLocal.ps1` — 本地 PowerShell 启动器（路径已更新）

对应输出产物在 `../../outputs_mgn/eta_sym/archive_dvdy_substitution/`，包含 `a_maxsize45/` 子目录下的 A 基线 Pareto 模型与 CSV。

注：`compare_a45_vs_bg.py` 的路径常量已更新指向归档目录。`experiment_dvdy_surrogate.py` 的默认 `BASE_OUT_DIR` 指向主 `outputs_mgn/`，重跑时需要通过 `--out-dir` 显式指定归档路径或临时新目录。

## 保留理由

不删除是因为：

1. 论文附录需要"为什么不选 B+g"的实证对照
2. 如有审稿人问"为什么不把数据场特征作为 SR 输入"，直接指向这个实验
3. 未来如果尝试 Strategy B'（dvdy 进 loss、不进函数空间），本实验是必要的反例参考
