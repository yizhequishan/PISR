# 从现有硬边界 PINN 继续优化：冻结符号残差权重

本实验复用 `outputs_pinn/cylinder/hard_boundary_refined_20261013/cylinder_poiseuille/checkpoint_last.pt`。该模型是 Re=20 稳态硬边界流函数 PINN；不适用涡脱落频率、POD 时间动力学或长期相位漂移分析。独立 CFD 参考场尚未提供，本实验衡量原 Navier–Stokes 方程和边界的满足程度，不能据此认证流场或升阻力精度。

## 诊断与方法

现有网格与新开发随机点均将主要动量残差定位在入口附近。连续性、入口/固壁无滑移、总通量已由原有流函数构造保证。符号回归不修改控制方程，也不把旧 COMSOL 时均尾迹式当作 Re=20 的真值。

从冻结的原 checkpoint 在新开发点计算

```text
energy = r_x^2 + r_y^2
target = log1p(energy / median_fit_energy)
x_D = x
y_D = 2y/H - 1
```

PySR 仅拟合这个空间评分，算子为 `+,-,*,square,exp,tanh`，不含除法。选式使用另一批独立开发点，选择接近最小选式误差的简单表达式。入选式必须优于常数预测并与残差有正排序相关性。选式点属于开发数据，不作为最终测试集。

表达式冻结后，在每个训练批次计算

```text
score = clip(expression(x_D,y_D), 0, fit_target_99th_percentile)
raw_weight = 0.5 + 0.5 * clip(score / mean_fit_score, 0, 10)
weight = raw_weight / batch_mean(raw_weight)
```

仅两个动量分量的损失变为 `mean(weight * residual^2)`，全部边界项和权重不变。权重不参与反向传播；L-BFGS 全阶段固定采样点和权重。这里得到的是优化权重，不是新的物理规律或对正确解的额外观测。

## 三组对照

| 组 | 空间权重 | 起点 |
|---|---|---|
| ordinary | 1 | 同一原 checkpoint |
| inlet | `normalize(1+4 exp(-x^2))` | 同一原 checkpoint |
| sr | 冻结符号评分生成的正权重 | 同一原 checkpoint |

三组均保留原来的 lifting、D、圆柱常数处理、4×64 网络、Re、几何、float64 和近圆柱采样比例。优化器重置，共用新的采样种子和批次序列；保存坐标哈希以验证一致性。统一采用 CPU 单线程、学习率 `1e-4`、100 步 Adam 后 L-BFGS，达到预算时停止。因此更新次数可能不同。

默认每组总墙钟预算 300 秒。SR 组扣除残差标注、Julia/PySR 启动/搜索及选式成本，其余组将该时间用于续训。计时包含模型加载和训练初始化；停止发生在优化器步边界，单次 L-BFGS 线搜索和最终保存可能造成小幅超时，实际用时写入 `arm_protocol.json`。这是同机墙钟预算比较，不是严格相同 FLOPs/CPU 时间；Julia 启动的线程配置与神经续训不同。公共独立评估不计入优化预算。

所有评价使用冻结表达式后才访问的同一套新 20000 个全域均匀内部点，完全不带训练权重。额外检查局部区域动量 RMS、出口牵引、速度边界误差和分段 Gauss 通量。SR 对 ordinary 的预定判据为两个全局动量分量都改善、出口不变差、硬速度边界和通量保持；同时单独比较简单入口加权，不能把一般局部加权收益归功于 SR 独有能力。

## 复现

本机 PySR 使用 `deeplearning` 解释器与现有 Julia 安装。搜索通过独立 Python 子进程运行，避免在同一进程先导入 Torch 再初始化 Julia。启动器明确设置 `PYTHON_JULIAPKG_PROJECT` 和 `PYTHON_JULIAPKG_EXE`，现有旧版本不能仅靠 `JULIA_PROJECT` 正确找到该安装。

```powershell
.\scripts\Run-PinnSRContinuation.ps1 `
  -OutDir outputs_pinn/cylinder/new_sr_residual_trial `
  -TotalSeconds 300 -SearchSeconds 60
```

输出目录必须为新目录。`prepare`、`run --kind ordinary|inlet|sr` 和 `summarize` 可分阶段调用，入口为 `scripts/pinn_sr_residual_trial.py`。首次 Julia 编译可能耗尽预算，此时程序不会偷偷给 SR 额外训练时间；应为整个新对照预先设置更长、相同的总预算。

核心产物：`protocol.json`（来源与协议）、`development.npz`（开发集）、`selection.json`（冻结式）、每组 checkpoint/时间/采样哈希/审计、`comparison.json`（比较），以及 `figures/` 下的图。原 checkpoint 和既有基线目录不被修改。

通用接口为 `train(..., batch_transform=None, max_seconds=None)`；批次可提供 `(N,1)` 的冻结正有限 `interior_weights`。训练终止标记 `budget_finished` 表示预算用完，不表示异常或物理收敛。最终评价使用最终 checkpoint，而不是可能早于终止时刻的最后一行训练日志。
