# 符号流函数与神经残差 PINN

后续的入口/壁面/圆柱硬边界对照基线见 [硬边界流函数](pinn_hard_boundary.md)。该实验先用解析 lifting 隔离软壁面漏流，不引入 SR。

2026-09-23 已完成三组同预算试验（各 1,500 Adam＋300 L-BFGS）：硬无散生效，但当前整流函数局部门控产生强过渡带误差，未显示 SR 额外收益。结果与诊断见 [试验报告](../reports/pinn_hybrid_trial_2026-09-23.md)。

三种架构共享同一套 Navier–Stokes、边界、观测、优化器和评估代码：

| `architecture` | 输出构造 | 连续性损失 |
|---|---|---|
| `uvp`（默认） | MLP 直接输出 `(u,v,p)` | 保留 |
| `streamfunction` | MLP 输出 `(ψ,p)`，速度由流函数求导 | 删除 |
| `hybrid_streamfunction` | `ψ = g(ξ,η)·[ψ_SR(ξ,η;c)−ψ_SR(ξ₀,η₀;c)] + Nθ(x,y[,t])` | 删除 |

`u=∂yψ`、`v=−∂xψ`；连续性仍在独立诊断中实际计算。这里的“删除”仅针对流函数架构的训练项，原基线和旧检查点行为保留。解析压力、牵引和黏性项不变。流函数需三阶空间导数计算动量残差；模型支持现有 `torch.no_grad()` 推理和 `torch.inference_mode()`。

## 坐标与门函数

PINN 输入使用绝对无量纲坐标 `x=x_dim/D, y=y_dim/D`；SR 使用圆柱中心坐标 `ξ=x−cylinder_x, η=y−cylinder_y`。默认圆心 `(2,2)`，SR 窗口对应绝对坐标 `[5,6.9]×[0.1,3.9]`，不是 `[3,4.9]×[−1.9,1.9]`。

若写成 `ψ_SR+wN` 并使 `w` 在窗口外衰减，窗口外会剩下 SR。为实现窗口外纯网络，本实现让门函数乘符号项。`g` 在可信窗口外严格为零；宽度为 `symbolic_taper` 的过渡带位于窗口**内部**。各轴使用七次 smoothstep：

```text
S(t) = 35t⁴ − 84t⁵ + 70t⁶ − 20t⁷,  t∈[0,1]
S(t≤0)=0, S(t≥1)=1
g = S((ξ−ξmin)/δξ) S((ξmax−ξ)/δξ)
    S((η−ηmin)/δη) S((ηmax−η)/δη)
```

该门函数全局 C³，端点前三阶导数为零。默认 `δξ=δη=0.4`，平台区为 `ξ∈[3.4,4.5]、η∈[−1.5,1.5]`。自动微分保留所有门函数导数，因此仍是无散速度场。直接对两组速度加权一般不能保证这一点。

流函数常数在乘门函数后会影响速度，所以先减去窗口中心 `(ξ₀,η₀)` 的 SR 值，固定符号项的 gauge；有时间坐标时按同一时刻减去。窗口外不求值原表达式；仅收集窗口内的点计算符号项，再通过可微分 scatter 写回，避免 `0×NaN` 污染梯度，也减少无用的高阶求导。

## SR 表达式及系数

示例配置 [cylinder_steady_hybrid.json](../configs/pinn/cylinder_steady_hybrid.json) 使用原记录 B-E14-C21；通过代数消去中间的可去奇点，改写为：

```text
ψ_SR = η + sin(η − η[cos(ωη)−a] / {ξ−b−d[cos(ωη)−a]} − phase)
ω=0.71782035, a=0.3375839, b=1.7832861,
d=0.68956524, phase=0.054244295
```

配置的 `symbolic_coefficients` 明确指定可训练符号；数值字面量、整数幂和表达式结构保持固定。Torch 编译器支持加、减、乘、除、固定数值幂，以及 `sin/cos/tanh/exp/log/sqrt/sinh/cosh/atan`；未知符号和不支持的算子会报错。字符串经 AST 白名单解析，不调用 `eval` 或 NumPy。

精修参数化为 `c=c_init+r·max(1,|c_init|)·tanh(raw_c)`，默认 `r=0.1`。这限制精修幅度；本配置分母在整个可信窗口及全部允许系数范围内大于 `0.4`。换表达式、窗口或增大精修范围时，必须重新验证其定义域，这个界限不适用于任意表达式。

源 SR 是 **COMSOL 时均场，Re 未记录**。本例只把其结构迁移到 Re=20 稳态试验，没有把这些均值当作稳态 NS 数据，也不假定旧 `U_in`、几何和物性与新算例匹配。源表达式提供局部结构先验，不构成全域预训练或精度保证。

## 正则与解释性

增加 `regularization_residual = mean(Nθ²)`，权重 `weight_regularization=1e-4`。在 SR 平台区单独取固定的 `12×12` 网格，避免全域均匀采样恰好没有足够的窗口点；不惩罚窗口外承担完整流场的网络。瞬态模式的时间取自当前训练批次，L-BFGS 固定批次期间目标保持固定。

这是残差范数惩罚，**没有实现或声称 OrthoReg 正交约束**。软惩罚并不能保证神经项不抵消符号项；试验额外输出窗口平台/过渡带的两分量速度范数、夹角和动量残差。流函数硬约束只保证无散，不能据此宣称梯度、动量或边界正确。

## 运行与重载

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
$python = 'D:\keyan\Anaconda\envs\deeplearning\python.exe'

# 完整混合模型配置：10000 Adam + 1000 L-BFGS
& $python train_pinn.py --config configs/pinn/cylinder_steady_hybrid.json --device cuda

# 三种架构、同采样和同更新次数；输出目录必须新建
& $python scripts/run_hybrid_pinn_trial.py `
  --out-dir outputs_pinn/cylinder/my_hybrid_trial --device cuda `
  --adam-steps 1500 --lbfgs-steps 300

# 既有接口支持混合模型重载及密集物理诊断
& $python train_pinn.py --evaluate outputs_pinn/cylinder/my_hybrid_trial/hybrid_streamfunction/checkpoint_last.pt `
  --out-dir outputs_pinn/cylinder/my_hybrid_reload --device cpu --no-plots
& $python scripts/audit_pinn_cylinder.py `
  --run-dir outputs_pinn/cylinder/my_hybrid_trial/hybrid_streamfunction --device cuda
```

三组使用相同随机种子、内部/边界点数、物理权重和隐藏层。纯流函数与混合流函数神经网络初值完全相同，混合模型多 5 个系数。原 u-v-p 输出头多一个通道，参数量略不同；同更新次数也不等于同计算时间或 FLOPs。应先比较纯流函数与混合流函数，再判断是否有符号先验的额外收益。

新增产物：`symbolic_coefficients.json` 保存初始/精修系数；`wake_audit.json` 检查过渡带；`comparison.json` 汇总三组。配置及 checkpoint 均包含架构、表达式、窗口和系数信息，不依赖运行时环境变量选择架构；旧 u-v-p checkpoint 仍可加载。

程序验证：`python -m pytest tests/test_pinn.py tests/test_pinn_streamfunction.py -q`。实验结果另记入报告，不将冒烟检查等同于物理收敛。
