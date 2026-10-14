# 真实尾流导数感知符号回归对照

分别拟合 u*、v*；模型的导数由表达式树解析求值，测试时独立用 SymPy 求导。
导数监督是数值 CFD 参考，不是真实连续解的精确导数。

## 固定协议

```json
{
  "seeds": [
    0
  ],
  "components": [
    "u"
  ],
  "niterations": 2,
  "populations": 2,
  "population_size": 40,
  "maxsize": 15,
  "ncycles_per_iteration": 30,
  "lambdas": [
    0.0,
    1.0
  ],
  "precision": 64,
  "parallelism": "serial",
  "operators": [
    "+",
    "-",
    "*",
    "/",
    "sin",
    "cos"
  ],
  "selection": [
    "normalized_value",
    "h1"
  ],
  "coordinate_shift": [
    4.0,
    0.0
  ]
}
```

训练/验证/测试点数：{'train': 992, 'validation': 186, 'test': 248}。原始列分区：{'train': [(0, 8), (13, 20), (26, 33)], 'validation': [(8, 13)], 'test': [(20, 26)]}（左闭右开）。
各分区端点各去掉一列/行；三点中心差分模板不跨分区，归一化尺度仅来自训练集。
连续测试块检验同一 CFD 算例内的空间插值；不代表跨几何泛化。
损失为 L=MSE(q)/Var_train(q)+λ/2·[MSE(qξ)/mean_train(qξ²)+MSE(qη)/mean_train(qη²)]，λ=0 或 1。
两组算子、复杂度、迭代数、种群和种子相同；解析导数增加每次求值成本，故不等同于相同耗时。
在独立验证块分别按值误差、H1选式并保存，再一次性评价测试块。主对照为 λ=0 按值选式与 λ=1 按 H1 选式。

## 测试结果（种子均值 ± 样本标准差）

|分量|训练|速度 MSE|ξ 导数相对 L2|η 导数相对 L2|合并梯度相对 L2|K_R 相对 L2|
|---|---|---:|---:|---:|---:|---:|
|u|仅值|0.02943 ± nan|100 ± nan%|63.02 ± nan%|63.94 ± nan%|66.81 ± nan%|
|u|值+导数|0.0284 ± nan|100 ± nan%|64.08 ± nan%|64.96 ± nan%|66.39 ± nan%|

## 配对变化（负值表示误差下降）

|分量|种子|速度 MSE 变化|梯度相对 L2 变化|K_R 相对 L2 变化|
|---|---:|---:|---:|---:|
|u|0|-3.47%|+1.60%|-0.63%|

`paired_changes.csv` 同时保存相同选式准则的交叉对照，区分搜索与选式的贡献。
`test_metrics.csv` 的 savgol5_stability 是分区内 5 点三阶 Savitzky–Golay 平滑后的导数参考敏感性检查，不参与训练、选式或调参。

K_R=−(ξ qξ+η qη)/R，此处评价坐标输运项相对数值梯度参考的误差；没有用它替代总参数响应 T=K+M。
这是直接速度符号回归，没有施加流函数无散硬约束。只有一份 CFD 快照和三个搜索种子，结果属探索性证据。
未把历史流函数模型与本轮不同训练协议直接作因果比较。

![测试对照](comparison.png)

## 复现

```powershell
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\experiments\wake_derivative_aware_sr.py --out-dir outputs/wake_derivative_aware_smoke_v2
```

数据、代码和协议指纹见 protocol.json；完整公式见各 fit 的 selection.json。
