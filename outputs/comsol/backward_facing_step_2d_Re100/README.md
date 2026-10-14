# Backward-facing-step COMSOL model

Saved model on phy: `E:\CY\comsol\models\working\backward_facing_step_2d_Re100_solved.mph`

## Definition

- COMSOL 6.4; 2D steady incompressible Newtonian laminar flow.
- Step height h = 0.01 m; upstream height h; downstream height 2h.
- Upstream length 5h; downstream length 30h; step at x=0.
- Density 1000 kg/m^3; dynamic viscosity 0.001 Pa s.
- Inlet mean speed 0.01 m/s; parabolic peak speed 0.015 m/s.
- Re_h = rho Umean h / mu = 100; inlet hydraulic-diameter Reynolds number = 200.
- No-slip walls; outlet gauge pressure 0 Pa.
- Coordinate-based named selections and a graded triangular mesh.
- Stationary solver relative tolerance: 1e-6.
- Computed in an independent COMSOL process using 8 processor cores.

## Numerical checks

| Quantity | Value |
|---|---:|
| Final mesh elements | 42679 |
| Inlet flow per unit span, m^2/s | 9.99722222e-05 |
| Outlet flow per unit span, m^2/s | 9.98934195e-05 |
| Relative flow imbalance | 0.000788 |
| Minimum streamwise velocity, m/s | -0.0016980213 |
| Near-wall reattachment x/h | 4.888762 |
| Reattachment change between two meshes | 0.800% |

Reattachment is estimated from the negative-to-positive streamwise-velocity crossing
at y/h=0.001, checked against y/h=0.002. This is a near-wall estimate, not a direct
wall-shear zero calculation. The two-grid comparison checks mesh sensitivity;
it is not a formal convergence-order or independent benchmark validation.

## Files

- `backward_facing_step_overview.png`: full channel, recirculation, mesh comparison.
- `flow_field.csv`: x/h, y/h, u, v, and p sampled from the final COMSOL solution.
- `flow_field.npz`: the same field data in a compact NumPy archive.
- `nearwall_baseline.csv`, `nearwall_refined.csv`: near-wall velocity samples.
- `build_state.json`: parameters, mesh counts, diagnostics, and saved-model metadata.

The existing `three_cylinder_flow_2d` model was left unchanged.
