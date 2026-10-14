# PINN-inspired symbolic derivatives, v3

The validation-selected model is `gpinn_continuity`, with 24 coefficients per
velocity component. Relative L2 test errors on the reused v2 spatial holdout:

| Metric | Previous u | Selected u | Previous v | Selected v |
|---|---:|---:|---:|---:|
| Combined gradient | 10.27% | 9.63% | 9.65% | 7.51% |
| Coordinate transport K_R | 8.83% | 8.21% | 9.39% | 6.68% |
| Streamwise derivative | 15.00% | 14.97% | 7.75% | 8.55% |

Velocity value errors increase slightly. The hard streamfunction candidate
performs better on this test block but was not selected by validation; the
recommendation remains frozen. This is continued development on one previously
studied CFD field, not an independent new-case benchmark.

Full methods, ablations, limitations and reproduction commands:
[Chinese report](../../reports/wake_pinn_informed_sr_v3_2026-09-29.md).

Use [selected_velocity_models.json](selected_velocity_models.json) for the
chosen formulas and coefficients. Coordinates are `x=xi-4, y=eta`, and velocities
are normalized by `U_in`. Only incompressibility was imposed; there is no
unclosed mean-momentum constraint. `K_R` is coordinate transport, not the full
parameter response `T=K+M`.
