# Radius-response stream-function experiment

## Selected symbolic response

Strategy B, complexity 25, selected by the minimum component-balanced derivative score over the PySR Pareto front.

```text
phi_R(xi, eta) = (11.504055 - x_D)*(y_D*(1.3596838 - y_D**2) + cos(x_D + (y_D + 0.38234538)**2) - cos(sin(sin(x_D))) + 0.19094007)
```

The response velocity is recovered structurally as
`M_R^u = d(phi_R)/deta`, `M_R^v = -d(phi_R)/dxi`. Its symbolic divergence simplifies exactly to `0`.

## Validation

| quantity | component | rel L2 | RMSE | correlation |
|---|---|---:|---:|---:|
| intrinsic response M | u | 0.2916 | 11.8767 | 0.9415 |
| intrinsic response M | v | 0.3921 | 2.9470 | 0.8715 |
| total K[psi0] + M[phi_R] | u | 0.8224 | 14.5741 | 0.6360 |
| total K[psi0] + M[phi_R] | v | 1.5924 | 13.8600 | 0.6853 |
| hybrid K[data] + M[phi_R] | u | 0.6948 | 12.3135 | 0.7346 |
| hybrid K[data] + M[phi_R] | v | 0.4452 | 3.8747 | 0.8805 |

The total-error row includes both the response-model error and the base psi-B transport-derivative error. The hybrid row replaces the latter with the matched-step base-CFD data gradient, thereby isolating the response-symbolization error.

For dimensional velocity, define `phi_R_dim = U0*phi_R + Uprime*psi0`; then `du/dR = U0*K_R[psi0] + d(phi_R_dim)/deta`, which contains the inlet-speed term explicitly and remains divergence-free by construction.
