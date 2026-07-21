# MLP-psi gradient audit

The audit itself does not train networks; it consumes the sensitivity fields from grid `dense_xi_v1` with shape `33 x 64`.

| model | u field MSE | epsilon_K vs S-M | epsilon_K vs data-K | kappa | predicted closure | actual closure |
|---|---:|---:|---:|---:|---:|---:|
| psi-B | 0.00395485 | 0.1687 | 0.1621 | 2.7171 | 0.4585 | 0.4585 |
| MLP-psi | 7.40083e-06 | 0.0558 | 0.0410 | 2.7171 | 0.1517 | 0.1517 |
| base-CFD data gradient | - | 0.0472 | 0.0000 | 2.7171 | 0.1282 | 0.1282 |

MLP field MSE is 534.4x (2.73 orders) lower than psi-B, but epsilon_K improves only 3.02x.
Across seeds, epsilon_K = 0.0564 +/- 0.0012; closure = 0.1533 +/- 0.0033.

The MLP gradient is already close to the base-CFD data-gradient floor, yet kappa amplification lifts its approximately 5.7% transport error to approximately 15.4% total-sensitivity error.
