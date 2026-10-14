# GENCO dynamic operating envelopes, case33bw

Model run: `~/gencodoe_work/mlruns/282115863240368177/6b52e3cac0fb4d9f8d9dc5497fe46549`; device: mps; 500 scenarios per interval; eps = 0.05; conformal alpha = 0.01; GENCO branch flows from injections.

## 1. Power flow accuracy on fresh envelope-like rows

| Quantity | MAE | 99th pct abs error | max abs error |
|---|---|---|---|
| Vm (pu) | 0.00653 | 0.03130 | 0.07760 |
| Va (deg) | 0.34923 | 1.43229 | 3.09141 |
| Branch loading | 0.00268 | 0.01630 | 0.05816 |
| Branch loading, flows from voltage differences | 1.23516 | 13.00456 | 65.46422 |

| True Vm bin | Vm MAE | count |
|---|---|---|
| [0.8, 0.94) | 0.00902 | 23627 |
| [0.94, 0.97) | 0.00573 | 33630 |
| [0.97, 1.0) | 0.00509 | 61670 |
| [1.0, 1.03) | 0.00727 | 29019 |
| [1.03, 1.06) | 0.00845 | 9973 |
| [1.06, 1.1) | 0.01466 | 2081 |
| [1.1, 1.2) | nan | 0 |

| True loading bin | Loading MAE | count |
|---|---|---|
| [0, 0.5) | 0.00152 | 88566 |
| [0.5, 0.9) | 0.00356 | 58923 |
| [0.9, 1.0) | 0.00601 | 6916 |
| [1.0, 1.1) | 0.00723 | 3464 |
| [1.1, inf) | 0.00823 | 2131 |

## 2. Conformal margins and screening

q_v = 0.0316 pu, q_l = 0.0167. Coverage on test rows: Vm 0.9903, loading 0.9908 (target 0.99).

| Direction | Sent to AC | Missed violations | False alarms |
|---|---|---|---|
| export | 0.021 | 0.0000 | 0.0000 |
| import | 0.525 | 0.0000 | 0.0000 |

## 3. Envelopes over 24 hourly intervals (proportional policy)

| Method | Direction | Mean abs gap vs AC (total) | Worst gap | Mean P(ok) | Min P(ok) | AC power flows | Search time s |
|---|---|---|---|---|---|---|---|
| AC power flow | export | 0.00% | +0.00% | 0.980 | 0.944 | 114,000 | 0.3 |
| AC power flow | import | 0.00% | +0.00% | 0.947 | 0.924 | 184,000 | 0.5 |
| LinDistFlow | export | 0.09% | -0.32% | 0.982 | 0.948 | 0 | 0.0 |
| LinDistFlow | import | 30.19% | +270.97% | 0.766 | 0.584 | 0 | 0.1 |
| GENCO raw | export | 0.05% | +0.27% | 0.979 | 0.938 | 0 | 219.0 |
| GENCO raw | import | 40.05% | -100.00% | 0.995 | 0.966 | 0 | 295.6 |
| GENCO + conformal + AC verify | export | 0.00% | +0.00% | 0.980 | 0.944 | 23,491 | 218.8 |
| GENCO + conformal + AC verify | import | 0.00% | +0.00% | 0.947 | 0.924 | 166,682 | 355.1 |
| GENCO + conformal only | export | 20.34% | -33.77% | 1.000 | 1.000 | 0 | 465.7 |
| GENCO + conformal only | import | 66.67% | -100.00% | 0.997 | 0.966 | 0 | 46.6 |

P(ok): share of fresh scenarios in which AC power flow finds no violation at the envelope (target >= 0.95). Intervals where the feeder already violates with zero flexible exchange (envelope 0 for every method) are excluded from P(ok): 0 export and 8 import intervals.

<details><summary>Per-hour totals (pu)</summary>

| Hour | Direction | Method | Total | Gap | P(ok) |
|---|---|---|---|---|---|
| 0 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 0 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 0 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 0 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 0 | export | GENCO + conformal only | 5.682 | -23.52% | 1.000 |
| 0 | import | AC power flow | 0.546 | +0.00% | 0.964 |
| 0 | import | LinDistFlow | 0.742 | +35.95% | 0.718 |
| 0 | import | GENCO raw | 0.037 | -93.25% | 1.000 |
| 0 | import | GENCO + conformal + AC verify | 0.546 | +0.00% | 0.964 |
| 0 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 1 | export | AC power flow | 7.430 | +0.00% | 0.998 |
| 1 | export | LinDistFlow | 7.430 | +0.00% | 0.998 |
| 1 | export | GENCO raw | 7.430 | +0.00% | 0.998 |
| 1 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 0.998 |
| 1 | export | GENCO + conformal only | 5.609 | -24.51% | 1.000 |
| 1 | import | AC power flow | 0.838 | +0.00% | 0.954 |
| 1 | import | LinDistFlow | 1.034 | +23.40% | 0.662 |
| 1 | import | GENCO raw | 0.315 | -62.41% | 1.000 |
| 1 | import | GENCO + conformal + AC verify | 0.838 | +0.00% | 0.954 |
| 1 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 2 | export | AC power flow | 7.430 | +0.00% | 0.988 |
| 2 | export | LinDistFlow | 7.430 | +0.00% | 0.988 |
| 2 | export | GENCO raw | 7.430 | +0.00% | 0.988 |
| 2 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 0.988 |
| 2 | export | GENCO + conformal only | 5.597 | -24.67% | 1.000 |
| 2 | import | AC power flow | 0.957 | +0.00% | 0.946 |
| 2 | import | LinDistFlow | 1.154 | +20.62% | 0.606 |
| 2 | import | GENCO raw | 0.397 | -58.51% | 1.000 |
| 2 | import | GENCO + conformal + AC verify | 0.957 | +0.00% | 0.946 |
| 2 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 3 | export | AC power flow | 7.430 | +0.00% | 0.966 |
| 3 | export | LinDistFlow | 7.430 | +0.00% | 0.966 |
| 3 | export | GENCO raw | 7.430 | +0.00% | 0.966 |
| 3 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 0.966 |
| 3 | export | GENCO + conformal only | 5.602 | -24.61% | 1.000 |
| 3 | import | AC power flow | 1.031 | +0.00% | 0.936 |
| 3 | import | LinDistFlow | 1.228 | +19.15% | 0.590 |
| 3 | import | GENCO raw | 0.428 | -58.48% | 1.000 |
| 3 | import | GENCO + conformal + AC verify | 1.031 | +0.00% | 0.936 |
| 3 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 4 | export | AC power flow | 7.430 | +0.00% | 0.982 |
| 4 | export | LinDistFlow | 7.430 | +0.00% | 0.982 |
| 4 | export | GENCO raw | 7.430 | +0.00% | 0.982 |
| 4 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 0.982 |
| 4 | export | GENCO + conformal only | 5.628 | -24.26% | 1.000 |
| 4 | import | AC power flow | 0.949 | +0.00% | 0.942 |
| 4 | import | LinDistFlow | 1.146 | +20.80% | 0.584 |
| 4 | import | GENCO raw | 0.374 | -60.53% | 1.000 |
| 4 | import | GENCO + conformal + AC verify | 0.949 | +0.00% | 0.942 |
| 4 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 5 | export | AC power flow | 7.430 | +0.00% | 0.998 |
| 5 | export | LinDistFlow | 7.430 | +0.00% | 0.998 |
| 5 | export | GENCO raw | 7.430 | +0.00% | 0.998 |
| 5 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 0.998 |
| 5 | export | GENCO + conformal only | 5.697 | -23.33% | 1.000 |
| 5 | import | AC power flow | 0.559 | +0.00% | 0.946 |
| 5 | import | LinDistFlow | 0.755 | +35.11% | 0.636 |
| 5 | import | GENCO raw | 0.052 | -90.64% | 1.000 |
| 5 | import | GENCO + conformal + AC verify | 0.559 | +0.00% | 0.946 |
| 5 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 6 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 6 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 6 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 6 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 6 | export | GENCO + conformal only | 6.258 | -15.78% | 1.000 |
| 6 | import | AC power flow | 0.000 | +0.00% | nan |
| 6 | import | LinDistFlow | 0.000 | +0.00% | nan |
| 6 | import | GENCO raw | 0.000 | +0.00% | nan |
| 6 | import | GENCO + conformal + AC verify | 0.000 | +0.00% | nan |
| 6 | import | GENCO + conformal only | 0.000 | +0.00% | nan |
| 7 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 7 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 7 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 7 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 7 | export | GENCO + conformal only | 6.529 | -12.13% | 1.000 |
| 7 | import | AC power flow | 0.000 | +0.00% | nan |
| 7 | import | LinDistFlow | 0.000 | +0.00% | nan |
| 7 | import | GENCO raw | 0.000 | +0.00% | nan |
| 7 | import | GENCO + conformal + AC verify | 0.000 | +0.00% | nan |
| 7 | import | GENCO + conformal only | 0.000 | +0.00% | nan |
| 8 | export | AC power flow | 7.283 | +0.00% | 0.952 |
| 8 | export | LinDistFlow | 7.266 | -0.23% | 0.956 |
| 8 | export | GENCO raw | 7.280 | -0.03% | 0.952 |
| 8 | export | GENCO + conformal + AC verify | 7.283 | +0.00% | 0.952 |
| 8 | export | GENCO + conformal only | 5.609 | -22.98% | 1.000 |
| 8 | import | AC power flow | 0.160 | +0.00% | 0.948 |
| 8 | import | LinDistFlow | 0.360 | +124.44% | 0.790 |
| 8 | import | GENCO raw | 0.000 | -100.00% | 0.986 |
| 8 | import | GENCO + conformal + AC verify | 0.160 | +0.00% | 0.948 |
| 8 | import | GENCO + conformal only | 0.000 | -100.00% | 0.986 |
| 9 | export | AC power flow | 6.481 | +0.00% | 0.948 |
| 9 | export | LinDistFlow | 6.472 | -0.15% | 0.950 |
| 9 | export | GENCO raw | 6.488 | +0.11% | 0.946 |
| 9 | export | GENCO + conformal + AC verify | 6.481 | +0.00% | 0.948 |
| 9 | export | GENCO + conformal only | 4.722 | -27.15% | 1.000 |
| 9 | import | AC power flow | 0.993 | +0.00% | 0.936 |
| 9 | import | LinDistFlow | 1.190 | +19.88% | 0.800 |
| 9 | import | GENCO raw | 0.478 | -51.86% | 1.000 |
| 9 | import | GENCO + conformal + AC verify | 0.993 | +0.00% | 0.936 |
| 9 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 10 | export | AC power flow | 6.091 | +0.00% | 0.966 |
| 10 | export | LinDistFlow | 6.072 | -0.31% | 0.974 |
| 10 | export | GENCO raw | 6.101 | +0.16% | 0.962 |
| 10 | export | GENCO + conformal + AC verify | 6.091 | +0.00% | 0.966 |
| 10 | export | GENCO + conformal only | 4.151 | -31.85% | 1.000 |
| 10 | import | AC power flow | 1.447 | +0.00% | 0.946 |
| 10 | import | LinDistFlow | 1.643 | +13.56% | 0.862 |
| 10 | import | GENCO raw | 0.968 | -33.11% | 0.996 |
| 10 | import | GENCO + conformal + AC verify | 1.447 | +0.00% | 0.946 |
| 10 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 11 | export | AC power flow | 6.015 | +0.00% | 0.952 |
| 11 | export | LinDistFlow | 5.996 | -0.32% | 0.956 |
| 11 | export | GENCO raw | 6.018 | +0.04% | 0.950 |
| 11 | export | GENCO + conformal + AC verify | 6.015 | +0.00% | 0.952 |
| 11 | export | GENCO + conformal only | 4.011 | -33.32% | 1.000 |
| 11 | import | AC power flow | 1.667 | +0.00% | 0.944 |
| 11 | import | LinDistFlow | 1.864 | +11.84% | 0.880 |
| 11 | import | GENCO raw | 1.251 | -24.96% | 0.996 |
| 11 | import | GENCO + conformal + AC verify | 1.667 | +0.00% | 0.944 |
| 11 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 12 | export | AC power flow | 5.942 | +0.00% | 0.954 |
| 12 | export | LinDistFlow | 5.925 | -0.28% | 0.958 |
| 12 | export | GENCO raw | 5.956 | +0.24% | 0.948 |
| 12 | export | GENCO + conformal + AC verify | 5.942 | +0.00% | 0.954 |
| 12 | export | GENCO + conformal only | 3.935 | -33.77% | 1.000 |
| 12 | import | AC power flow | 1.765 | +0.00% | 0.956 |
| 12 | import | LinDistFlow | 1.963 | +11.18% | 0.890 |
| 12 | import | GENCO raw | 1.381 | -21.75% | 0.990 |
| 12 | import | GENCO + conformal + AC verify | 1.765 | +0.00% | 0.956 |
| 12 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 13 | export | AC power flow | 5.944 | +0.00% | 0.968 |
| 13 | export | LinDistFlow | 5.925 | -0.32% | 0.978 |
| 13 | export | GENCO raw | 5.956 | +0.20% | 0.960 |
| 13 | export | GENCO + conformal + AC verify | 5.944 | +0.00% | 0.968 |
| 13 | export | GENCO + conformal only | 3.980 | -33.04% | 1.000 |
| 13 | import | AC power flow | 1.712 | +0.00% | 0.962 |
| 13 | import | LinDistFlow | 1.909 | +11.53% | 0.912 |
| 13 | import | GENCO raw | 1.299 | -24.10% | 0.992 |
| 13 | import | GENCO + conformal + AC verify | 1.712 | +0.00% | 0.962 |
| 13 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 14 | export | AC power flow | 6.056 | +0.00% | 0.956 |
| 14 | export | LinDistFlow | 6.046 | -0.16% | 0.964 |
| 14 | export | GENCO raw | 6.063 | +0.12% | 0.954 |
| 14 | export | GENCO + conformal + AC verify | 6.056 | +0.00% | 0.956 |
| 14 | export | GENCO + conformal only | 4.118 | -32.00% | 1.000 |
| 14 | import | AC power flow | 1.574 | +0.00% | 0.946 |
| 14 | import | LinDistFlow | 1.770 | +12.46% | 0.892 |
| 14 | import | GENCO raw | 1.121 | -28.78% | 1.000 |
| 14 | import | GENCO + conformal + AC verify | 1.574 | +0.00% | 0.946 |
| 14 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 15 | export | AC power flow | 6.372 | +0.00% | 0.944 |
| 15 | export | LinDistFlow | 6.353 | -0.30% | 0.948 |
| 15 | export | GENCO raw | 6.370 | -0.04% | 0.944 |
| 15 | export | GENCO + conformal + AC verify | 6.372 | +0.00% | 0.944 |
| 15 | export | GENCO + conformal only | 4.510 | -29.22% | 1.000 |
| 15 | import | AC power flow | 1.044 | +0.00% | 0.924 |
| 15 | import | LinDistFlow | 1.241 | +18.91% | 0.846 |
| 15 | import | GENCO raw | 0.493 | -52.73% | 1.000 |
| 15 | import | GENCO + conformal + AC verify | 1.044 | +0.00% | 0.924 |
| 15 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |
| 16 | export | AC power flow | 6.974 | +0.00% | 0.952 |
| 16 | export | LinDistFlow | 6.959 | -0.20% | 0.956 |
| 16 | export | GENCO raw | 6.993 | +0.27% | 0.938 |
| 16 | export | GENCO + conformal + AC verify | 6.974 | +0.00% | 0.952 |
| 16 | export | GENCO + conformal only | 5.587 | -19.88% | 1.000 |
| 16 | import | AC power flow | 0.074 | +0.00% | 0.962 |
| 16 | import | LinDistFlow | 0.273 | +270.97% | 0.872 |
| 16 | import | GENCO raw | 0.000 | -100.00% | 0.966 |
| 16 | import | GENCO + conformal + AC verify | 0.074 | +0.00% | 0.962 |
| 16 | import | GENCO + conformal only | 0.000 | -100.00% | 0.966 |
| 17 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 17 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 17 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 17 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 17 | export | GENCO + conformal only | 6.850 | -7.81% | 1.000 |
| 17 | import | AC power flow | 0.000 | +0.00% | nan |
| 17 | import | LinDistFlow | 0.000 | +0.00% | nan |
| 17 | import | GENCO raw | 0.000 | +0.00% | nan |
| 17 | import | GENCO + conformal + AC verify | 0.000 | +0.00% | nan |
| 17 | import | GENCO + conformal only | 0.000 | +0.00% | nan |
| 18 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 18 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 18 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 18 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 18 | export | GENCO + conformal only | 7.390 | -0.54% | 1.000 |
| 18 | import | AC power flow | 0.000 | +0.00% | nan |
| 18 | import | LinDistFlow | 0.000 | +0.00% | nan |
| 18 | import | GENCO raw | 0.000 | +0.00% | nan |
| 18 | import | GENCO + conformal + AC verify | 0.000 | +0.00% | nan |
| 18 | import | GENCO + conformal only | 0.000 | +0.00% | nan |
| 19 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 19 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 19 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 19 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 19 | export | GENCO + conformal only | 7.430 | +0.00% | 1.000 |
| 19 | import | AC power flow | 0.000 | +0.00% | nan |
| 19 | import | LinDistFlow | 0.000 | +0.00% | nan |
| 19 | import | GENCO raw | 0.000 | +0.00% | nan |
| 19 | import | GENCO + conformal + AC verify | 0.000 | +0.00% | nan |
| 19 | import | GENCO + conformal only | 0.000 | +0.00% | nan |
| 20 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 20 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 20 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 20 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 20 | export | GENCO + conformal only | 7.430 | +0.00% | 1.000 |
| 20 | import | AC power flow | 0.000 | +0.00% | nan |
| 20 | import | LinDistFlow | 0.000 | +0.00% | nan |
| 20 | import | GENCO raw | 0.000 | +0.00% | nan |
| 20 | import | GENCO + conformal + AC verify | 0.000 | +0.00% | nan |
| 20 | import | GENCO + conformal only | 0.000 | +0.00% | nan |
| 21 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 21 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 21 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 21 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 21 | export | GENCO + conformal only | 7.012 | -5.63% | 1.000 |
| 21 | import | AC power flow | 0.000 | +0.00% | nan |
| 21 | import | LinDistFlow | 0.000 | +0.00% | nan |
| 21 | import | GENCO raw | 0.000 | +0.00% | nan |
| 21 | import | GENCO + conformal + AC verify | 0.000 | +0.00% | nan |
| 21 | import | GENCO + conformal only | 0.000 | +0.00% | nan |
| 22 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 22 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 22 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 22 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 22 | export | GENCO + conformal only | 6.227 | -16.19% | 1.000 |
| 22 | import | AC power flow | 0.000 | +0.00% | nan |
| 22 | import | LinDistFlow | 0.000 | +0.00% | nan |
| 22 | import | GENCO raw | 0.000 | +0.00% | nan |
| 22 | import | GENCO + conformal + AC verify | 0.000 | +0.00% | nan |
| 22 | import | GENCO + conformal only | 0.000 | +0.00% | nan |
| 23 | export | AC power flow | 7.430 | +0.00% | 1.000 |
| 23 | export | LinDistFlow | 7.430 | +0.00% | 1.000 |
| 23 | export | GENCO raw | 7.430 | +0.00% | 1.000 |
| 23 | export | GENCO + conformal + AC verify | 7.430 | +0.00% | 1.000 |
| 23 | export | GENCO + conformal only | 5.789 | -22.08% | 1.000 |
| 23 | import | AC power flow | 0.263 | +0.00% | 0.942 |
| 23 | import | LinDistFlow | 0.459 | +74.66% | 0.714 |
| 23 | import | GENCO raw | 0.000 | -100.00% | 1.000 |
| 23 | import | GENCO + conformal + AC verify | 0.263 | +0.00% | 0.942 |
| 23 | import | GENCO + conformal only | 0.000 | -100.00% | 1.000 |

</details>

## 4. Runtime per power flow

| Solver | µs per power flow |
|---|---|
| GENCO (mps, batched) | 1991.2 |
| AC backward/forward sweep (numpy, batched) | 4.9 |
| LinDistFlow (numpy, batched) | 0.60 |
| pandapower Newton-Raphson (one at a time) | 28372 |
