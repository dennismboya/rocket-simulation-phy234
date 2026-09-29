# FIXTURE model comparison (test fixture, not a result)

FIXTURE: every number in this file is made up for tests/test_report.py.

## Split (b): held-out context compositions (FIXTURE)

| model | family | n_params | held-out NLL/response [CI95] | Brier | AUC |
|---|---|---:|---|---|---|
| B1 | classical | 7 | 0.6931 [0.6900, 0.6960] | 0.2500 | 0.500 |
| Q2 | quantum | 6 | 0.6930 [0.6899, 0.6961] | 0.2500 | 0.501 |

Best classical: B1; best Q-model: Q2 (FIXTURE).
