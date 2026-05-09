# Calibration report (TASK-040 three-way)

- run_id: `20260505_234500_three_way`
- three_way artifacts: `artifacts\three_way_hybrid\20260505_232616_three_way_promoted`
- chosen: **isotonic** (min_ece_with_brier_tiebreak)

| | |
|---|---|
| ECE val_pick pre | 0.446364 |
| ECE val_pick Platt | 0.067076 |
| ECE val_pick Isotonic | 0.010196 |
| ECE val_fit post (chosen) | 0.000000 |
| ECE test (chosen) | 0.018548 |
| Brier test (chosen) | 0.216407 |
| device ladder | `[('cuda', 64, None)]` |
