# bf16 vs fp32 option readout, same trajectories (fp32_compare.py)

## huginn (bf16) vs huginn_fp32 (fp32 letters): 50 shared questions
- bf16 steps in the rerun vs the earlier run: max |diff| 0.00e+00 (must be 0)
- fp32 vs bf16 option logits, per-question max |diff|: median 0.062, max 0.062
- final answer (argmax at the last loop) agrees between readouts on 49/50

| quantity | bf16 readout | fp32 readout |
|---|---|---|
| leader changes per question | 5.540 [5.000, 6.060] | 5.200 [4.720, 5.660] |
| transitions with a side margin < one ulp | 195/277 | 156/260 |
| Cui-Ye event rate | 0.660 [0.520, 0.780] | 0.660 [0.520, 0.780] |
| (correct, top distractor) pair ever crosses | 0.900 [0.820, 0.980] | 0.920 [0.840, 0.980] |
| distractor pairs ever cross | 0.980 [0.953, 1.000] | 0.987 [0.967, 1.000] |
| last-change loop, mean (median) | 12.140 [10.260, 14.140] (10) | 11.440 [9.920, 13.100] (10) |
| answer at loop 8 differs from final | 25/50 | 25/50 |
| answer at loop 16 differs from final | 5/50 | 4/50 |
| final accuracy | 0.380 [0.260, 0.520] | 0.360 [0.240, 0.500] |
| best loop (fp32) | — | loop 16: 0.38; loop 14: 0.36; last: 0.36 |

## huginn_easy (bf16) vs huginn_easy_fp32 (fp32 letters): 50 shared questions
- bf16 steps in the rerun vs the earlier run: max |diff| 0.00e+00 (must be 0)
- fp32 vs bf16 option logits, per-question max |diff|: median 0.062, max 0.062
- final answer (argmax at the last loop) agrees between readouts on 47/50

| quantity | bf16 readout | fp32 readout |
|---|---|---|
| leader changes per question | 5.300 [4.700, 5.900] | 4.980 [4.500, 5.460] |
| transitions with a side margin < one ulp | 173/265 | 139/249 |
| Cui-Ye event rate | 0.680 [0.540, 0.800] | 0.760 [0.640, 0.880] |
| (correct, top distractor) pair ever crosses | 0.980 [0.940, 1.000] | 1.000 [1.000, 1.000] |
| distractor pairs ever cross | 0.987 [0.967, 1.000] | 0.987 [0.967, 1.000] |
| last-change loop, mean (median) | 11.440 [9.880, 13.120] (10) | 11.540 [9.820, 13.380] (10) |
| answer at loop 8 differs from final | 24/50 | 23/50 |
| answer at loop 16 differs from final | 8/50 | 5/50 |
| final accuracy | 0.500 [0.360, 0.640] | 0.540 [0.400, 0.680] |
| best loop (fp32) | — | loop 13: 0.66; loop 14: 0.66; last: 0.54 |
