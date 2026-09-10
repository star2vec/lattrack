# Accuracy by loop at n=200 (peak.py, huginn_easy_fp32_n200)

## all questions: n=200, K=30

| loop | accuracy |
|---|---|
| 4 | 0.295 [0.235, 0.360] |
| 8 | 0.435 [0.365, 0.505] |
| 10 | 0.440 [0.370, 0.510] |
| 12 | 0.500 [0.430, 0.565] |
| 13 | 0.550 [0.480, 0.620] |
| 14 | 0.550 [0.480, 0.615] |
| 15 | 0.540 [0.470, 0.605] |
| 16 | 0.520 [0.450, 0.585] |
| 18 | 0.510 [0.440, 0.580] |
| 20 | 0.485 [0.415, 0.550] |
| 24 | 0.475 [0.405, 0.545] |
| 29 | 0.485 [0.415, 0.555] |

Best loop 13 (0.550); loops 13-14 mean 0.550; last loop 0.485.
Paired: right at loop 13 minus right at last loop = 0.065 [0.005, 0.120]  (gained 23, lost 10 questions)
Paired: right at loop 14 minus right at last loop = 0.065 [0.010, 0.120]  (gained 22, lost 9 questions)
Paired: right at loop 13 minus right at last loop = 0.065 [0.005, 0.120]  (gained 23, lost 10 questions)

## first 50 (the n=50 set): n=50, K=30

| loop | accuracy |
|---|---|
| 4 | 0.300 [0.180, 0.420] |
| 8 | 0.580 [0.440, 0.720] |
| 10 | 0.540 [0.400, 0.680] |
| 12 | 0.620 [0.480, 0.740] |
| 13 | 0.660 [0.520, 0.780] |
| 14 | 0.660 [0.520, 0.780] |
| 15 | 0.640 [0.500, 0.760] |
| 16 | 0.600 [0.460, 0.720] |
| 18 | 0.600 [0.460, 0.740] |
| 20 | 0.560 [0.420, 0.700] |
| 24 | 0.500 [0.360, 0.640] |
| 29 | 0.540 [0.400, 0.680] |

Best loop 13 (0.660); loops 13-14 mean 0.660; last loop 0.540.
Paired: right at loop 13 minus right at last loop = 0.120 [0.000, 0.240]  (gained 8, lost 2 questions)
Paired: right at loop 14 minus right at last loop = 0.120 [0.000, 0.240]  (gained 8, lost 2 questions)
Paired: right at loop 13 minus right at last loop = 0.120 [0.000, 0.240]  (gained 8, lost 2 questions)

## questions 51-200 (new): n=150, K=30

| loop | accuracy |
|---|---|
| 4 | 0.293 [0.220, 0.367] |
| 8 | 0.387 [0.313, 0.460] |
| 10 | 0.407 [0.333, 0.487] |
| 12 | 0.460 [0.380, 0.540] |
| 13 | 0.513 [0.433, 0.587] |
| 14 | 0.513 [0.433, 0.593] |
| 15 | 0.507 [0.427, 0.587] |
| 16 | 0.493 [0.413, 0.573] |
| 18 | 0.480 [0.400, 0.560] |
| 20 | 0.460 [0.380, 0.540] |
| 24 | 0.467 [0.387, 0.547] |
| 29 | 0.467 [0.393, 0.547] |

Best loop 13 (0.513); loops 13-14 mean 0.513; last loop 0.467.
Paired: right at loop 13 minus right at last loop = 0.047 [-0.013, 0.107]  (gained 15, lost 8 questions)
Paired: right at loop 14 minus right at last loop = 0.047 [-0.013, 0.100]  (gained 14, lost 7 questions)
Paired: right at loop 13 minus right at last loop = 0.047 [-0.013, 0.107]  (gained 15, lost 8 questions)

## RESULT 2 / 10 / 11 quantities at n=200 (all)

- Cui-Ye event rate 0.635 [0.565, 0.700]; (correct, top distractor) ever crosses 0.965 [0.935, 0.990] vs distractor pairs 0.992 [0.983, 0.998]
- accuracy on questions with an event 0.488 [0.402, 0.575] vs without 0.479 [0.370, 0.589]
- last-change loop mean 11.945 [11.065, 12.890], median 10; answer at loop 16 differs from final 34/200
- final accuracy 0.485 [0.415, 0.555]
