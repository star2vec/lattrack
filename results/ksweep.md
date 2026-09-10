# K sweep, Huginn ARC-Easy, 50 shared questions (ksweep.py)

## Are shorter runs prefixes of the longer one?

- K=16 vs first 16 loops of K=30: max |logit diff| 0.00e+00
- K=30 vs first 30 loops of K=64: max |logit diff| 0.00e+00
- K=16 vs first 16 loops of K=64: max |logit diff| 0.00e+00

## Last change: loop count vs fraction

| K | last-change loop mean | median | as fraction of K-1 | last change > loop 16 | final answer differs from K=64's |
|---|---|---|---|---|---|
| 16 | 9.4 [8.5, 10.3] | 9 | 0.624 | 0.00 | 0.14 |
| 30 | 11.4 [9.9, 13.1] | 10 | 0.394 | 0.20 | 0.06 |
| 64 | 14.9 [11.1, 19.1] | 10 | 0.236 | 0.22 | 0.00 |

Last-change loop histogram at K=64: 3:1 4:2 5:3 6:4 7:3 8:8 9:3 10:4 11:3 12:3 13:1 14:1 15:3 17:2 20:2 24:1 38:2 42:1 60:1 62:2

Last loop at which a DIFFERENT letter than the final one led by more than tol, K=64: tol 0.0: 14.8; tol 0.13: 9.0; tol 0.3: 7.1; tol 0.5: 4.3

## Answers still moving late (K=64 trajectory)

- answer at loop 8 differs from the answer at loop 63: 23/50
- answer at loop 16 differs from the answer at loop 63: 6/50
- answer at loop 24 differs from the answer at loop 63: 2/50
- answer at loop 32 differs from the answer at loop 63: 3/50
- answer at loop 48 differs from the answer at loop 63: 2/50

## Accuracy by loop (argmax at loop k == correct letter), K=64 trajectory

| loop | accuracy |
|---|---|
| 4 | 0.34 [0.22, 0.48] |
| 8 | 0.58 [0.44, 0.70] |
| 12 | 0.62 [0.48, 0.76] |
| 14 | 0.64 [0.50, 0.78] |
| 16 | 0.60 [0.46, 0.74] |
| 20 | 0.54 [0.40, 0.68] |
| 24 | 0.48 [0.34, 0.62] |
| 29 | 0.50 [0.36, 0.64] |
| 32 | 0.54 [0.40, 0.68] |
| 40 | 0.50 [0.36, 0.64] |
| 48 | 0.52 [0.38, 0.66] |
| 56 | 0.52 [0.38, 0.66] |
| 63 | 0.52 [0.38, 0.66] |

Best single loop: 13 (accuracy 0.66).

Mean top-1 minus top-2 margin by loop: 4: 0.27, 8: 0.35, 16: 0.60, 24: 0.61, 32: 0.64, 48: 0.63, 63: 0.62

Fraction of questions whose leader changes at transition k->k+1, binned: loops 0-8: 0.500, loops 8-16: 0.113, loops 16-24: 0.043, loops 24-32: 0.013, loops 32-48: 0.029, loops 48-63: 0.013
