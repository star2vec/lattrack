# Distribution behind RESULT 10 (commitment_dist.py, no new runs)

## graph seed0: n=419

K=3, positions ['root', 'l0', 'l1', 'A']. Where the LAST leader change falls (0 = never changed):

| never | root->l0 | l0->l1 | l1->A |
|---|---|---|---|
| 85 (0.421) | 39 (0.193) | 73 (0.361) | 5 (0.025) |

Graphs committed before A (n=197): mean |gap| 2.53 at commitment, 8.76 at A; grew in 0.96.

K=4, positions ['root', 'l0', 'l1', 'l2', 'A']. Where the LAST leader change falls (0 = never changed):

| never | root->l0 | l0->l1 | l1->l2 | l2->A |
|---|---|---|---|---|
| 57 (0.263) | 30 (0.138) | 65 (0.300) | 56 (0.258) | 9 (0.041) |

Graphs committed before A (n=208): mean |gap| 2.31 at commitment, 8.93 at A; grew in 0.95.

## graph seed1: n=419

K=3, positions ['root', 'l0', 'l1', 'A']. Where the LAST leader change falls (0 = never changed):

| never | root->l0 | l0->l1 | l1->A |
|---|---|---|---|
| 63 (0.312) | 53 (0.262) | 83 (0.411) | 3 (0.015) |

Graphs committed before A (n=199): mean |gap| 3.41 at commitment, 8.26 at A; grew in 0.92.

K=4, positions ['root', 'l0', 'l1', 'l2', 'A']. Where the LAST leader change falls (0 = never changed):

| never | root->l0 | l0->l1 | l1->l2 | l2->A |
|---|---|---|---|---|
| 39 (0.180) | 40 (0.184) | 67 (0.309) | 69 (0.318) | 2 (0.009) |

Graphs committed before A (n=215): mean |gap| 3.07 at commitment, 8.70 at A; grew in 0.95.

## Huginn ARC-Challenge: n=50, K=32

Last-change loop: mean 12.1 (frac 0.392), median 10; histogram 1:1 3:1 5:1 6:1 7:9 8:7 9:5 10:3 11:2 12:4 13:1 14:1 15:4 16:1 17:1 19:1 21:1 26:2 27:1 28:1 30:1 31:1
Letter first becomes the argmax at loop 3.9 mean, 4 median.
Leader-change transitions 277; with one side's margin below one bf16 ulp (0.13): 195 (0.70).
Last loop at which a DIFFERENT letter led by more than tol: tol 0.0: loop 11.8 (frac 0.381); tol 0.13: loop 8.5 (frac 0.275); tol 0.3: loop 6.6 (frac 0.212); tol 0.5: loop 4.8 (frac 0.154)
Some other letter ever led by > tol (event presence): tol 0.0: 1.00, tol 0.13: 1.00, tol 0.3: 1.00, tol 0.5: 0.98; Cui-Ye event rate as logged 0.66.
Answer at loop 8 differs from the final answer: 25/50; margins at loop 8: 0.00, 0.00, 0.00, 0.00, 0.12, 0.12, 0.12, 0.12, 0.13, 0.13, 0.13, 0.13, 0.13, 0.25, 0.25, 0.25, 0.25, 0.25, 0.37, 0.37, 0.37, 0.38, 0.62, 0.62, 0.75
Answer at loop 16 differs from the final answer: 5/50; margins at loop 16: 0.00, 0.00, 0.12, 0.25, 0.50
Margin at the last change 0.15 -> at the last loop 0.68; grew in 0.86.

Accuracy by loop (argmax at loop k == correct letter), bootstrap CI:

| loop | accuracy |
|---|---|
| 0 | 0.22 [0.10, 0.34] |
| 4 | 0.22 [0.12, 0.34] |
| 8 | 0.32 [0.20, 0.46] |
| 12 | 0.32 [0.20, 0.44] |
| 14 | 0.36 [0.24, 0.50] |
| 16 | 0.38 [0.26, 0.52] |
| 20 | 0.34 [0.22, 0.48] |
| 24 | 0.32 [0.18, 0.46] |
| 31 | 0.38 [0.24, 0.52] |

## Huginn ARC-Easy: n=50, K=30

Last-change loop: mean 11.4 (frac 0.394), median 10; histogram 3:1 4:2 5:3 6:4 7:3 8:8 9:3 10:4 11:3 12:3 13:1 14:1 15:4 17:2 19:1 20:2 22:1 24:3 27:1
Letter first becomes the argmax at loop 3.9 mean, 4 median.
Leader-change transitions 265; with one side's margin below one bf16 ulp (0.13): 173 (0.65).
Last loop at which a DIFFERENT letter led by more than tol: tol 0.0: loop 11.4 (frac 0.394); tol 0.13: loop 8.8 (frac 0.303); tol 0.3: loop 7.3 (frac 0.252); tol 0.5: loop 4.2 (frac 0.146)
Some other letter ever led by > tol (event presence): tol 0.0: 1.00, tol 0.13: 1.00, tol 0.3: 0.98, tol 0.5: 0.96; Cui-Ye event rate as logged 0.68.
Answer at loop 8 differs from the final answer: 24/50; margins at loop 8: 0.00, 0.12, 0.12, 0.12, 0.12, 0.13, 0.13, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.37, 0.37, 0.50, 0.50, 0.62, 0.75, 0.75, 0.87, 1.00
Answer at loop 16 differs from the final answer: 8/50; margins at loop 16: 0.00, 0.00, 0.12, 0.12, 0.12, 0.37, 0.50, 0.50
Margin at the last change 0.18 -> at the last loop 0.64; grew in 0.80.

Accuracy by loop (argmax at loop k == correct letter), bootstrap CI:

| loop | accuracy |
|---|---|
| 0 | 0.18 [0.08, 0.28] |
| 4 | 0.34 [0.20, 0.48] |
| 8 | 0.58 [0.44, 0.72] |
| 12 | 0.62 [0.48, 0.74] |
| 14 | 0.64 [0.52, 0.76] |
| 16 | 0.60 [0.46, 0.72] |
| 20 | 0.54 [0.40, 0.68] |
| 24 | 0.48 [0.34, 0.62] |
| 29 | 0.50 [0.36, 0.64] |

## RESULT 9, alpha=0: are the crossings bf16 ties?
null pair: crossed 11/29; end gap within one ulp 0; opposite leader ahead by > 0.13 at some loop 11.
real pair: crossed 10/29; end gap within one ulp 1; opposite leader ahead by > 0.13 at some loop 10.
