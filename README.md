# lattrack

Working name. When a latent reasoning model's decoded leaning changes mid-computation,
is it revisiting an earlier latent state, overriding it, or is the change only in the
decoder? `BRIEF.md` has the question, `WORKSTYLE.md` how we work, `lit-sweep.md` the
literature, `LOG.md` the running record.

The pilot runs on the two-layer COCONUT-style model from RRR/phoenix (graph
reachability, trained from scratch). Infrastructure files under `src/lattrack/` are
copies from the phoenix repo; each carries its origin in a header line.

## Setup (Mac, CPU)

```bash
git clone --depth 1 https://github.com/Ber666/reasoning-by-superposition.git vendor/reasoning-by-superposition
git -C vendor/reasoning-by-superposition fetch --depth 1 origin 72d22af7ec1533aef6bd208e3f08a819d3e1b47f
git -C vendor/reasoning-by-superposition checkout -q FETCH_HEAD
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.5.1 numpy==2.1.3 transformers==4.46.2 datasets==3.1.0 tqdm==4.67.0 pyyaml
```

Checkpoints go in `ckpts/<run>/best.pt` (gitignored). `seed0` and `seed1` are the
paper's accepted `s1_seed0_win` and `s1_seed1_win`; their sha256 prefixes are in
`LOG.md`.

## Tests (run before anything else, after any change to `fast_coconut.py`)

```bash
.venv/bin/python tests/test_fast_equivalence.py   # EQUIVALENCE: PASS
.venv/bin/python tests/test_prompts.py            # PROMPTS: PASS
.venv/bin/python tests/test_attn_hooks.py         # HOOKS: PASS
```

## Runs

```bash
.venv/bin/python src/lattrack/lens.py --run-name seed1 --graphs pilot   # smoke, 10 graphs
.venv/bin/python src/lattrack/lens.py --run-name seed0
.venv/bin/python src/lattrack/lens.py --run-name seed1
.venv/bin/python src/lattrack/lens.py --run-name random
.venv/bin/python src/lattrack/lens.py --summarize seed0 seed1 random
```

Results land in `results/<run>/`. Every number in `LOG.md` points at a file there.
