# How we work on this

Lessons from previous projects. The failure mode we're avoiding is a protocol so strict it became an obstacle: fixed in advance, checked against constantly, hard to change once understanding improved, and eventually a record nobody wanted to look at. The opposite failure is no baselines at all. We want the middle.

## Rules that stay

- Every experiment has a null or baseline. What would this look like if the interesting story were false? Run that.
- Every gate or threshold comes with one sentence saying what it means and how it was chosen. If we can't write that sentence, the number isn't a gate yet, it's a guess, and we say so.
- Noise gets measured, not assumed. Thresholds that depend on noise are set after seeing it.
- Positive controls are good: a case where the effect must appear, so we know the tooling can see it.

## Rules that flex

- Decisions can change when understanding changes. A change gets a one-line note (what, why), not a re-litigation.
- Prior work (RRR, the lit sweep, tarcle) is guidance, not a constraint. If it points somewhere, follow it; if it's wrong for this project, say so and move on.
- Don't pre-commit to methods before the pilot has shown which ones work here.

## Record keeping

- One short running log, chronological, a few lines per session: what was tried, what came out, what changed. That's the whole record. No separate protocol document to check against.
- Numbers in the log carry their conditions (model, task, n, seed) so they can be trusted later.
- If something embarrassing happened, log it in one line and continue. The log is for future us, not for a reviewer.

## When unsure

Ask, propose options with trade-offs, don't decide silently. Especially on anything that would lock a task, model or definition.
