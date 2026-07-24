# Honest evaluation rules

These rules are baked into the envs where possible and expected of every
result you report. They exist because we have watched policies game every
loophole we left open.

## 1. Post-reset blindness

Never sense or score the first ~2 steps after any WORLD-MODEL reset. World
models settle after a reset; early frames carry artifacts. `LocalDreamEnv`
enforces this (`sensor_valid=0`, zeroed proximity, suppressed collision) —
your training loop must respect the flag. The physics exam is different by
design: exact physics has no settle window, so the exam senses from tick 0
and `sensor_valid` is always 1 there — never make a policy DEPEND on
post-reset blindness.

## 2. Dream scores are not capability claims

A policy that scores well inside the dream has learned the dream — including,
possibly, its exploits (wall-grinding, floor-staring, pivot-in-place). The
**frozen Webots oracle** is the sole scoreboard: fixed worlds, fixed
contract, anti-exploit control arms (a cruiser policy that MUST fail — if it
doesn't, the exam run is invalid, not a pass).

Report gym/dream numbers as training telemetry. Report capability only as an
exam verdict, and quote the verdict honestly — including which bars failed.

## 3. Determinism is platform-scoped

Training inside `LocalDreamEnv` with a fixed seed is bit-deterministic on a
fixed platform, but champions differ across platforms (arm64 vs amd64 float
paths). Reproducibility claims must name the platform. Exam results are
deterministic to aggregates (third-decimal physics drift across launches).

## 4. Sessions cost real money (RemoteSimEnv)

One session = one live world-model GPU stream, billed by wall-time. `close()`
every session; an orphaned session keeps billing. Coordinate parallel-session
counts with your key's budget.
