# adtrain — native autodiff training example

Status: IMPLEMENTED and EXERCISED on the host (aarch64 Linux, DGX Spark) for
gates 1–4 and the host arm of gate 5, plus the exact-rational gradient oracle
(Wolfram). The GB10 arm of gate 5 is NOT_RUN. Nothing here is QUALIFIED, MERGED
or DEPLOYED. Tracks aien-train #6 (trainer, optimizer, checkpoint) on top of
omega-mv #664 (native tape).

## What it is

`native/adtrain/mlp.om` is handwritten seeded `.om` with native `f32` words. It
trains a relu MLP by full-batch gradient descent. Gradients come from the
native reverse-mode tape `native/autodiff664/tape.om` (omega-mv); this
repository contains no hand-derived derivative. Build order: omega-mv
`native/ecosystem663/metadata.om`, omega-mv `native/autodiff664/tape.om`,
`native/adtrain/mlp.om`, then the caller (`tests/adtrain/main.om`). Pins:
`dependencies.json`.

- Data and model: `X[8,4]`, `T[8,2]`, `W1[4,8]`, `b1[1,8]`, `W2[8,2]`, `b2[1,2]`.
- Graph (tape nodes 0–16, checked at every build):
  `X W1 b1 W2 b2 T MM1 BC1 ADD1 RELU MM2 BC2 ADD2 SUB MUL SUM1 SUM0`, that is
  `H = relu(X·W1 + b1)`, `Y = H·W2 + b2`, `loss = Σ (Y − T)²`.
- Initialisation: one xorshift64 stream (seed 17; mode Z uses 29) gives
  `k = (next mod 17) − 8`. `X` and `T` are `k/8`, then `W1` and `W2` are `k/16`.
  Biases are 0. Every value is a dyadic rational, exact in binary32.
- Training: `lr = 1/64` (computed as `div 1.0f32 64.0f32`), 200 steps,
  `p = p − round(lr·g)`. The loss and all 58 parameters must stay finite
  (`affinite` bit test, ecosystem6), else exit 86.

## Gates and how each is proven

`OMC=<pinned omc> OMEGA=<omega-mv checkout> OUT=<scratch> sh tests/adtrain/run-adtrain.sh`
ends `ADTRAIN_SUITE_PASS`. The suite materialises the dependencies from their
pinned Git objects and checks every SHA-256 first.

| gate | mode | check |
|---|---|---|
| 1 forward + loss | T | 201 `L <step> <loss bits>` rows from the tape forward |
| 2 gradients vs independent reference | G | tape gradient against a central difference per parameter, ε = 1/64 (rule below); ≥ 50 checked, none failing |
| 2 (second reference) | D + host arm | C M21 tape on the same inputs, value-exact |
| 3 updates reduce loss | T | `loss[200] < loss[0]` and `loss[200] < loss[100]`, in the program (exit 88) and again on the printed bits by the runner |
| 4 checkpoint/resume | S, R | train to 120, save, fresh process loads and continues to 200; `L 120..200` and all `P` rows `cmp`-identical to the uninterrupted T run |
| 5 CPU/GPU agreement | D | fixture replayed by omega-mv `tests/test_omega_ad_fixture.c`: host arm (M20 CPU table) PASS; GB10 arm `tests/run_ad_chip.sh` NOT_RUN |

Central-difference rule (si19 `qfdone`, extended for small gradients):
`d = (f(p+ε) − f(p−ε))·32`. The check passes iff `|g| ≥ 1/64` and
`16·|d − g| ≤ |g|`, or `|g| < 1/64` and `|d − g| ≤ 1/64`. A parameter is
skipped when either perturbation changes the relu pattern. Observed: 58
checked, 0 skipped, 0 failed.

Declared tolerances for gate 5: PASS is 0 ULP on all 75 values (Y, loss,
dW1, db1, dW2, db2). PASS_TOL allows at most 2 ULP. Anything else is FAIL.
Observed on the host: `exact=75/75 max_ulp=0 verdict=PASS`. The
`--mutant-sign` control (expected dW1 negated) FAILs as required.

## Checkpoint `adtrain.ck`

The image is 536 bytes of u64 little-endian words:

| word | content |
|---|---|
| 0 | magic: the bytes of the text `ADCKPT01` |
| 1 | version 1 |
| 2 | step |
| 3 | 58 |
| 4 | lr bits |
| 5 | loss bits |
| 6 | data hash (FNV-1a 64 over the X and T bit patterns) |
| 7 | 0 |
| 8–65 | the 58 parameters (f32 bits, zero-extended) |
| 66 | FNV-1a 64 over bytes 0..527 |

Saving uses `fopen` (create/truncate, 0600), a full `write`, `fsync` and
`fclose`. The load (mode R) refuses, leaving every parameter unchanged, in
these cases:

- exit 65: size ≠ 536, checksum, magic, version, count, step > 200, lr,
  non-zero padding, or a non-finite parameter.
- exit 66: the image was trained on other data (data hash ≠ current X, T).

There is no `rename` host word, so atomic publication is not claimed; the
checksum rejects a torn image. The suite exercises each refusal path:

| image | exit |
|---|---|
| seed-29 image | 66 |
| truncated to 535 bytes | 65 |
| byte 96 flipped (parameter mantissa, checksum only) | 65 |
| byte 100 flipped (zero-extension padding) | 65 |

## Exit codes

| code | meaning |
|---|---|
| 1 | gradient check failed |
| 65 | corrupt checkpoint |
| 66 | data binding mismatch |
| 74 | file I/O |
| 86 | non-finite loss or parameter |
| 87 | tape refusal or unexpected node id |
| 88 | loss not reduced |
| 90 | bad mode byte |

## Compiled mutants (`tests/adtrain/mutate.om`)

| mode | fault | observed |
|---|---|---|
| 0 | update direction reversed | T exits 86 |
| 1 | data-hash check removed | the seed-29 image then loads (exit 0) |
| 2 | checksum check removed | the byte-96 image then loads (exit 0) |

Each mutant compiles with rc 0. Modes 1 and 2 show that the two checks carry
the refusals; without them the load is wrongly accepted.

## Exact-rational gradient oracle (gate 2, third reference)

`tests/adtrain/gradient.wl` regenerates the seed-17 data with the same
xorshift64, binds to the `K <i> <k>` rows of mode D, fixes the relu pattern
(`x > 0`) from the exact forward, differentiates the exact loss symbolically
and compares every `G` row against `|g − g_exact| ≤ 2^-16·max(1, |g_exact|)`.
It ran on the hosted Wolfram service (`WolframLanguageEvaluator`). Observed:
the K rows match, 58/58 gradients equal the exact values, and the step-0 loss
equals `4155649/524288` exactly. A copy with one gradient's sign bit flipped
reports `allWithin False`. The exactness comes from the small dyadic data:
every binary32 operation of this step is exact. No pre-activation is exactly 0,
so this oracle does not exercise the relu rule at 0; `tests/autodiff664`
does. Evidence: `evidence/ADTRAIN/oracle.json`, `evidence/ADTRAIN/oracle/`.

## Not run, and why

- **GB10 arm: NOT_RUN.** It needs an explicit GPU window (live 35B on 18091;
  `GB10_CHIP_RUN=1`, `/tmp/aien-gb10.lock`). Blocker record:
  `evidence/ADTRAIN/gb10/NOT_RUN.md`.

## Not claimed

General training, GPU launch from `.om` (omega-mv #638 / CUDApp #35),
performance, atomic checkpoint publication, or hardware qualification.
