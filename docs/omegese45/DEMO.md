# OE45 demo: English request to a verified Aienet-0 microjob and back (aien-train #45)

Demo lane of issue #45. One bounded episode, end to end:

```
English request
  -> en2goal (translator lane)              validated Omegese goal G   (or EN_* refusal, nothing dispatched)
  -> hive (this lane) asks the core for A   core: oe45core decode CKPT G|U (pipeline lane)
  -> validator + binding check (msg.om)     invalid A: one guided re-decode (mode G); bind refusal: exit 62
  -> journal A, lower A, aienet0-run/serve  the EXISTING merged Aienet-0 microjob path, loopback
  -> journal R, ask the core for the next A or the final report P
  -> facts.tsv from runtime artifacts only  (interface §5)
  -> rep2en (translator lane)               English response, refused if P contradicts G (G2E_CONTRADICT)
  -> check-demo (eval lane)                 independent check: signatures, recompute, policy, English
```

Status: IMPLEMENTED and EXERCISED (GB10 Spark Grace CPU, cores 10-14, nice 19) for the
reference CONTROL core, all nine cases PASS. The run with a model checkpoint is EXERCISED
only with a pipeline tuning checkpoint, not a published trained checkpoint: 6 of 9 cases
PASS, the 3 failures are model errors caught by the independent check (below). The demo with
the published trained checkpoint is NOT_RUN (no checkpoint has been published by the
pipeline lane).

The only admitted job is `matmul_u32` with m, k, n in 1..256 and a u64 seed (interface §5).
Worker, verifier and driver run on the same Spark over 127.0.0.1, CPU only. No GPU.

## Files

| Path | Role |
|---|---|
| `native/omegese45/hive/hive.om` | the driver (journal, decode, repair, binding, dispatch, reconcile, facts) |
| `tests/omegese45/demo/setup-runtime.sh` | builds the pinned Aienet-0 runtime (open-humanity 53c39bf2, aien-protocols e705fdc9) |
| `tests/omegese45/demo/baseline.sh` | one direct aienet0-run/serve round trip, the runtime baseline |
| `tests/omegese45/demo/run-demo.sh` | the nine cases |
| `evidence/OMEGESE45/demo/` | receipts (`baseline/`, `run-reference/`, `run-model-ts7/`, `core-refuse-prefix/`) |

Build of the driver, with omc at omega-mv bf54b19f (sha256 `62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b`):

```
cat native/omegese45/vocab.om native/omegese45/msg.om native/omegese45/mjr1.om \
    native/omegese45/hive/hive.om | /tmp/om-main/seed/out/omc > hive
```

## Commands

```
# runtime (once): builds aienet0-run, aienet0-serve and the matmul_u32 recompute binary
sh tests/omegese45/demo/setup-runtime.sh /tmp/oe45-demo-rt

# reference CONTROL core: exercises every non-model step
OE45_CORE=reference sh tests/omegese45/demo/run-demo.sh /tmp/oe45-demo-rt /tmp/oe45-demo-ref

# model core: the pipeline oe45core with a checkpoint directory
OE45_CORE=model CKPT=/path/to/ck sh tests/omegese45/demo/run-demo.sh /tmp/oe45-demo-rt /tmp/oe45-demo-model
```

Environment: `MODE=G|U` (first decode, default G), `PORTBASE` (first loopback port, default
14340, never 14323), `OMC` and `OMEGA` (default `/tmp/om-main`). Heavy steps run under
`taskset -c 10-14 nice -n 19 ionice -c3`. OUT must not exist.

Output: `PIN` lines (compiler, commits, runtime binaries), `BIN` lines (sha256 of every built
binary), one `CASE <name> PASS|FAIL <facts>` line per case, then `OE45_DEMO_PASS` (exit 0) or
`OE45_DEMO_FAIL core=… failed: <cases>` (exit 67). Per-case receipts are in
`OUT/cases/<case>/`: `request.en`, `en2goal.*`, `goal.oe45`, `goal.review.en`, `hive.*`,
`episode.oe45`, `dispatch.log`, `facts.tsv`, `rep2en.*`, `response.en`, `check.out`,
`check.rc`, `artifacts/`, `keys/`, `pins/`, `runtime.env`, `worker/`, `logs/`.

## The nine cases (interface §9)

| Case | Request | What it proves | Pass condition |
|---|---|---|---|
| happy-mm | "please multiply a 16 by 32 matrix by a 32 by 8 matrix with seed 7" | full path for one job | hive 0, check FAITHFUL, P `rep ok mm …0x0007` |
| happy-pair | "run two matmuls of size 4 x 4 x 4 with seeds 1 and 2" | two jobs in one episode | hive 0, FAITHFUL, P `rep ok pair …` |
| bound | "run a matmul of size 300 x 8 x 8 with seed 5" | core refusal: nothing dispatched, worker state unchanged | P `rep refused bound end`, header-only facts, sha256 of worker `seen` equal before and after |
| translator | "make me a sandwich" | translator refusal: no goal, nothing dispatched | en2goal 20 EN_UNSUPPORTED, hive not run, FAITHFUL on header-only facts |
| transport | "run a matmul of size 8 x 8 x 8 with seed 11" | worker down once, retry, recovery | facts `1 1 10 fail transport` then `1 2 0 ok`, `intent 1 2 --retry=1` in dispatch.log, FAITHFUL |
| replay | same request as happy-mm, with happy-mm's signed document planted | replay is refused, no second execution | P `rep refused replay end`, worker `executing dag_id=<dag>` count stays 1, one `done <dag>` line in `seen` |
| repair | "run a matmul of size 12 x 6 x 3 with seed 21" | injected invalid A rejected by the validator, guided re-decode, executed | `hive: repair` logged once, hive 0, FAITHFUL |
| restart | "run a matmul of size 32 x 16 x 8 with seed 33" | kill -9 after dispatch, restart from the journal, no duplicate execution | kill rc 0 and driver wait rc 137; journal ends in `act job`; `hive: reconcile 1 1`; worker executions 1; FAITHFUL |
| tampered | copies of happy-mm | negative controls for the independent check | receipt byte 200 flipped: UNFAITHFUL (signature); `seed 7` changed to `seed 8` in response.en: UNFAITHFUL (english.seeds) |

Fault injection is harness glue only (`OUT/work/wrap/`): `serve-down` keeps the worker down
once (flag file), `core-bad` strips the final ` end` of one core answer (flag file),
`run-hang` lets one real aienet0-run finish, records its rc, then sleeps until killed.

### Restart and replay behaviour, as observed

The journal `episode.oe45` is append-only. An A job line is synced before dispatch and its R
line after it. On restart, hive rescans the journal; an A job without its R is reconciled
(`hive: reconcile <job> <attempt>`) by sending the same signed document again with `--retry`.
The worker keeps its state in `WSTATE/seen` and returns the cached receipt for a document it
has already executed, so the restart case shows one `executing` line in the worker logs, one
`done` line in `seen`, and a VERIFIED outcome
([dispatch.log](../../evidence/OMEGESE45/demo/run-reference/cases/restart/dispatch.log),
[worker-state.txt](../../evidence/OMEGESE45/demo/run-reference/cases/restart/worker-state.txt),
[kill.txt](../../evidence/OMEGESE45/demo/run-reference/cases/restart/kill.txt)).
A new episode that submits an already executed document gets `REFUSED WORKER AN0_REPLAY`
from aienet0-run exit 11 (the worker's refusal frame), which hive maps to
`res refused replay end`
([replay facts](../../evidence/OMEGESE45/demo/run-reference/cases/replay/facts.tsv)).

## Driver interface

`hive CONF`, run in the case directory. CONF has 15 lines, one value each: RUN, SERVE, KEYS,
PINS, WKEYS, WPINS, WSTATE, RECOMP, RECSHA, ADDR, CORE, CKPT, MODE (U|G), VERIFIER (hex P-256
compressed key), ACCEPT (seconds). Inputs: `goal.oe45`. Outputs: `episode.oe45`,
`dispatch.log`, `hive.log`, `facts.tsv` (interface §5, v1.2 rules), `disp/`, `out/`, `logs/`.

| Exit | Meaning | Owner |
|---|---|---|
| 0 | episode complete: P journaled, facts.tsv written | hive |
| 61 | usage or config | hive |
| 62 | binding refused before dispatch (msg.om 41/42) | hive |
| 63 | decode failed after one guided repair, or checkpoint refused (core 51/52) | hive |
| 64 | journal, goal or policy guard | hive |
| 65 | driver I/O or spawn failure | hive |
| 66 | unmapped aienet0-run exit, or VERIFIED/MJR1 inconsistent | hive |
| 69 | bundle payload differs from the lowered A | hive |
| 60 | setup or build failed | run-demo.sh |
| 67 | at least one case missed its expectation | run-demo.sh |
| 68 | the eval checker is not runnable | run-demo.sh |

aienet0-run exits map to R exactly as interface §5: 0 `res ok`, 10 `fail transport`,
11 `refused replay|budget|worker` (from the `REFUSED WORKER <tag>` line), 12 `fail receipt`,
13 `fail recomp`, 14 `fail verdict`, 15 `fail io`.

## Results

### Reference CONTROL core: OE45_DEMO_PASS

`OE45_CORE=reference` replaces the model with the eval lane's deterministic reference policy
(`eval/omegese45/oe45lib.py policy_next`). It is a CONTROL: it proves the translator, driver,
journal, binding, Aienet-0 dispatch, restart, replay, rep2en and the independent check, and
says nothing about the model. Run from a fresh clone of `feat/45-hive-demo` at 8a8daa6,
EXERCISED (GB10 Spark Grace CPU, cores 10-14, nice 19): all nine cases PASS, rc 0
([run-demo.txt](../../evidence/OMEGESE45/demo/run-reference/run-demo.txt),
[cases](../../evidence/OMEGESE45/demo/run-reference/cases/)).

### Model core, pipeline tuning checkpoint t-s7: OE45_DEMO_FAIL (6 of 9)

`CKPT=/tmp/op45w/t-s7/ck` is a pipeline tuning checkpoint (step 40000, lr 2^-7, info line in
[run-demo.txt](../../evidence/OMEGESE45/demo/run-model-ts7/run-demo.txt)), not a published
trained checkpoint. Same fresh clone, same host. PASS: happy-mm, bound, translator, replay,
restart, tampered. FAIL: happy-pair, transport, repair. In all three the model emitted
`act refuse bound end` for an in-bound goal. Nothing was dispatched, rep2en refused to render
the report (rc 30 `G2E_CONTRADICT bound`) and check-demo returned UNFAITHFUL
`episode.reference_policy` ([happy-pair check](../../evidence/OMEGESE45/demo/run-model-ts7/cases/happy-pair/check.out)).
The safety layers held; the model is not good enough.

### Core fix found by this demo

The first model run showed that `oe45core` rejected any prefix holding `act refuse` (rc 40),
so the core could never decode the report after a refusal. Fixed on
`fix/45-core-refuse-prefix` (00d66c37, on pipeline 0e908203): a prefix may end in G followed
by `act refuse C end`; a refuse after an R, or any message after the refuse, stays rc 40.
`tests/omegese45/core/run-core.sh` carries the regression and three negatives; the suite
passed with the fix and the old binary returns 40 on the same prefix
([core-refuse-prefix/](../../evidence/OMEGESE45/demo/core-refuse-prefix/)).

## Pins

| Item | Value |
|---|---|
| omc | omega-mv bf54b19f, sha256 `62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b` |
| hive (8a8daa6) | `5afe47201ca88477f65f1745e5a74712be79a1427311421524d5947ce14098a1` |
| en2goal / goal2en / rep2en | `cc043b5b…a7c1` / `7e1d6fd5…6290` / `6d9efa43…e133` ([binaries.sha256](../../evidence/OMEGESE45/demo/run-reference/binaries.sha256)) |
| aienet0-run | `b8685830e175105b254d707d395257cc531532107da1618959f6f309c1e6a08d` (open-humanity 53c39bf2) |
| aienet0-serve | `a56de94baeb56dad79392f5480fb85e9d7bb71bf4b884285c1f24944b458a2de` |
| matmul_u32 recompute | `d2cd7382e0dc64e49498ae3b1ceffc73f1ddcbd3ddcf93e6d607498ccd58ea12` (aien-protocols e705fdc9, SHA unit omega-mv 55845e70) |

## NOT_RUN

- The demo with a published trained checkpoint: no checkpoint has been published by the
  pipeline lane. The t-s7 run above is a tuning checkpoint.
- GPU or Spark-window training and inference (interface §10).
- The transformer model: softmax needs `exp`, `.om` has none (interface §6, §10).
- Hive multi-agent configurations, peer handoff, cancellation, stale dependencies (§10).
- A scientific Turing score, energy measurement, tokenizer comparison (§10; R45 is a fixed
  lexical vocabulary).
- Multi-host dispatch: worker and verifier run on one Spark over 127.0.0.1.
- Jobs other than `matmul_u32` with dims 1..256.
