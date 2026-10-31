# OMEGESE45 pipeline receipts (aien-train #45)

All runs were on the **GB10 Spark Grace CPU, cores 10-14 (Cortex-A725), nice 19, ionice idle**, with
`taskset -c 10-14 nice -n 19 ionice -c3`. No GPU was used. Raw command outputs were copied from
`/tmp/op45w/`; Markdown files describe them. Compiler: omc at omega-mv `bf54b19f`, sha256
`62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b`.

The suite was rerun from a fresh clone of clean branch `feat/45-pipeline` at `158e712`, and again
after the generator comment correction at `0a876fe`. `fresh-suite.log` and `fresh-final-suite.log`
both end in `OE45_CORE_PASS` (rc 0), with both mutants KILLED; their `.txt` companions record the
commands, fixture decode-U rc 0 and binary hashes. `clean-branch.txt` proves initial source/test
equivalence to the old pipeline plus contract `80cc357`, with PR #42 absent. The only final source
delta is generator comments (`clean-branch-final.diff`). Regenerated train/dev files are still
byte-identical (`data/final-regen.txt`). No old branch was rewritten.

| Directory | What | Command |
|---|---|---|
| `suite/` | Core suite after merging `fix/45-core-refuse-prefix` (00d66c3). `run-core.log` ends in `OE45_CORE_PASS`, 14.89 s wall. `*.rc`, `*.stdout` and `*.stderr` are the per-step outputs; files over 64 KiB are recorded only by sha256 in `large-outputs.sha256`. | `OMC=/tmp/om-main/seed/out/omc OMEGA=/tmp/om-main OUT=/tmp/op45w/suite sh tests/omegese45/core/run-core.sh` |
| `data/` | Generator stdout counters and the sha256 and sizes of the train and dev files. The files were regenerated with the head binary and compared with `cmp`: identical. | `oe45gen 4501 20000 train.oe45 train.ids`; `oe45gen 4502 1000 dev.oe45 dev.ids` |
| `train/r1`, `train/r2` | The full 400,000-step training run, done twice. Each has `log` (rc per step), `init.out`, `info0.out`/`info.out`, `dev0.out`/`dev.out`, `train.out` (T lines), `time.txt` (`/usr/bin/time -v`) and `sha.txt`. | `train/final.sh` |
| `train/repro-cmp.txt` | `cmp` of r1 against r2 for `step0.ck`, `ck.a`, `ck.b` and `train.out`: identical. | |
| `train/tune-*` | 40,000-step tuning runs: s10, s8, s7 and s6 are learning rates 2⁻¹⁰, 2⁻⁸, 2⁻⁷ and 2⁻⁶, and s8p5 is 2⁻⁸ with P1 5000. | `train/tune.sh` |
| `batch/` | Full-dev batch decode (2410 oracle prefixes), U and G, on the step-0 and trained checkpoints. It holds the `vok` counts, the exact-match counts, errors by message type, `evaldev` rerun with the head binary, binary sha256 and checkpoint sha256. | see `docs/omegese45/PIPELINE.md` §4 |

## Binaries

The training run used `/tmp/op45w/b2/oe45core` (sha256 `c7bb4233…482d`). It was built from
`0e908203` before the refuse-prefix merge, which changed only `opprefix` in `cli.om`:
`git diff --stat 0e90820 HEAD` over model.om, ckpt.om, run.om and gen.om is empty.

The head binary `/tmp/op45w/suite/bin/oe45core` (sha256 `7dcedc8c…6c0f`) reproduces both `evaldev`
lines exactly (`batch/evaldev-*-head-binary.out`).

## Checkpoints

The checkpoints are not committed (3,989,908 B per slot).

| Checkpoint | Path | sha256 |
|---|---|---|
| step 0 | `/tmp/op45w/final/ck0/ck.a` | `02d85006cd96c9c450c66147538a286b07954964fa14552fb40501fcf77225f2` |
| trained `ck.a` (newest, gen 21) | `/tmp/op45w/final/ck/ck.a` | `2ebb99de9a0aa04dfba95e0f008e392b99c3f4c4c3836086e24cf7b3a8204058` |
| trained `ck.b` | `/tmp/op45w/final/ck/ck.b` | `71118b656eb3aa8ac5a5d75e5d5a35710d49a44e78bfcbba88cfcec81a25892e` |

## Not run

- GPU training or inference: NOT_RUN.
- Transformer: NOT_RUN (no `exp` intrinsic).
- Native tape: NOT_RUN, because it does not fit; see PIPELINE.md §5.
