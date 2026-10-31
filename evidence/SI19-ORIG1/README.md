# SI19-ORIG1: frozen selector task and dataset admission (aien-train#24)

Host run of `sh tests/si19/run-omexp.sh` with `RECORDS` pointing at the local owner
dataset. Source commit `09e5228b48405bc10dc5bd067a9600c332742790` (branch
`feat/24-orig1-selector-contract`, on top of PR #22 `fa0ca1c3`). Host
`Linux 7.0.0-1019-nvidia aarch64` (DGX Spark, CPU only). Compiler
`191b776e...98e9` (pinned); the R6 compiler `734f0a39...6b61` compiled the same test unit
and passed it too (`run/omexp-test-r6.*`). Raw outputs: `run/`, hashed in `run/raw.sha256`;
build artifacts hashed in `run/build.sha256` (not committed).

## Status

| item | status |
|---|---|
| contract `docs/si19/ORIG1-CONTRACT.md`, manifest `docs/si19/orig1-manifest.json` | FROZEN |
| admission and refusal tests (`tests/si19/omexp-test.om`) | PASS, 1497 checks, 6 ms; 6 fault-injected units compiled and failed the unchanged test |
| process-path refusals (zero file, trailing damaged record, forged mismatch word, unlisted hash) | PASS: exit 66/66/66/67, status -2/-2/-7/-13 |
| SYNTHETIC fixture `sxgen 7 780` | admitted as SYNTHETIC (allow status 2), fnv `626048380840256264` |
| owner dataset (738 records, local) | admitted as AUTHORIZED (allow status 1), fnv `16622604375676129865`, sha256 `403156f5...8d37` |
| sealed final holdout | BLOCKED_OPERATOR on this machine; every TEST number is development-only. ORIG-3 seals by a fresh tap after the candidate hash is pinned |
| GB10 / hardware | NOT_RUN |

## Owner dataset: collection receipt

Collected 2026-10-25T03:47:37Z to 03:48:39Z (62 s wall) by the consumer's own harness
`tests/runtime/rx_empirical_optimizer.c` with the default-off tap, built from the tree of
omega-mv `1ec81372` (`make AIENOS_LOCK_REPO=... build/rx_empirical_optimizer_test`, binary
sha256 `d26aec02...81ee`), `OMEGA_EMPIRICAL_RECORDS=<local path> EMPIRICAL_ALLOW_NO_WIN=1`.
The harness reported `checked 20222 runs against the reference: 0 mismatches`,
`checks 44 failures 0`, and its own verdict `learned selection wins` (the C cost model's
result, recorded for reference; it is not a teacher and nothing of it enters the dataset
beyond the measured tables). Families: 308 TRAIN (154 per core class: 77 shapes x 2
passes), 70 DEV, 120 H1, 120 H2, 120 H3. Eligible mask 31 (5 arms; `planted_wrong`
refused by the harness). The file is 0600 under `~/.local/state/aien-train/orig1/` and is
not published; it contains no personal data.

## Deterministic baselines on the owner dataset (`run/inspect-owner.stdout`)

Unit: sum of ps per call over the partition's records (contract s.7, unweighted). Oracle =
cheapest admissible arm per record (analysis bound). Best static by this metric is `quad4`
on both core classes (the harness's calls-weighted total names `reference` on A725; both
are correct for their metric).

| partition | core | n | oracle | fixed rule | reference | best static | retrieval | rule top-1 | retrieval top-1 |
|---|---|---|---|---|---|---|---|---|---|
| TRAIN | X925 | 154 | 598629162 | 1.748 | 1.678 | 1.0038 | 1.0000 | 5 | 153 |
| TRAIN | A725 | 154 | 1333277451 | 1.813 | 1.060 | 1.0124 | 1.0001 | 0 | 153 |
| DEV | X925 | 35 | 114072181 | 1.787 | 1.717 | 1.0000 | 1.0000 | 0 | 35 |
| DEV | A725 | 35 | 258287978 | 1.826 | 1.066 | 1.0093 | 1.0210 | 0 | 27 |
| TEST (H1+H2+H3) | X925 | 180 | 1962560697 | 1.559 | 1.515 | 1.0048 | 1.0001 | 0 | 176 |
| TEST (H1+H2+H3) | A725 | 180 | 3233131109 | 1.447 | 1.092 | 1.0137 | 1.0041 | 0 | 144 |

Ratios are cost over oracle. Reading: the incumbent fixed rule is 45 to 83 % above the
oracle; the cheap heuristics (best static, retrieval) are within 0.0 to 2.1 %. Any learned
scorer therefore has to be judged against best static and retrieval, not against the fixed
rule; ORIG-3's decision rule gates on that. This is a finding, not a failure.

## Fixture (`run/inspect-fixture.stdout`)

The SYNTHETIC fixture's planted rule is deliberately core-and-N dependent; its baselines
are in the raw output and are reported separately from the owner numbers above.

## Not covered here

Training, the scorer, checkpoints and the learned baselines (ORIG-2); the independent
evaluation (ORIG-3); the default-off shadow consumer (ORIG-4).
