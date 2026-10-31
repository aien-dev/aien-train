# SI19-ORIG2: original candidate scorer trained from random initialization (aien-train#25)

Two host runs of `sh tests/si19/run-select.sh` at source commit
`736530dd70d7a380c0d65bf5d7824f08f2ac0eea` (branch `feat/25-orig2-native-scorer`, on
ORIG-1 `3e30ee41` and PR #22 `fa0ca1c3`). Host `Linux 7.0.0-1019-nvidia aarch64`
(DGX Spark, CPU only). Compiler `191b776e...98e9` (pinned); the R6 compiler
`734f0a39...6b61` compiled the same units and passed the same test.

- `fixture-run/`: SYNTHETIC fixture (`sxgen 4242 400`, fnv `13341539012208942374`), the
  full script including 6 fault-injected units. Fixture evidence only.
- `owner-run/`: the local owner dataset (738 records, fnv `16622604375676129865`, sha256
  `403156f5...8d37`, admitted AUTHORIZED). Real-record evidence, development-only TEST.
- `candidates/`: the immutable candidate images for ORIG-3 (11560 and 1816 bytes):
  `mlp-scratch-seed19024.img` sha256 `8dbd9b04a446b02d8e7bf2f1e2aa808ca439e813abfcb52fb34e57d40538acb5`,
  `affine-seed19024.img` sha256 `f04bc778b9347daa10a175a6d2a300a3cc513f6efc53e5412361ba1e9804121b`.
  A second independent run of the script reproduced both images byte for byte.

## Status

| item | status |
|---|---|
| native parameter updates from declared random init; predictions change (`tchange`) | PASS |
| independent numerical checks: exact integer golden oracle for two MLP updates (scale 2^24, no rounding), central finite differences for both kinds, ReLU convention at 0 | PASS, 50333 checks, 269 ms, max RSS 320 KiB |
| fault injection: 6 altered units (gradient sign, ReLU gate, update sign, dataset-binding predicate, cursor persistence, shuffle permutation) | 6/6 compiled and failed the unchanged test |
| restart: partial at epochs 1 and 3, resumed in a new process | image, report and per-record lines byte-identical to the uninterrupted run (fnv `9232652764948363499`) |
| controls | shuffled targets, no-update (26180 updates, params bit-identical to init), untrained |
| refusals through the process path | damaged image 65 (-5), zero-padded short image 65, different-image compare 1, bad config 68, REVOKED 67 (-12), unlisted 67 (-13) |
| provenance (`tprovenance`) | every SS word and every initial parameter recomputed from seed 19024 and the LCG; lineage = seed + init fnv + dataset fnv |
| lr choice on DEV only | affine 2^-8 (DEV loss sum 7.26 vs 7.52 / 7.50), MLP 2^-8 (2.41 vs 2.53 / 8.60); TEST never read before the images were final |
| candidate | inactive; immutable; awaits ORIG-3 (sealed records) and ORIG-4 (default-off shadow) |
| GB10 / deployment | NOT_RUN / not part of this leaf |

## Identities (owner run)

| field | MLP (kind 2) | affine (kind 1) |
|---|---|---|
| parameters | 705 (20 x 32 + 32 + 32 + 1) | 96 (8 x 12) |
| seed, init | 19024; LCG 1664525/1013904223; weights uniform in {-16..15}/128, biases 0 | same |
| init params fnv | 11780572735689526495 | 1536629692949879461 |
| normalization (TRAIN only) | mu16 = 3, sh = 0 | same |
| lr | 2^-8 | 2^-8 |
| epochs / updates | 56 (best DEV epoch kept) / 86240 | 39 / 60060 |
| best params fnv | 9348584540880858499 | 1908154717227521143 |
| image fnv | 9232652764948363499 | 3416091170583141300 |
| dataset fnv bound in image | 16622604375676129865 | same |

## Learning (owner run, loss sums in binary32 decoded)

| arm | TRAIN loss sum | DEV loss sum |
|---|---|---|
| MLP | 11.09 | 2.41 |
| affine | 71.60 | 7.26 |
| shuffled targets (MLP) | 265.42 | 61.32 |
| no-update (MLP, lr 0) | 335.04 | 75.46 |
| untrained (MLP) | not trained | (selection quality below) |

Learning changes parameters and lowers held-out DEV error by more than an order of
magnitude against the shuffled and no-update controls.

## Selection quality (owner run, development-only; `owner-run/eval_*.stdout`)

Cost = sum over records of the chosen arm's ps per call. Ratios are to the oracle
(cheapest admissible arm per record). TEST = families H1 + H2 + H3 (360 records).

| chooser | TEST X925 (180) | TEST A725 (180) | TEST all (360) | DEV all (70) | top-1 TEST |
|---|---|---|---|---|---|
| fixed rule (incumbent) | 1.559 | 1.447 | 1.489 | 1.814 | 0 / 360 |
| always reference | 1.515 | 1.092 | 1.252 | 1.265 | |
| best static (quad4) | 1.0048 | 1.0137 | 1.0103 | 1.0065 | |
| retrieval (nearest TRAIN shape) | 1.0001 | 1.0041 | 1.0026 | 1.0146 | 320 / 360 |
| affine, trained | 1.0001 | 1.0679 | 1.0423 | 1.0008 | 309 / 360 |
| **MLP scratch, trained** | **1.0001** | **1.0185** | **1.0116** | **1.0005** | 306 / 360 |
| shuffled-target control | 1.0122 | 1.1055 | 1.0703 | 1.0591 | 194 / 360 |
| untrained control | 1.0917 | 1.0304 | 1.0536 | 1.1675 | 204 / 360 |

Paired per-record wins/losses of the MLP on TEST: vs fixed rule 359/1, vs reference
197/27, vs best static 109/27, vs retrieval 14/28.

Reading, honestly:

- The trained MLP is 0.68 of the incumbent fixed rule's cost and within 1.2 % of the
  oracle; it beats the affine baseline (1.042) and both controls.
- It does **not** beat the cheap heuristics: on A725 it is 1.0144 x retrieval and
  1.0047 x best static; on X925 it ties retrieval exactly (same choices on all 180 records).
  Against ORIG-3's preregistered rule (non-inferiority to min(best static, retrieval),
  no tolerance; NO-GO above 1.01) this development result points to **NO-GO**. The
  sealed-record evaluation decides; nothing here was tuned on TEST.
- Both controls also "beat" the fixed rule (0.72 and 0.71 of it) because the rule picks
  the two slowest arms; a win over the incumbent alone is therefore not evidence of
  learning, which is why the prereg gates on the heuristics and the controls.

## Resource ledger (owner run, host)

| item | measured |
|---|---|
| experience collection (consumer harness, tap on) | 62 s wall, shared machine, 20222 verified runs, amortized over every run below |
| compile (4 deps + 6 units, two callers, two compilers) | inside the 21 s script wall; binaries 90 KiB |
| tests | 274 ms, 50329 checks |
| lr sweep (6 trainings) | 9.5 s |
| training, MLP | 2.81 s for 56 epochs / 86240 updates (33 us per update, software binary32) |
| training, affine | 0.21 s for 39 epochs / 60060 updates |
| checkpoint I/O | 11560 B image (MLP), 1816 B (affine); save + load inside the run times |
| evaluation (load + admit 738 records + features + scoring + baselines) | 126 ms MLP, 71 ms affine: about 0.17 ms per record, versus a fixed rule in C measured in nanoseconds |
| memory, separately | params 705 words, gradients 705 words, best copy 705 words, activations 152 words, epoch order 2048 words, shuffle table 16384 words, record buffers 512 KiB raw + 512 KiB decoded, checkpoint buffer 32 KiB; process max RSS 320 KiB (`/usr/bin/time -v`) |
| failed experiments | 0 abandoned runs; one mutant rewritten during development because it was semantically equivalent (noted in the glue) |
| energy | NOT_MEASURED |

## Not covered

Independent, sealed evaluation (ORIG-3); the default-off shadow consumer (ORIG-4);
low-rank or phase-dependent forms (later ablation, not done); any GPU run.
