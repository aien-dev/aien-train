# ORIG-1: original mini-model selector task and owner-controlled experience dataset

aien-train#24, child of #19. Frozen before any ORIG-2 training. A change to any
numbered rule below is a new contract version, never an edit of this one.

Status words used here: PASS, FAIL, INCONCLUSIVE, NOT_RUN, BLOCKED. Host-only
results are development results, never hardware qualification.

## 1. The consumer (pinned)

| item | identity |
|---|---|
| consumer | the Omega empirical cost model's matvec realization choice, `rx_cm_decide` in `src/runtime/rx_costmodel.{c,h}`, exercised by `tests/runtime/rx_empirical_optimizer.c` (`make test-empirical`) |
| repository, revision | aien-dev/omega-mv `18c653b3101fa09c157dca8d222efab41ab5e89d` (main, 2026-10-25); the consumer files are byte-identical to `a0eafa39` |
| record tap | `tests/runtime/rx_empirical_optimizer.c` at `1ec813726aef5f5abf8ab449c5d8bd9738f88d2e` (branch `feat/731-orig4-shadow-advice`): `OMEGA_EMPIRICAL_RECORDS=<path>` appends one record per cost table; default off; nothing in the consumer reads it |
| admissible-candidate generator | the consumer's `eligible` bit mask: an arm is admissible only when its forked differential verification passed (`g_verified`) and the caller allows it. The reference arm (0) is always admissible |
| transformation family | integer matvec `y = A x` realizations on one AArch64 core: 0 `reference` (compiled), 1 `scalar`, 2 `unroll2`, 3 `unroll4_dual` (M12 emitters), 4 `quad4` (R10), 5 `planted_wrong` (never admissible). Cap: 8 arms (`RX_CM_ARMS`) |
| incumbent planner | `omega_matvec_dispatch`'s fixed rule: `unroll2` when N <= 8, else `unroll4_dual` |
| verifier | every timed run's output digest equals the reference digest for that job (`time_arm`); a table with any mismatch is ineligible |
| cost | picoseconds per call, best of 5 rounds per arm, rotating order (`truth_table`) |
| hardware profile | DGX Spark host: `Linux 7.0.0-1019-nvidia aarch64`; core classes from MIDR: 0 Cortex-X925 (last private cache 2097152 B), 1 Cortex-A725 (524288 B), 2 other (1048576 B) |
| compiler for the .om side | seeded `omc` sha256 `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9` (the pin of `docs/si19/dependencies.json`, R5b, source omega-mv `29b12760`); the R6 compiler `734f0a39...6b61` (omega-mv main `seed/out/omc`) must also compile every unit unchanged |

The scorer ranks the admissible arms of one decision. It never proposes an arm,
never widens `eligible`, and has no execution authority: deterministic code
generates, verifies, selects and runs.

## 2. Record format `omexp-v1`

One record is 32 little-endian 64-bit words (256 bytes). A file is a whole
number of records, at most 2048.

| word | field | rule |
|---|---|---|
| 0 | magic | 3553143591248351811 (`0x314f4d4558505243`) |
| 1 | version | 1 |
| 2 | family | 0 train grid, 1 validation grid, 2 H1 unseen shapes, 3 H2 thin-heavy mix, 4 H3 = H1 shapes under memory pressure |
| 3 | core class | 0, 1 or 2 |
| 4 | pressure bucket | 0..15; the consumer uses 0 and 1 |
| 5, 6 | M, N | 1..8192 each |
| 7 | calls | timed calls per measurement, >= 1 |
| 8 | job seed | the consumer's job seed (operands are regenerated from it) |
| 9 | state bytes | (M*N + M + N) * 8 |
| 10 | cache bytes | must equal the class value of section 1 (profile check) |
| 11 | thermal C | recorded, never a feature |
| 12 | eligible mask | bit 0 must be set; no bit at or above the arm count |
| 13 | arm count | 1..8 |
| 14 | sequence | strictly increasing within a file |
| 15 | monotonic ns | recorded, never a feature |
| 16..23 | ps per call, arm 0..7 | 0 = not measured; an eligible arm must be measured |
| 24..31 | per arm verified flag (24..29), reference digest (30), mismatches (31) | word 31 must be 0 |

A decision instance is one record: the state is words 2..13, the candidates
are the arms whose eligible bit, measurement and verified flag are all set.
Incorrect results are ineligible (word 31 != 0 refuses the record). An
unmeasured cost is unknown (0), never a fabricated number.

## 3. Typed features (frozen)

All quantities are integers in 1/16 units, converted to binary32 exactly with
`wmbits n 4`. `lg16 v` for v >= 1 with bit length L and e = L - 1 is
`(v << 4 >> e) + 16 (e - 1)`: floor(log2 v) plus a truncated 4-bit fraction.

State vector x (12):

| j | value | note |
|---|---|---|
| 0 | 16 | constant 1 (bias) |
| 1 | lg16 M | |
| 2 | lg16 N | |
| 3 | (lg16 M * lg16 N) >> 8 | log2 M * log2 N / 16 |
| 4 | (lg16 N * lg16 N) >> 8 | (log2 N)^2 / 16 |
| 5 | 16 if N <= 3 else 0 | |
| 6 | 16 if N rem 4 != 0 else 0 | |
| 7 | max(0, lg16 state_bytes - lg16 cache_bytes), capped at 256 | doublings past the private cache |
| 8, 9, 10 | 16 for the matching core class 0, 1, 2 | one hot |
| 11 | 16 * min(pressure, 15) | |

Candidate vector c (8): 16 at the arm's index, 0 elsewhere. The model input is
x (affine: per arm) or x concatenated with c (nonlinear). Thermal state, calls,
seed, sequence and time are never features. These are the consumer's own
basis (spec/empirical-optimizer.md, "Features and targets") plus core and
pressure as inputs instead of cells.

## 4. Target and objective (frozen)

Per (record, admissible arm a): `t16 = lg16 ps_a - lg16 ps_0`, the cost of the
arm relative to the always-admissible reference, in 1/16 log2 units. The
reference anchors the scale, and argmin over arms of t16 is argmin of ps.

Normalization is fitted on the TRAIN partition only: `mu16` = floor of the mean
of t16 over train pairs; `sh` = max(0, bitlen(s16) - 5) where s16 is the
integer square root of the mean of (t16 - mu16)^2. The learning target is
`y = (t16 - mu16) / 2^(4 + sh)` as binary32, exact while |t16 - mu16| < 2^24.
The predicted score is in the same units; selection is the arm with the
smallest score among admissible arms, lowest index on a tie. The loss is the
squared error of the score against y, summed over the admissible arms of a
record and averaged over records.

## 5. Provenance, authorization, privacy

- Random initialization only. No pretrained weights, teacher outputs or logits,
  distilled answers, external model traces, or pretrained tokenizer anywhere in
  the primary arm. The test `tprovenance` fails if the checkpoint image carries
  any lineage word other than the declared seed and the admitted dataset hash.
- A dataset file is admitted only when its FNV-1a 64 over all bytes appears in
  `docs/si19/orig1-datasets.txt` with status `AUTHORIZED` (owner measurement) or
  `SYNTHETIC` (project fixture). `REVOKED` refuses; an absent hash refuses.
  The list is public and holds hashes only.
- The owner dataset stays local: `~/.local/state/aien-train/orig1/` on the
  Spark (mode 0600). Public artifacts carry its hash, record counts and the
  receipts of runs over it, never the bytes. Releasing it is a separate owner
  decision; it holds no personal data (hardware timings of a public harness).
- Synthetic fixture: `native/si19/omexpfix.om` writes a labelled SYNTHETIC
  record file from a seeded generator with a planted cost structure. Fixture
  evidence and real-record evidence are reported under separate headings and
  never mixed in one number.

## 6. Splits (frozen, by family and shape lineage)

| partition | families | lineage rule |
|---|---|---|
| TRAIN | 0 | the consumer's training grid, M in {1,4,16,64,256,1024}, N in {1,2,3,5,8,12,16,33,64,100,256,512,1024}; two timing passes of each shape stay together |
| DEV | 1 | validation grid, M in {2,8,32,128,512}, N in {4,6,10,24,48,200,700}: no M or N value shared with TRAIN |
| TEST-H1 | 2 | 60 log-uniform shapes per core with no M or N value seen in TRAIN or DEV |
| TEST-H2 | 3 | thin-heavy mix: unseen M values; 70% reuse TRAIN N values in 1..7 (declared partial lineage overlap) |
| TEST-H3 | 4 | the H1 shapes under memory pressure, a condition absent from TRAIN and DEV (out-of-distribution) |

The loader refuses a file in which any TEST-H1 (M, N) key also appears in
TRAIN or DEV (future/held-out contamination). The trainer admits only TRAIN
records; DEV is used for the lr choice and early stopping; TEST is read only
by the evaluation mode after the checkpoint is final. There is no sealed
judge-held holdout on this machine: every TEST number here is
development-only. ORIG-3 obtains a sealed test set by running the tap again
after the candidate checkpoint is frozen (the records do not exist at training
time), which is the only sealing available without an independent operator.

## 7. Baselines (frozen)

| name | choice |
|---|---|
| fixed rule (incumbent) | arm 2 when N <= 8 else arm 3, if admissible, else arm 0 |
| always reference | arm 0 |
| best static | the single arm with the lowest total TRAIN cost for that core class |
| retrieval | the TRAIN record of the same core class nearest in (lg16 M, lg16 N) by L1 distance (lowest sequence on a tie); choose its cheapest admissible arm |
| affine | per-arm linear scorer on x (8 x 12 = 96 parameters), trained as in ORIG-2 |
| oracle | the cheapest admissible arm of the record (analysis bound only) |
| controls | untrained scorer (declared init), shuffled targets (t16 permuted among train pairs by the trainer RNG), no-update run (lr = 0; parameters must stay bit-identical) |

Metric: total chosen cost over oracle cost, sum of ps per call over the
partition's records (unweighted by calls), per core class and overall; top-1
agreement with the oracle; paired per-record wins and losses against each
baseline.

## 8. Budgets and stop rules (predeclared)

| item | limit |
|---|---|
| seeds | trainer 19024; init from the same LCG as `wsrand` |
| nonlinear model | at most 100000 trainable parameters (engineering budget, not an optimum); ORIG-2's first model is 705 |
| learning rate | one of 2^-8, 2^-10, 2^-12 chosen on DEV loss; nothing is chosen on TEST |
| epochs | at most 64 full passes over TRAIN, per-record order from the trainer RNG; stop early when DEV loss has not improved for 16 epochs |
| updates | at most 2^20 |
| data | at most 2048 records per file; at most one owner dataset per run |
| memory | program image under 16 MiB; checkpoint under 64 KiB |
| wall | compile + tests + one training run under 30 minutes on the host |
| failed experiments | counted in the evidence README; a failed or inconclusive run is kept |

A hardware (GB10) run is not part of ORIG-1 or ORIG-2: NOT_RUN. Any later
Spark experiment needs an exclusive reservation and never loads a second model
beside the resident service.

## 9. Acceptance mapping

| #24 acceptance | where |
|---|---|
| versioned manifest, exact identities | `docs/si19/orig1-manifest.json`, this file |
| provenance tests | `tests/si19/omexp-test.om` `tprovenance`, `tallow` |
| refusal tests: revoked, unauthorized, corrupt, duplicate, wrong profile, contaminated | `tests/si19/omexp-test.om` `trefuse*` |
| family/episode isolation, holdout boundary testable | `tsplit`, `tleak` |
| baselines, units, fixture vs real, stop rules explicit | sections 4, 7, 8 |
| consent-safe dataset local, public fixtures synthetic | section 5 |
| missing real experience or sealed holdout = BLOCKED/development-only | section 6; receipts say `development-only` |
