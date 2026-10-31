# ORIG-1 contract v2: integrity admission and workload-disjoint splits

aien-train#24. Supersedes sections 2 (admission), 6 (splits) and the retrieval row
of section 7 of `ORIG1-CONTRACT.md` for every v2 run. `ORIG1-CONTRACT.md` itself
is unchanged and still governs the v1 runs and their evidence. Everything not
named here (consumer, record layout, features, target, normalization, budgets,
provenance, allow-list) is copied unchanged from v1.

## 1. Why a v2 (audit of the v1 data)

`omexpmain` mode `A` over the v1 owner file (A, fnv `16622604375676129865`) and
the v1 SEALED-FINAL file (B, fnv `12602815733741495377`), evidence in
`evidence/SI19-ORIG1-v2/`:

| counter | value | reading |
|---|---|---|
| B records whose workload (core, M, N, job seed) occurs in A | 738 / 738 | the harness regenerates the same jobs from fixed seeds: SEALED-FINAL re-measured exactly the workloads of the development file |
| of those, same oracle arm | 710 / 738 | the re-measurement mostly repeats the label |
| B state keys (core, pressure, M, N) present in A | 738 / 738 | |
| A duplicate state keys | 155 | TRAIN measures every shape twice (154) plus one repeated H2 shape |
| A workloads under two family words | 240 | H3 is the H1 jobs under pressure |
| A H1/H2 records whose log2 shape family holds a TRAIN record of the same core | 116 / 240 | half the "unseen" shapes sit in a family that TRAIN covers |

No TEST (M, N) key occurs in TRAIN, so the v1 candidate did not train on test
shapes. The v1 sealed set was nevertheless not a fresh sample: it repeated the
development TEST workloads whose numbers had been inspected in ORIG-2, and half
of them share a shape family with TRAIN. That supports, but does not by itself
prove, the retrieval parity seen in v1: the per-family decision is smooth enough
that a neighbour in the same log2 cell carries the label.

## 2. Admission v2 (`oxadm2`)

Whole-file refusals (nothing admitted): size not a positive multiple of 256 or
above 524288 bytes (-1); a split conflict (-11, section 3). Otherwise every
record gets one code, the first failing check in this order, and the count of
each code is reported (`RQ`):

| code | reason |
|---|---|
| -2 | unsupported format: magic or version |
| -3 | unsupported enumeration: family word above 5, core class other than 0 (Cortex-X925) or 1 (Cortex-A725), pressure bucket above 1, arm count other than 6 |
| -4 | shape or calls out of range: M or N outside 1..8192, M N above 2^19, calls outside 1..2^20 |
| -9 | wrong profile: cache bytes not the class value, state bytes not (M N + M + N) 8 |
| -14 | unpaired: the record is not the complete arm set of this state: arms 0..4 eligible (mask exactly 31), measured and verified; arm 5 (planted) not eligible, not measured, not verified; arms 6, 7 absent |
| -10 | out of range: a measured ps per call at or above 2^40 (the integer record's analogue of a non-finite value) |
| -7 | unverified: digest mismatches during the table (word 31 != 0) |
| -8 | sequence not above the previous record's |
| -15 | duplicate state: an earlier admitted record has the same state key (core class, pressure bucket, M, N); the job seed is operand data, not state |
| 0 | admitted |

Pairing is per record because one omexp-v1 record carries every arm of one
state; a paired comparison needs a cost for any arm any policy may choose.

## 3. Workload-family splits (frozen)

Family of a record: `(fm, fn) = (floor(log2 M), floor(log2 N))` (`oxcell`).
Split of a family: `h` = FNV-1a 64 over the two bytes `fm, fn`; `k = (h >> 32) mod 10`;
k 0..5 TRAIN, 6..7 VALIDATION, 8..9 HELD-OUT (`oxhspl`). The table over the
whole domain (168 families, 100 / 32 / 36) is `docs/si19/orig1-v2-splits.txt`,
printed by `omexpmain` mode `S` and compared byte for byte by
`tests/si19/run-omexp.sh`. The record's family word (word 2) is not used.

Consequences: a shape, and every pressure bucket and job seed of it, lives in
exactly one split; held-out families are absent from TRAIN and VALIDATION by
construction, and `oxdisj` re-checks every admitted pair (same family or same
shape in two splits refuses the file, -11). Refused records are in no split.

## 4. Retrieval v2

The nearest TRAIN record of the same core class **and the same pressure
bucket** by L1 on (lg16 M, lg16 N), lowest index on a tie; its cheapest
admissible arm if admissible in the decision, else arm 0. (v1 ignored pressure.)

## 5. Tests

`tests/si19/omexp-test.om` (`tadm2`, `trefuse2`, `tdup2`, `tsplit2`, `tnear2`,
`taudit`): every code above through a single-word mutation that refuses only that
record and counts it once; duplicates (same key with another seed, a later
duplicate of a refused record, two later duplicates); split values computed
outside the unit; pairwise disjointness; a planted split conflict; v2 retrieval
pressure; the audit counters. `tests/si19/run-omexp.sh` adds 10 v2 mutants to the
6 v1 mutants; each must fail the unchanged test.
