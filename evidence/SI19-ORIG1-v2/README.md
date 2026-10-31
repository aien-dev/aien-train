# SI19-ORIG1-v2: integrity audit and v2 admission (aien-train#24)

Host run of `tests/si19/run-omexp.sh` at source commit `e95862f` (see
`run/summary.txt`), compiler `191b776e...` (pin), the same test unit also under
R6 `734f0a39...`. Host CPU only; no GPU, service or model process touched.

```
D=~/.local/state/aien-train
OUT=$PWD/evidence/SI19-ORIG1-v2/run RECORDS=$D/orig1/records-20261025T034737Z.bin \
AUDIT_A=$D/orig1/records-20261025T034737Z.bin AUDIT_B=$D/orig3/records-sealed-20261025T042409Z.bin \
nice -n 19 sh tests/si19/run-omexp.sh      # exit 0, SI19_ORIG1_RUN_PASS
```

| item | result |
|---|---|
| `omexp-test` (pin and R6) | PASS, 11654 checks each |
| mutants | 16/16 killed (6 v1, 10 v2: pairing, duplicate, unsupported core, split by family word, range, disjointness, sequence, retrieval pressure, duplicate key without pressure, audit oracle) |
| split table (mode S) | 168 families, 100 TRAIN / 32 VALIDATION / 36 HELD-OUT, identical to `docs/si19/orig1-v2-splits.txt` |
| v2 admission, SYNTHETIC fixture 780 | 468 admitted; 310 duplicate state (-15), 2 shape (-4) |
| v2 admission, v1 owner file 738 | 583 admitted; 155 duplicate state (-15): the second TRAIN pass of every shape plus one repeated H2 shape |

## Audit: v1 SEALED-FINAL against the v1 development file

A = owner file (fnv `16622604375676129865`, sha256 `403156f5...8d37`), B = SEALED-FINAL
(fnv `12602815733741495377`, sha256 `99da7bd9...7fd7`); `run/audit.stdout`.

| counter | value |
|---|---|
| B workloads (core, M, N, job seed) present in A | 738 / 738 (TRAIN 308, DEV 70, H1 120, H2 120, H3 120) |
| matched B records with the same oracle arm as A | 710 / 738 |
| B state keys present in A | 738 / 738 |
| A duplicate state keys | 155 |
| A workloads under two family words (H1 = H3 jobs) | 240 |
| A H1 + H2 records whose log2 shape family holds a TRAIN record of the same core | 116 / 240 |

Finding: the v1 sealed set was a re-measurement of exactly the development
workloads (the harness regenerates jobs from fixed seeds), not new workloads.
There was no TRAIN/TEST shape overlap, but half of the "unseen" shapes share a
log2 shape family with TRAIN, and the TEST workloads re-measured as SEALED had
been inspected during ORIG-2. The v1 verdict (INCONCLUSIVE) stays as sealed; it
is evidence about re-measurement noise, not about generalization to new
workload families. v2 (`docs/si19/ORIG1-CONTRACT-v2.md`) splits by workload
family and refuses unpaired, duplicate and unsupported records with counts.
