# SI19-ORIG2-v2: v2 training and evaluation on C2 (aien-train#25)

Frozen by benchmarks-mv `manifests/prereg/orig3_selector_v2.json` (commit `0ca6b9c`, pushed
2026-10-25T12:08:30Z) before the data existed. C2 = one post-freeze run of the omega-mv
`85641e1e` harness with `OMEGA_EMPIRICAL_COVERAGE=600` (14:15:15Z to 14:16:25Z), 1938 records,
fnv `6887491069590809475`, local only. Host CPU only; GB10 NOT_RUN.

```
V2=1 RECORDS=<C2> ALLOW=<benchmarks evidence/ORIG3-SELECTOR-v2/allow-v2.txt> \
OUT=evidence/SI19-ORIG2-v2/run BUILD=/tmp/ov2/v2build nice -n 19 sh tests/si19/run-select.sh
```

| item | result |
|---|---|
| v2 admission (`c2-admission.stdout`, omexpmain mode V) | 1714 admitted, 224 refused (-15 duplicate state); TRAIN 976, VALIDATION 315, HELD-OUT 423 |
| candidate MLP (705 params, seed 19024, lr 2^-8) | 64 epochs, 312320 updates, 10.0 s, VALIDATION loss 11.82; image `candidates/mlp-scratch-v2-seed19024.img` sha256 `16abb8fc...965c` |
| shuffled control | 17 epochs (early stop), 2.7 s, VALIDATION loss 297.5; `candidates/v2_ctrl_shuffled.img` |
| untrained control | 0 epochs, params = declared init; `candidates/v2_ctrl_untrained.img` |
| mode E per-record files | `run/v2_eval_{mlp,shuffled,untrained}.records` (VALIDATION + HELD-OUT) |
| verdict (benchmarks-mv `bench_orig3_judge_v2`) | **NO-GO**, controls fail (valid run); see benchmarks-mv `evidence/ORIG3-SELECTOR-v2/README.md` |

Glue incident (disclosed): the final `v2_cross_v1` check expected exit 68, but on C2 the v1
reader refuses the file first (family-5 coverage records, exit 66), so the script stopped
after every training and evaluation step had finished. The remaining bookkeeping lines of
`run/summary.txt` (image hashes, raw.sha256, build.sha256) were produced by hand with the
same commands; nothing was rerun. The glue now accepts 66 or 68 for that refusal.
