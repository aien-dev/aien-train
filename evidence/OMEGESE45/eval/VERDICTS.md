# Omegese45 independent evaluation: first trained-checkpoint verdict

Date: 2026-10-29. Implementation `3c6e165`, frozen oracle sources/seed/set committed in `8e3b27c9`. All EXERCISED measurements here are **GB10 Spark Grace CPU host**, cores 10-14, nice 19, ionice c3. No GPU, no RL, no production-service action. This is a bounded workflow evaluation, not a general model or Hive qualification.

## Provenance and machinery

The [final full-control command/output](review2-controls.log) and [checkpoint-free DET/RET command/output](review2-detret.log) both exited 0, `MACHINERY PASS`. The runner reads only the original committed sealed/seed hashes; [hash-check](controls/hash-check.txt) confirms the unchanged `oe45lib.py` and `oe_sealgen.py` too. No model number preceded a change to those frozen sources because **neither source was changed**. The later v1.3 external shape wrapper rejects illegal refusal placement without changing sealed answers; [spec audit](controls/spec-audit.txt) reports no disagreement on 20000 train, 1000 dev or 517 sealed episodes.

[Input digests](controls/inputs-sha256.json) bind the data, model CLI and checkpoint slots. Step-zero ck.a: `02d85006cd96c9c450c66147538a286b07954964fa14552fb40501fcf77225f2`; trained ck.a: `2ebb99de9a0aa04dfba95e0f008e392b99c3f4c4c3836086e24cf7b3a8204058`. [ck0 metadata](controls/ck0-info.txt) and [ck1 metadata](controls/ck1-info.txt) retain the actual CLI output. Pipeline final source branch is `feat/45-pipeline` at `a3ee223bb1644e2111812e9d7afcd203c4c5312d`; see that lane's lineage receipt for generator seeds and supervised training. No checkpoint was selected or updated from these sealed results.

[Fresh-clone suite](review2-fresh-suite.log): `OE45_EVAL_TESTS PASS`, `OE45_CONTROL_TESTS PASS`, **11 mutants KILLED**, `OE45_EVAL_MUTANTS PASS`, `OE45_EVAL_PASS`, rc 0. The fresh clone used the exact contract vocabulary at the default integrated path (sha256 in the receipt), not a bypass flag. [Missing default vocabulary negative](review2-missing-vocab.log): rc 1, `contract.vocab_tsv_present FAIL`, `OE45_EVAL_FAIL`; the file was restored after the negative. This is machinery PASS, not model-correctness PASS.

## All controls, full thresholded set

The 507 thresholded episodes are also the complete goal-disjoint subset: [overlap report](controls/overlap.txt) has zero exact goals and zero full episodes in either train or dev. Counts are copied from the runner's [summary](controls/summary.txt); all per-unit Wilson 95% intervals, including binding-needed, are in [intervals.tsv](controls/intervals.tsv). Zero denominators are NOT_DEFINED, not success.

| Control | Validity | Action exact | Closed loop | Completion | Recovery | Bound precision | Bound recall | Retry | Report exact | Binding needed |
|---|---|---|---|---|---|---|---|---|---|---|
| DET | 1256/1256 | 1256/1256 | 507/507 | 206/206 | 247/247 | 92/92 | 92/92 | 122/122 | 507/507 | 0 |
| RET | 1120/1256 | 53/1256 | 23/507 | 0/206 | 0/247 | 23/53 | 23/92 | 1/122 | 30/507 | 665 |
| U0 | 0/1256 | 0/1256 | 0/507 | 0/206 | 0/247 | 0/0 | 0/92 | 0/122 | 0/507 | 0 |
| G0 | 1256/1256 | 68/1256 | 0/507 | 0/206 | 0/247 | 68/381 | 68/92 | 0/122 | 0/507 | 297 |
| U1 | 1256/1256 | 1088/1256 | 379/507 | 118/206 | 176/247 | 90/91 | 90/92 | 79/122 | 430/507 | 90 |
| G1 | 1256/1256 | 1088/1256 | 379/507 | 118/206 | 176/247 | 90/91 | 90/92 | 79/122 | 430/507 | 90 |

Binding-needed count is before any dispatcher repair. The denominator for its descriptive rate is emitted grammar-valid job A's: DET 657, RET 665, U0 0, G0 297, U1/G1 658. These are not counts of executed microjobs. U1 and G1 are identical on this checkpoint and set; the grammar mask does not account for their gain over G0. That gain does not satisfy the accuracy thresholds.

XCLASS remains **separate and unthresholded**: U1/G1 action exact 19/20, closed loop 9/10, report 10/10. [xclass.tsv](controls/xclass.tsv) reports each R pair's measured training occurrence count and full sequence membership. Every XCLASS R line is public grammar seen in training; no XCLASS full episode appears in train/dev. It is not an unseen R-line composition. The dated clarification in `EVAL-CONTRACT.md` explicitly records that it was written after DET/RET/U1/G1 results existed; no threshold, metric, denominator or sealed artifact changed.

## Unchanged three verdicts

1. **Workflow correctness: FAIL, not INVALID.** [Wolfram threshold receipt](wolfram-verdicts.txt): W1 FAIL, W2 FAIL, W3 PASS, W4 FAIL, W5 FAIL, W6 PASS. G1 closed-loop Wilson interval is [0.707954872992, 0.783391308227], below the predeclared W1 lower-bound threshold. W7 is also FAIL on the actual model demo snapshot: [independent runtime checker receipt](review2-runtime-check.log) finds happy-pair UNFAITHFUL because the model report repeats seed2 instead of seed1. Happy-mm, bound, transport, replay, repair, restart and translator cases are FAITHFUL; receipt and English tampering are both caught. This receipt checks the existing real-runtime artifacts and does not relaunch the jobs. A later demo run cannot erase the measured sealed failure.
2. **Scientific Turing: NOT_QUALIFIED; T NOT_RUN.** No calibrated probability stream, two independent real coders or frozen Hive-specific profile. `docs/omegese45/TURING-HIVE-PROFILE-DRAFT.toml` is a review proposal, not an adopted scoring profile.
3. **Resources: measured report, no pass threshold.** Exact wall time and peak RSS are in the command receipts, not estimated from throughput. Training/runtime resource receipts belong to their lanes. Eval user/system CPU seconds were not recorded by this command's time format and are NOT_RUN here. GPU and energy NOT_RUN. Host EXERCISED is not device QUALIFIED.

Wilson verification: [three boundary/interior cases](wilson-verification.txt) and [every reported nonempty interval](wolfram-all-intervals.txt), with exact WolframLanguageEvaluator queries and responses. The comparison of all printed endpoints against Wolfram returned True. Intervals are dimensionless and unit-specific; correlated decode-point intervals are descriptive.

The model failed honestly under the original bar. There was no retraining, checkpoint selection, threshold relaxation, hidden postprocessing or conversion of a model failure into a deterministic-policy success. Reference controls and runtime checker safety controls remain valuable even when the learned policy fails.
