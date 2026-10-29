# aien-train #45: independent evaluation contract (eval lane)

Status: the scoring contract and sealed commitment were predeclared at interface v1.1, before any trained-model result (commit `8e3b27c9`). Current interface: **v1.3**. The sealed-set commitment in [`evidence/OMEGESE45/eval/SEALED-COMMITMENT.txt`](../../evidence/OMEGESE45/eval/SEALED-COMMITMENT.txt) was pushed with this contract before model results. Later clarifications are dated explicitly below, not represented as preregistration.

Scope: the first #45 deliverable only: one Omegese-45 (R45) workflow, one allowlisted Aienet-0 job (`matmul_u32`, m, k, n in 1..=256, u64 seed), executed on the merged Aienet-0 microjob path. Nothing here evaluates general agent competence, Hive multi-agent configurations, a transformer, or GPU execution; those are NOT_RUN by design (interface §10).

Sources: the parent-owned shared interface v1.3 (vocabulary R45, grammar, episode format, reference policy, exit codes); aien-train issue [#45](https://github.com/aien-dev/aien-train/issues/45); omega-mv `bf54b19f` `calibration/profiles/Turing-profile-v1.1.toml` and `docs/turing/protocols/turing-instrument-calibration-validation-protocol-v1-0.tex` (verdict 2); aien-protocols `e705fdc9` (demo checker).

## 1. Independence and edit authority

- The eval lane is the only writer of `docs/omegese45/EVAL-CONTRACT.md`, `docs/omegese45/TURING-HIVE-PROFILE-DRAFT.toml`, `eval/omegese45/`, `tests/omegese45/eval/` and `evidence/OMEGESE45/eval/` (interface §1).
- **The sealed set, this contract's metrics and thresholds, and the sealed-set generator are outside the proposers' edit authority.** The proposers are the pipeline lane (data generator, model, training), the translator lane and the demo lane. The set is frozen by the sha256 values in `SEALED-COMMITMENT.txt`. Any change to the sealed file, the seed, the generator sources (`eval/omegese45/oe45lib.py`, `eval/omegese45/oe_sealgen.py`), the metric definitions (§5) or the thresholds (§6) after this commit makes the evaluation INVALID (§7), unless the change is a new, separately committed sealed set made before any result on it is seen.
- Ground truth never comes from a proposer's code. The eval lane wrote its codec, grammar parser, binding check and reference policy (`eval/omegese45/oe45lib.py`) and sealed generator (`eval/omegese45/oe_sealgen.py`) independently from the original interface. Their committed bytes remain frozen. Current v1.3 shape refinements live in `oe_episode.py`, called by both `check-demo` and `oe_audit.py`; no sealed artifact or reference answer is changed. Pipeline output supplies only train/dev retrieval, overlap and spec checks (§4).
- The pipeline lane never reads the sealed files (interface §7). They live outside every repository until the reveal, in `/home/drakestapleton/handoffs/oe45-sealed/` (directory mode 700, files mode 600).
- Python standard library is used for these oracles. This is allowed for eval oracles (interface §0) and keeps the ground truth in a different language and code base from the `.om` product path, so a shared implementation error is less likely.

## 2. Message and episode facts used (interface v1.3)

Ids per message, from D = `ushort 0x H H H` (5 ids) and S = `ulong 0x H H H H` (6 ids): G mm 24, G pair 30, A job 25, A refuse 4, R 3 or 4, P ok mm 25, P ok pair 31, P refused/fail 4. MAXLEN = 176; decode output budget = 31. The longest episode the reference policy can produce is 175 ids (pair, each job `res fail transport` then `res ok`, then `rep ok pair`); the sealed set contains it (the audit receipt reports `maxlen_ids=175`).

Runtime class of each C' code (the class `aienet0-run` reports, interface §2/§5): `replay`, `budget`, `worker` → `refused` (exit 11); `transport`, `receipt`, `recomp`, `verdict`, `io` → `fail` (exits 10, 12-15). The grammar also admits the other class for any C'; the reference policy mirrors whatever R it is given, and only `res fail transport end` triggers the single retry.

**2026-10-29 v1.3 shape clarification, after trained-checkpoint numbers existed:** an `act refuse` is the first and only A, directly after G, directly followed by `rep refused bound end`; P directly after G is invalid. The new `oe_episode.check_episode_shape` wraps the frozen v1.1 oracle and enforces this stricter rule through the real checker entry point. All committed sealed episodes already satisfy it. `oe45lib.py` and `oe_sealgen.py`, the commitment, all thresholds, metrics and denominators remain unchanged.

## 3. Sealed-set construction

Generator: `eval/omegese45/oe_sealgen.py` (`oe_sealgen v1`). Input: one 32-byte lineage seed (64 hex characters). Randomness is SHA-256 in counter mode keyed by the seed and the stratum label, with rejection sampling for uniform integers, so the output depends on nothing version-specific. Output: one bundle file; each episode is a line `@ <id> <stratum>` followed by its `OE45 v1` text.

Lineage: generator = sha256 of `oe_sealgen.py` and `oe45lib.py` (in the commitment file); lineage seed = 32 bytes from `os.urandom`, sha256 committed, seed withheld until the reveal. It is a different generator lineage from the pipeline's (interface §7).

Strata (517 episodes):

| Stratum | n | Construction |
|---|---|---|
| EDGE | 30 | each dim position (m, k, n) × {0x000, 0x001, 0x100, 0x101, 0xfff} × {mm, pair}; other dims uniform in 1..256; admitted goals get all `res ok` |
| CODE | 21 | each of the 7 non-transport C' codes at its runtime class: mm job 1, pair job 1, pair job 2 |
| RETRY | 51 | 3 replicas of: mm {T ok, T T, T X for each of 7 X}; pair {(T ok, ok), (ok, T ok), (T ok, T ok), (T T), (ok, T T), (T ok, T T), (T X1), (ok, T X2)} where T = `res fail transport` |
| PAIRREPLAY | 5 | pair with seed1 = seed2; job 2 is the same document, so `res refused replay` |
| XCLASS | 10 | grammar-legal R with the class the runtime never reports for that code (`res fail replay`, `res refused transport`, …). Reported separately, never thresholded |
| RAND | 400 | kind mm/pair 1/2 each; dims uniform 1..256; with p = 1/5 the goal is out of bound (3/4 one bad dim, 1/4 two; a bad dim is 0 with p = 1/8, else uniform 257..4095); seeds uniform 0..65535 (pair seeds distinct); per attempt `res ok` 12/20, `res fail transport` 3/20, a uniform non-transport C' at its runtime class 5/20; retry only after the first transport failure of a job |

Compositions covered include pair episodes with a retry on both jobs, retry followed by another stopping code, equal pair seeds, two out-of-bound dims, the 0xfff edge and XCLASS. Their novelty is measured, never inferred from construction. XCLASS's generic R strings are public grammar and are deliberately present in pipeline training; XCLASS is not unseen at the R-line level.

Metric denominators on the thresholded set (all strata except XCLASS, 507 episodes), counted by the eval code: 1256 decode points, 507 report points, 122 decode points directly after `res fail transport end`, 92 out-of-bound goals.

**Reveal.** After every control and model result on this set is committed, the eval lane commits the seed and the sealed file under `evidence/OMEGESE45/eval/sealed/`; anyone can rerun `oe_sealgen.py` and compare both sha256 values with the commitment. Once revealed, the set is spent: no later checkpoint may be selected or reported on it as sealed.

## 4. Overlap and leakage checks

Run before scoring; receipts under `evidence/OMEGESE45/eval/`.

1. **Hash check.** sha256 of the sealed file and the seed equal the commitment; otherwise INVALID.
2. **Exact-goal overlap.** Count sealed episodes whose G line appears byte-for-byte in the train split (and in the dev split separately).
3. **Exact-episode overlap.** Count sealed episodes whose full OE45 v1 text appears byte-for-byte in train (and dev).
4. **Primary scoring subset.** Thresholds (§6) are evaluated on the goal-disjoint subset: thresholded-set episodes whose G line is absent from train and dev. Full-set numbers are reported beside it.
5. **Lineage check.** The pipeline's train/dev lineage seeds and generator sha256 are recorded; the sealed lineage is a different generator, so seed reuse cannot produce identical streams.
6. **Spec check of the pipeline generator.** Every train and dev episode is run through `eval/omegese45/oe_audit.py` (codec, grammar, shape, MAXLEN, reference-policy agreement given its own R messages). Disagreements are reported per episode. A disagreement rate above 0 is reported as a pipeline defect; it does not by itself invalidate the evaluation, because the sealed ground truth is independent.
7. **No sealed data in training.** If an exact-episode overlap exceeds what goal collisions explain, or any sealed-only XCLASS line appears in train, the evaluation is INVALID for leakage.

   **2026-10-29 parent clarification, written AFTER trained-checkpoint numbers existed** (DET closed-loop 507/507, RET 23/507, U1=G1 379/507 already seen): “sealed-only XCLASS line” means sealed-specific content, not any of the 16 public grammar-legal `res <class> <code> end` strings derivable from the R rule and `vocab.tsv`. The pipeline deliberately trains all these strings as grammar coverage (`PIPELINE.md` outcome distribution). Their presence alone is not leakage. Sealed goals and exact episodes/full message sequences are audited against train/dev; unexplained sealed-specific overlap is leakage. The runner records XCLASS's per-pair train R counts and each full sequence's train/dev membership in `xclass.tsv`. XCLASS remains separate and unthresholded. This clarification changes no threshold, metric, denominator or sealed artifact; it prevents mistaking public grammar coverage for disclosure of sealed data.

## 5. Controls and metrics

Controls, all on the same sealed set (interface §8):

| Control | Definition |
|---|---|
| DET | eval reference policy `policy_next` (deterministic control; its score is also a machinery check, §7) |
| RET | retrieval: for each sealed episode, the train episode whose G has the smallest goal-token Hamming distance (ids compared position by position; a length difference counts each missing position as a mismatch; ties broken by first in train order); its model messages are copied verbatim in order without substitution, and when the copied episode has no message for a decode point the control emits nothing (invalid) |
| U0, G0 | step-0 (untrained) checkpoint, unguided argmax / DFA-guided argmax |
| U1 | supervised-trained checkpoint, unguided argmax |
| G1 | supervised-trained checkpoint, DFA-guided argmax |

A decode point is every position in a sealed episode whose message is a model message (A or P). The model input is the sealed prefix up to that position.

Metrics (each reported with a Wilson 95% interval computed by Wolfram, query quoted):

| Metric | Unit | Definition |
|---|---|---|
| validity | decode point | the emitted message parses as a model message (A or P) under the eval grammar; syntax only |
| action exact match | decode point | emitted text equals the reference message exactly (teacher-forced R) |
| closed-loop success | episode | starting from G, the model's messages are fed the sealed episode's scripted R; success iff every model message equals the reference; the first mismatch ends the episode as failure |
| completion | episode | closed-loop success restricted to episodes whose reference report is `rep ok` (unseen goal completion) |
| recovery | episode | closed-loop success restricted to admitted-goal episodes containing at least one non-ok R |
| bound-refusal precision / recall | episode, first decode point | positive = emitted `act refuse bound end`; reference positive = out-of-bound goal |
| retry correctness | decode point | decode points directly after `res fail transport end`: exact match (an identical retry after the first transport failure, `rep fail transport end` after the second) |
| report exact match | decode point | decode points whose reference is a P: exact match |
| binding-check needs | count | emitted `act job` messages that fail the interface §3 binding check against the sealed G and current job (would be refused 41/42 before dispatch) |

Validity is reported separately from every behavioural metric so a grammar mask cannot be counted as learned reasoning. The G-versus-U difference and the trained-versus-untrained difference are both reported.

## 6. Predeclared thresholds: workflow-correctness verdict

Evaluated for G1 (the mode the demo uses) on the goal-disjoint subset (§4.4). Wilson 95% lower bound (LB). For scale: with the full thresholded set, LB ≥ 0.90 requires at least 470/507 episodes, 117/122 retry points and 89/92 refusals (Wolfram `wl[x_,n_]:=With[{z=InverseCDF[NormalDistribution[],0.975],p=x/n},N[(p+z^2/(2n)-z Sqrt[p(1-p)/n+z^2/(4n^2)])/(1+z^2/n)]]; SelectFirst[Range[0,n], wl[#,n]>=0.9&]` → 470, 117, 89 for n = 507, 122, 92).

| # | Requirement |
|---|---|
| W1 | closed-loop success LB ≥ 0.90 |
| W2 | completion LB ≥ 0.90 and recovery LB ≥ 0.90 |
| W3 | bound-refusal precision LB ≥ 0.90 and recall LB ≥ 0.90 |
| W4 | retry correctness LB ≥ 0.90 |
| W5 | report exact match LB ≥ 0.90 |
| W6 | learned, not masked: G1 closed-loop success Wilson interval lies entirely above the G0 and RET intervals |
| W7 | live demo (interface §9): every required case ends with its documented rc; `eval/omegese45/check-demo` reports FAITHFUL and all artifact checks passing for each executed case; the tampered-receipt case is caught; no unbound action was dispatched |

U1 is reported against the same numbers but is not thresholded (the demo decodes in mode G).

## 7. Verdicts

1. **Workflow correctness** (§6): PASS iff W1-W7 all hold; FAIL if any fails; INVALID if a machinery check fails: the commitment hashes differ (§4.1), leakage (§4.7), DET scores below 1.0 on any metric of the thresholded set, or the eval tests (`tests/omegese45/eval/run-eval.sh`) do not print `OE45_EVAL_PASS`. A correctness FAIL cannot be offset by a better predictive score or lower cost (issue #45, "Three separate verdicts").
2. **Scientific Turing**: NOT_QUALIFIED for this deliverable. No frozen Hive-specific Turing profile exists; the CTR1 profile is specific to its own outcome stream and does not qualify R45 episodes; and the MSE-trained scorer provides no calibrated codelength, so no T score is computed (NOT_RUN). The eval lane delivers `docs/omegese45/TURING-HIVE-PROFILE-DRAFT.toml` as a DRAFT for its owners to freeze.
3. **Resources**: CPU seconds, peak RSS, bytes of checkpoint, data and codec, and model/tool call counts from the lanes' receipts, on the GB10 Spark Grace CPU (cores 10-14, nice 19). GPU and energy are NOT_RUN. This verdict is a measured report with no pass threshold.

## 8. What a FAIL means

- The trained model, as evaluated, does not reliably execute the R45 workflow on goals it has not seen. The demo cases may still run end to end (the runtime, binding check and validator are independent of the model), and they are reported as EXERCISED with the workflow verdict FAIL.
- A FAIL triggers no weight update, reward, preference signal or automatic retraining (issue #45, "no reinforcement learning"). Any later supervised change is a new training run with declared data, evaluated on a new sealed set; this set is spent once revealed.
- A FAIL is published with the same receipts as a PASS; failed metrics and regressions are never omitted.

## 9. Independent demo checks (`eval/omegese45/check-demo`)

For a demo receipt directory the checker, independently of `aienet0-run`'s VERIFIED line: parses each AN0B bundle and MJR1 receipt from the byte layouts in aien-protocols `e705fdc9` (`crates/aien-microjob/src/{lib.rs,aienet0.rs}`, `specs/microjob/SPEC.md`); re-derives m, k, n and seed and checks them against the G line; checks the MJR1 outcome; verifies the P-256 signature over bytes [0, 340); recomputes the `matmul_u32` output digest from the seed per `specs/aienet/jobpath/PROFILE.md`; parses the English response with an eval-owned parser of the translator's fixed templates and compares every stated fact with `facts.tsv` and the receipts (FAITHFUL / UNFAITHFUL). Negative controls: one tampered receipt byte, one tampered output digest and one edited English number must each be caught (`tests/omegese45/eval/run-eval.sh`).

## 10. Executable evaluation and checker trust root

`eval/omegese45/oe_controls.py SEALED SEED TRAIN DEV CORE CK0 CK1 OUTDIR` runs the whole committed sealed set, not a hand-picked subset. `CORE` is the compiled `oe45core` CLI and CK arguments are checkpoint directories. Pass `- - -` for CORE/CK0/CK1 to run DET and RET without checkpoints; the four model controls are explicitly NOT_RUN. A fixture checkpoint may test plumbing but is never a scored trained result. The runner refuses any changed sealed/seed/frozen-oracle hash before scoring, records hashes of every supplied artifact and checkpoint generation metadata, audits all train/dev/sealed episodes, reports all controls and both thresholded and goal-disjoint subsets, and reports XCLASS separately and unthresholded.

Outputs: `summary.txt`/`counts.json` retain integer numerators and denominators; `intervals.tsv`/`intervals.json` give two-sided Wilson 95% intervals (zero denominators are NOT_DEFINED, not zero success); `xclass.tsv` gives per-pair public R exposure and full-sequence membership; `spec-audit.txt`, `overlap.txt` and `inputs-sha256.json` disclose provenance. `binding_needs` remains a count; its descriptive interval uses emitted grammar-valid job A's as the explicitly reported `binding_opportunities` denominator. Points within an episode are correlated; point intervals are descriptive, not an independence claim or a T confidence interval. The local Wilson implementation is checked against Wolfram at zero, middle and full success, with exact queries/results in [`wilson-verification.txt`](../../evidence/OMEGESE45/eval/wilson-verification.txt).

`check-demo` pins coordinator submitter/verifier P-256 public keys, coordinator Ed25519 public identity, worker Ed25519 public identity and executor P-256 pin as constants, from the existing demo setup's public identity exports. Case-provided keys are evidence to compare, never a trust root. Public values are in `eval/omegese45/check-demo` and the runtime checker receipt records their match; no private key is committed. Tests use a visibly separate `tests/omegese45/eval/check_fixture.py` import harness for synthetic identities. The production entry point has no override option or environment variable. A regenerated key set with freshly re-signed valid receipts fails only `keys.trust_root`, and the test must fail if that check is removed.

Every non-ok facts row must also match an actual dispatch outcome's job, attempt, exit and dag, and a `logs/*-run.err` failure diagnostic for that dag. A replay additionally needs a `done` entry in raw `worker/seen`; adapter summaries alone cannot establish replay. Restart can leave more runner logs than outcome rows, so evidence is matched by dag/failure, never by lexicographic log/outcome position. The forged-replay-after-real-execution negative must remain UNFAITHFUL even if the adapter's outcome line is also forged.

`tests/omegese45/eval/run-eval.sh [VOCAB]` defaults to the integrated `docs/omegese45/vocab.tsv` and fails if it is absent. In the isolated eval-only branch, pass the contract lane's exact vocabulary as the explicit argument; final integration verification uses the default path. The suite calls the real checker entry-point implementation for shape, binding, trust-root, replay and English negatives, then proves sensitivity with mutants. A PASS is eval machinery correctness, not workflow qualification. Measured controls and the unchanged W verdicts live under [`evidence/OMEGESE45/eval/`](../../evidence/OMEGESE45/eval/); the Hive compression proposal is [`TURING-HIVE-PROFILE-DRAFT.toml`](TURING-HIVE-PROFILE-DRAFT.toml), not a frozen profile or a T result.
