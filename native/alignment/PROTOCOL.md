# NL-2: model-specific projection experiment

Refs: [aien-train #34](https://github.com/aien-dev/aien-train/issues/34), [architecture #349](https://github.com/aien-dev/aien-architecture-mv/issues/349). This is the isolated latent-projection ablation reserved by #18, not the shared-base specialist pilot #26/#27 or an inference engine.

## Frozen plan before execution

1. Use existing seeded Omega scalar f64 arithmetic, fsqrt, fixed pools and bounded loops. Fit separate H_model -> P_model -> Z_shared projections. No device, capture, checkpoint or serving calls.
2. Compare full-rank orthogonal Procrustes (reflections allowed), truncated spectral least squares (principal-component regression, PCR), and a learned nonlinear quadratic feature projection. The last is nonlinear in its input, not an MLP or a trained transformer. Do not call PCR the generally different optimal reduced-rank regression.
3. Keep training and held-out latent rows disjoint. Independent Wolfram constructs representations and exact expected projectors. A held-out synthetic algebra check is not evidence of model/task generalization. Freeze all constants and fixtures before running; do not adjust them from held-out output.
4. Exercise wrong basis, null and shuffled-label controls, lossy rank, stale bindings, nonfinite input, bad shape and convergence/rank refusal. Run a changed-coefficient mutant and require the independent expected result to fail.
5. Publish only synthetic arrays, source/compiler/binary digests, commands, exits and oracle call/result evidence. A numerical vector proposes information; it carries no authority and this code executes no proposed operation.

## Source and scope

Training base: aien-train `b6c33e3a82611361b37c651df70d9a645045773a`, origin `git@github.com:aien-dev/aien-train.git`, default `main`.
Seeded compiler: omega-mv `1fe5976d398646b17acabbf6d9dfae96b017e439`, committed `seed/out/omc` SHA256 `d6290c7663c2b4bafb59eb3fdf812476e977f4c144def71095c6248ef9b96b84`. The file is unchanged relative to that commit. This resolves draft omega #919's compiler identity: it is this committed newer compiler, not documented historical `2c824805` or `49923639`. Its `.om` source is `seed/omc/omc.om`; its scalar contract is `docs/one_layer/OMEGESE.md` section 9.9. No compiler rebuild or trust-policy change.

The bounded experiment supports at most 32 paired rows, four input features and four shared coordinates, all f64, fixed row stride four, with an explicit caller-supplied coordinate convention and no automatic centering or intercept. Quadratic lifting supports two raw coordinates into four features. These are declared host experiment bounds, not support for unmodified transformer dimensions. A future real-model adapter must supply an owner-qualified bounded readout or obtain an explicit capacity/composition extension through Omega #663; it cannot silently truncate dimensions.

## Frozen numerical contract

The exact derivations, fixture arrays, thresholds and independent expected results live in `evidence/wolfram.jsonl` (raw tool-call/tool-result events). Alignment is normalized residual sum of squares (NRSE), with a separate maximum absolute feature error and orthogonality residual. NRSE is undefined on zero target energy and must refuse. No cosine-only success criterion.

Cyclic symmetric Jacobi diagonalization is bounded to 64 sweeps, at most four features. Stop only when all off-diagonal terms are within `1e-14 * max(1,max(abs(diagonal)))`; nonconvergence refuses. Retain eigenvalues above `1e-10 * largest`; requested rank must be available. Procrustes requires square, full-rank cross covariance. Frozen synthetic thresholds: NRSE <= `1e-12` and max absolute error <= `1e-6`. These are preregistered choices, not measured hardware/model tolerances. Numerical oracle verifies the formulas and constants before execution.

All five bindings (model revision, layer/readout definition, tokenizer revision, precision profile, projection version) must match before a projection is used. Changing any one yields refusal, never a cosine-based acceptance. Schema IDs here are synthetic fixture bindings, not a second transport; any later message uses NL-3's canonical envelope and existing microjob authorization. Binding checks in this host harness are experiment-integrity checks, not cryptographic authorization.

## Real-model preregistration: BLOCKED

Keep three experimental modes separate:

- **Hidden-state readout:** internal residual-stream tensors captured at a precise native operation boundary. Not messages, text or proof of reasoning.
- **Learned messages:** sender-generated numerical communication trained against the shared task labels. Not necessarily a transformer hidden state.
- **Latent reasoning/recurrence:** repeated internal-state computation. NOT_RUN here; neither alignment nor message decoding demonstrates it.

Before any capture, freeze checkpoint/source/weight/tokenizer/chat-template digests, model architecture and text-only layer numbering, state dimension/shape, exact pre/post-normalization tap, last-nonpadding-token aggregation, precision/quantization/scale, native backend revision, deterministic settings, consent scope and data split manifest. Do not mix KV or GDN state, model weights or recurrence with residual readouts.

Use public synthetic task identifiers, paired by semantic task and byte-span rather than assuming tokenizer positions are compatible. Fit each projector using training only; use validation to select rank/regularization, then seal projections and all identities before NL-6's independent single-use holdout. Never fit or select on the final holdout. Compare within-model cross-layer, cross-model, wrong basis, identity/null, shuffled labels and frozen random controls; report same-task feature preservation, exact typed operator/shape/dtype/effect recovery, independent authorized task completion, abstention and catastrophic wrong-operation count. The deterministic typed checker, not the numerical channel, grants semantic acceptance; existing authority decides effects.

For model/layer/tokenizer/precision/projection changes, first report changed-field vector and mismatch count, refusal count and accepted-stale count (required zero). Refit as a new preregistered version, never silently reuse. On admitted paired calibration data report new versus original NRSE and max error, cross-projection drift, orthogonality residual where relevant, retained rank, condition/eigenvalue profile, held-out feature and task fidelity, refusal rate and wrong-operation rate. Report quantization/noise separately with the exact profile. No comparison treats a model rename as basis compatibility.

Record complete capture, fitting, validation and inference costs: wall/CPU/device time, all calls/retries, raw/readout/projector/cache/scratch bytes, cold/warm IO, encode/project/decode/verify overhead and amortization assumptions. Host wall time here is process observation, not GPU latency or energy. Real-model p50/p95, memory headroom and lifecycle economics remain NOT_RUN until instrumented by existing owners and independent NL-6 evaluation.

## Exact access request (not authorization)

Request owner-reviewed, private-local, read-only activation access for these candidates, in this order:

| Candidate | Exact requested readout | Present evidence / blocker |
| --- | --- | --- |
| `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` | text residual after decoder blocks 0, 5, 11, 17, 23, before final normalization, last nonpadding token | Public text config: 24 blocks, hidden width 1024, declared bfloat16, hybrid linear/full attention. Core #737 reports staging only; actual local checkpoint/tokenizer/precision digests and native capture qualification unavailable. BLOCKED. |
| `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc` | text residual after blocks 0, 5, 11, 17, 23, before final normalization, last nonpadding token | Public text config: 24 blocks, hidden width 2048, declared bfloat16, hybrid linear/full attention. Native owner Omega #694 must admit the actual local checkpoint and tap. Not acquired here. BLOCKED. |
| private reference 27B | block 0, exact middle block and final block, each post-block/pre-final-normalization, last nonpadding token | Exact model ID/revision, block count, source license, precision, tokenizer and native readout manifest must come from Core #735/#724; nickname is not identity. BLOCKED. |
| private reference 35B | same three relative taps, with exact integer middle/final block IDs frozen by owner before capture | No live tap or serving modification authorized. Exact manifest and separate offline capture path required. BLOCKED. |

Public metadata read on 2026-10-28: [0.8B revision](https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B/revision/main), [2B revision](https://huggingface.co/api/models/Qwen/Qwen3.5-2B/revision/main), [0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json), [2B config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json). Apache-2.0 is declared by those public metadata records. This is a source inventory, not a local checkpoint or license-admission receipt. The owner must match the local hashes and text-only config before capture; an unavailable or changed layer refuses and requires a revised preregistration. Relative private-reference taps are frozen into integer IDs by the owner before any run, never chosen from alignment outcomes.

Capture method: existing qualified native Omega/Core prefill path emits bounded, explicitly allowlisted residual slices into an owner-controlled local file, with task ID, model/config/tap/tokenizer/precision digests and record shape. No public HTTP instrumentation, private conversations, raw user traces, KV export or activation publication. The existing native owner must implement/qualify the tap against a pinned independent reference, including wrong-layer and wrong-checkpoint mutants. This PR does not install a tap or call a model.

Resource request: one explicitly scheduled offline GB10 exclusive window of 30 minutes, sequential candidates only, after existing whole-Spark memory ledger and spark-window/quietlock procedure. The resident service may not be interrupted by this request; if an uncontended exclusive window cannot coexist, defer until its owner separately authorizes a maintenance window. No second large model is loaded beside it. Requested limits: only the fixed synthetic calibration corpus, no capture of live sessions, no checkpoint downloads without distinct acquisition approval, no public endpoint, no production promotion. Exact peak bytes and approved start/end must be supplied by the native owner before opening the window; the duration is a scheduling request, not a runtime estimate.

Required approvals: Drake/operator for the exact window and resources; Omega #694/#695/#663 and Core #735/#724/#737 owners for the model identity, tap and native correctness/memory admission; data owner under Open Humanity #26/#33 for exact capture package, retention and separate training use, including withdrawal. Capture consent does not authorize publishing raw activations. Derived projectors inherit provenance and withdrawal/quarantine obligations. Without all approvals and manifests, real-model results remain **BLOCKED**, not FAIL, PASS or synthetic substitutes.

## Dependencies and status

Product/runtime dependencies: only the pinned existing seeded Omega compiler and Linux AArch64 host. Wolfram hosted service is an independent verification oracle, not shipped runtime. Git, gh and crumb are read-only/source custody and contribution tooling. No foreign trainer, numerical library, compiler, scheduler or authority is introduced. Synthetic validation is host-only; model-native, latent-reasoning, cross-model activation, GPU, energy, production and deployment claims are NOT_RUN/BLOCKED. No shared Omega guide semantics, compiler, protocol or ownership changes are made by this bounded experiment.

## Host API and replay

`project.om` is a reusable bounded unit, not a model loader. Populate `HX` and `ZY` at `4*row+column`, and `HM[0..4]` with the five immutable identity fields. `fit(n,d,o,rank,mode)` learns a candidate and installs `HP`, shape `HS` and identity `HI` only after successful validation. Mode 1 is square orthogonal Procrustes; mode 2 is truncated spectral PCR. `lift(n)` expands raw two-coordinate inputs to `[x,y,x*x,y*y]` before a mode-2 fit. `measure(n)` checks all bindings and finite data, stores residual sum, target energy and maximum error in `HE`, and refuses zero energy. Return codes: 0 success, 1 bounds/shape, 2 nonfinite input or result, 3 nonconvergence, 4 insufficient retained rank, 5 unready/stale identity, 6 zero target energy. A failed fit preserves the prior installed projector, not a newly authorized use of it. Pools are process-local and not reentrant.

Replay on Linux AArch64 with the exact compiler above:

```sh
cat native/alignment/project.om native/alignment/check.om > /tmp/nl2-check.om
/path/to/pinned/seed/out/omc < /tmp/nl2-check.om > /tmp/nl2-check
chmod +x /tmp/nl2-check
/tmp/nl2-check
```

The integer-only output is lossless: each line is `case field unsigned-binary64-bits`, except cases 10 (binding mismatch counts), 997 (check ID and pass flag for every check), 998 (field 0 failed-check count, field 1 total-check count), and 999 (failed-check ID). Fields 0/1/2/3 are residual sum/target energy/max error/NRSE; field 4 is orthogonality error where relevant; fields `100+4*i+j` are coefficients and `200+4*r+j` are held-out predictions. No decimal formatter or foreign numerical implementation sits in the executed path.

Cases 0/1 are identity and independently rotated representations with separately fitted projectors. Case 2 is the rank-two minimum-norm map from `[u,v,u+v]`; case 3 learns quadratic feature coefficients from nonlinear targets. Cases 4/5 are raw-linear and reversed-training-label controls against the same nonlinear held-out target. Cases 11/12/13 exercise an incompatible identity basis, null map and wrong transpose. Case 14 drops a genuine coordinate with rank-one PCR. `evidence/run.json` contains structured native per-check records/counts, exact source/binary digests, raw output and mutant failure counts. Mutants are temporary verification inputs, never an alternate product path. Canonical independent numerical evidence is now `evidence/oracle-return-values.jsonl` and its returned value in `evidence/oracle-values.json`: each query returns a value explicitly, not `Print` output followed by `Null`.

The NL-6 benchmark contract was frozen at benchmarks `8ee0a494bf73230c9d37647eb3caea792298fa8d` before host results. This algebra experiment does not claim the separate multi-seed task campaign, superiority, economics, real-model compatibility or independent task-completion acceptance. Numerical alignment alone cannot qualify those claims.

## Observed synthetic result

Native host campaign: **PASS**, exit 0, total-check count 63 and failed-check count 0 (`evidence/run.json`: `total_check_count`, `failed_check_count`, `per_check_results`). WolframLanguageEvaluator's self-contained query returns an explicit JSON value in `evidence/oracle-return-values.jsonl`; lossless decimal tokens are also filed as strings in `evidence/oracle-values.json` and `run.json` under `numerical_evidence.cases`. Cases 0/1 have exact zero held-out NRSE; case 2's returned `native.NRSE` is `1.8304038191456289e-31`; case 3's is `4.2872875283750640e-32`. Case 3's returned `native.max_error` is `1.7763568394002505e-15`. These are synthetic unitless coordinates, not measured model activation tolerances. Exact returned projector matrices are in each case's `oracle_matrix_exact`; the oracle independently reconstructs them from the fixture, without using native coefficients as ground truth.

Independent exact control oracles: nonlinear raw-linear NRSE `41/46`, reversed-label NRSE `10/23`, wrong-basis/null/transpose NRSE `2/1/4`, rank-one NRSE `1/2` (`oracle-values.json`: each case's `oracle_NRSE_exact`). Native observations agree within the frozen thresholds and ordinary f64 rounding; the raw values, rather than rounded exact claims, are retained. Each of the five identity changes produces mismatch count 1 and refusal. `run.json.negative_controls` retains every per-check result, failed-check ID and count: removed metadata guard = 5 failures; negated learned coefficients = 16 failures; each has 63 checks and exit 1. The returned Wolfram value independently recomputes the metrics from 160 held-out prediction values and compares 52 coefficient values and 40 metrics, rather than accepting native self-report alone.

Qualification limits: only the stated bounded synthetic families are exercised, not arbitrary ill-conditioned matrices. Nonconvergence refusal is implemented defensively but **NOT_RUN** as an observed failure in these convergent fixtures; rank, shape, nonfinite, zero-energy and identity refusals are exercised. No performance or condition-number superiority is claimed. Real-model alignment remains **BLOCKED** for the exact access prerequisites above; latent recurrence, GPU, energy and end-to-end authorized model-task execution remain **NOT_RUN**.

### Evidence correction

The first draft's `Print`/`Null` oracle record and unstructured escaped stdout were insufficiently auditable. They are superseded, not retroactively qualified: the original receipt is retained unchanged as `evidence/run-41589735-superseded.json`, with its historical oracle files. The corrected campaign emits every check result and both total/failure counts natively. The corrected Wolfram query returns all numerical comparisons as an explicit JSON value, which is copied without arithmetic into the current `run.json`. Historical narrative summaries are not the acceptance oracle.

PR #35 remains a **draft with zero GitHub checks**, not merge-qualified. No merge is requested or performed. An executed local host receipt does not silently replace repository merge policy.
