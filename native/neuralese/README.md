# NL-1: synthetic numerical channel, not authority

Tracks [train #33](https://github.com/aien-dev/aien-train/issues/33), under [architecture #349](https://github.com/aien-dev/aien-architecture-mv/issues/349). This directory alone is owned by this change. The shared-base/adapters work in #26 / draft #27 is separate and unchanged. Numerical messages carry information only; no execution, authorization, consent, model activation capture, GPU use, or service integration is implemented here.

## Result labels

- **PASS / EXERCISED:** exact Wolfram matrix reproduction, independent native numerical comparisons, bounded native learning execution, admission checks, and killed gradient/update/discrete-causality mutants. These are synthetic CPU-host results, not general trainer qualification or device qualification.
- **FAIL:** the frozen learned model does not reach all-16 training reconstruction and does not beat the fixed or discrete controls on this public parity split. A running program is not a successful learned-language experiment.
- **INCONCLUSIVE:** transfer/compositional-language/economic-benefit hypotheses. This tiny public split is not the NL6 hidden confirmatory campaign and does not establish a population-level failure either.
- **NOT_RUN:** real model states, private activations, secret campaign, authorized completed work, device/GPU behavior, energy/DRAM/network measurement, deployment, and production authority.

## Exact sources and facility reuse

Base aien-train revision: `b6c33e3a82611361b37c651df70d9a645045773a`. The explicit derivative / staged-check-commit SGD facility originates in `native/ecosystem6/affine.om` at `c7642d96c0b84b6d176dbb2dc8558b536cabe729`, owned by train #3 / #6. This is a bounded shape/nonlinearity/binary64 extension of that facility, not a second generic optimizer. Existing ecosystem6 paths are not edited. Its older FP32 qualification is not inherited as qualification of this extension.

The inspected omega revision is `e79566c191ea8ef11888e69ce50d8fadf375d4fd` (draft #919). Its `mk/autodiff.mk`, `mk/tensor.mk`, and `mk/reservoir.mk` select C implementations, not an available seeded handwritten `.om` trainer. Omega #663 is not evidence that a general native trainer already exists. No missing primitive blocked this bounded implementation, so it was implemented rather than filed as an external dependency.

The actually executed seeded compiler SHA-256 is `d6290c7663c2b4bafb59eb3fdf812476e977f4c144def71095c6248ef9b96b84`, source `seed/omc/omc.om` SHA-256 `3f937160c823d3a90a6b4cc296f00f517f36ede7fd27a06505c998985ec62240`, introduced at omega `1fe5976d398646b17acabbf6d9dfae96b017e439`. This is **not** the older documented `2c824805...` or `49923639...` compiler. No compiler was rebuilt or substituted. The pinned compiler is a verification/build dependency, not a hidden C/Python learning implementation.

Freeze history: `66dcca8fa22ca18ebdd5e7514c2fb10bb3d6ef7f` committed configuration before any learned outcomes. `2eba4f1e0be57c01eec280d0cd3563fe95f69c45` fixed only loop grammar after compiler error 8. `0e01c58a41f1a2fb976efa75c1cc554503aa2ee4` split function scopes after register exhaustion, error 12. No architecture, hyperparameter, data, split, or threshold was changed in response to outcomes. `c5992ca` repairs a discovered discrete-control side channel, not the learned model: a real `uchar[1]` carrier is now the receiver's only input. Earlier discrete transport/collision claims are superseded; their original receipts remain historical.

## A. Independent matrix oracle

`oracle.wl` is the read-only Wolfram oracle, separate from product code. It evaluates the issue's literal S, Hadamard Kronecker construction W, selected-column E, H, decoding and quantization. `evidence/oracle.json` and the raw tool-call/result JSONL retain the query and result. The required verification calls were executed: `Integrate[x^2,{x,0,1}] -> 1/3` and WolframAlpha `sqrt(27)/2 -> 3 Sqrt[3]/2`.

**PASS:** dimensions S=3x4, W=8x8, E=8x4, H=3x8; ranks respectively 3,8,4,3. `Transpose[E].E == IdentityMatrix[4]` and `H.E == S` are exact. `Max[Abs[Round[16 H]/16 . E-S]] -> 1/(16 Sqrt[2])`, numerically `0.044194173824159220`. All 12 thresholded features recover. Thus the issue's approximate figure reproduces.

**Dimensional correction:** H contains three examples, each an **eight-coordinate message**. Its three rows do not make it a 3D communication channel. The learned experiment below is a genuinely different 4-to-3-to-4 claim.

Wolfram's matched row-basis permutation reconstructs exactly. Unmatched basis, sign, column and scale controls produce nonzero error. Exhaustive fixed sign-corner perturbations, quantization denominators 1,2,4,8,16,32,64 and shrinking first-row subsets are recorded, not inferred from a width slogan. For first-k-row channels k=1..8, ranks are 1,2,3,4,4,4,4,4 and distinct messages across all sixteen inputs are 5,9,15,16,16,16,16,16. The recorded projection error uses the **truncated transpose decoder**: it is not the best possible inverse. Full rank can permit a refitted left inverse; a finite codebook can fit many meanings in one real coordinate at a precision/robustness cost. No impossibility theorem for all 3D encodings follows.

`check.om` independently exercises all sixteen inputs at clean/1/16/1/32 and rejects wrong length, NaN, infinity and out-of-range messages without changing the sentinel output. Its sign/scale/coordinate controls must fail reconstruction. `verify.wl` decodes the native integer IEEE carriers and compares them with independently calculated exact Wolfram values: 192 feature comparisons, maximum difference `2.220446049250313e-16`, correct thresholded predictions. The earlier oracle JSON's NOT_RUN native-admission field is historical; later native execution receipts are the exercised evidence, not a rewritten history.

## B. Learned sender and receiver

`experiment.json` is the preregistered contract. `affine.om` has four input features, three softsign hidden/message coordinates, four bounded receiver scores, 31 parameters, explicit derivatives, sum-squared error, and staged finite/range-checked online SGD. The learning rate is 1/10, the epoch count 2000, order ascending input ID, deterministic xorshift initialization and seeds 17,29,43. No checkpoint, general autodiff dependency, foreign runtime or GPU is used.

Split 0 fits and evaluates all sixteen combinations: **TRAINING ONLY**. Split 1 trains even-parity IDs 0,3,5,6,9,10,12,15 and evaluates held-out odd-parity IDs 1,2,4,7,8,11,13,14. All sixteen cases are emitted for every applicable arm/mode/context.

Arms: 0 learned encoder+decoder; 1 frozen same-seed random encoder with trained decoder; 2 fixed first-three-row Hadamard-derived encoder with trained decoder; 3 direct four-bit semantic code in one actual byte with no learned dictionary/training; 4 seeded shuffled training targets, evaluated against true meaning. Arm 2 is a deliberately smaller fixed comparison, **not** the exact eight-coordinate oracle. Equal accounting means charge both sides and their state; it does not pretend these algorithms perform equal work.

Primary metrics and split/seed rules are frozen by NL6 at benchmarks-mv `8ee0a494bf73230c9d37647eb3caea792298fa8d` / [draft #56](https://github.com/aien-dev/benchmarks-mv/pull/56). Primary modes are clean, 1/16 and 1/32, context=1. Decoder context=0 ablates the learned receiver bias only, not arbitrary shared agent state. Sign change, unmatched coordinate permutation, eight fixed additive 1/32 sign corners, coarser quantization and coordinate/bit masking are explicitly exploratory controls.

### Frozen outcomes

Counts below are exact four-feature recovery **and not abstained**, out of eight held-out combinations; seed order is 17,29,43. Final repaired-stream Wolfram receipts are named `final-metrics-*`; historical first-stream receipts omit `final-`.

| Arm | Clean | 1/16 | 1/32 |
|---|---|---|---|
| Learned | 5/8, 5/8, 6/8 | 5/8, 5/8, 6/8 | 5/8, 5/8, 6/8 |
| Random frozen encoder | 2/8, 4/8, 5/8 | 0/8, 4/8, 5/8 | 2/8, 4/8, 5/8 |
| Fixed three-coordinate encoder | 8/8, 8/8, 8/8 | 8/8, 8/8, 8/8 | 8/8, 8/8, 8/8 |
| Discrete byte | 8/8, 8/8, 8/8 | not applicable | not applicable |
| Shuffled-label control | 0/8, 1/8, 0/8 | 0/8, 1/8, 0/8 | 0/8, 1/8, 0/8 |

All-sixteen fit learned reconstruction is 14/16 for every seed in the clean condition, not 16/16. It is a **FAIL of the full-training-reconstruction target under this frozen schedule**, not evidence of a compiler feature gap and not a tuned-away failure. The discrete full-fit control recovers 16/16. Learned clean held-out abstentions are 2,2,1; accepted counts 6,6,7. Thus accepted outputs still contain errors: the margin rule is not a safety certificate.

Metrics retain raw prediction correctness separately from accepted-correct counts. Feature MSE, five-bin feature confidence calibration error, abstention/coverage, exact IEEE-carrier collision pairs, distinct messages and per-call encoder/decoder timing are in every primary receipt. Calibration is descriptive on this tiny deterministic set; receiver scores were not externally calibrated probabilities. No confidence bound, IID sample claim or calibrated authorization claim is made. Zero exact carrier collisions does not imply robust semantic separation. Quantized random seed29 does collide even in the primary held-out cases.

`final-sensitivity-*` contains per-seed/split exploratory results. `worstCornerExact` is threshold-only correctness across every fixed corner for each input, **without** excluding margin abstentions; `casesAnyAbstain` is separate. Its field `maxFeatureErrorAmongAcceptedFrames` means syntactically admitted frames (status=0), not margin-accepted decisions. Learned worst-corner threshold counts are 5/8,5/8,6/8; masking the third coordinate gives 4/8 for each seed, and masking the second/third gives 2/8. These fixed corners do not estimate a stochastic corruption distribution. Masking reduces retained coordinates/bits; the receiver is frozen rather than retrained. The discrete shrink modes actually mask the transmitted byte, so their collisions are counted on the real carrier.

Admission requires exactly three finite coordinates in [-1,1]. Wrong count, NaN, infinity and out-of-range controls must refuse without changing receiver outputs; NaN gradient and invalid learning-rate controls must not partially commit parameters. The byte-causality control holds sender input 15 fixed, changes only transmitted byte 15 to 0, and requires changed decoded output. A direct-input-ID receiver mutant exits 77 before the learned experiment; the unmutated program exits 0. This prevents a truth-ID side channel from qualifying as communication.

### Gradient and update evidence

`gradientcheck.om` checks all 31 derivatives with central differences and checks staged updates. `gradient.wl` independently evaluates the same point with exact rational forward/explicit derivative calculations and central differences. `evidence/native-oracle.json` compares actual native carriers: gradient maximum difference `3.7333037248227125e-17`, update maximum difference `3.1488879340451627e-17`. The attempted general symbolic simplification timed out; its HTTP 504 is retained and **NOT_RUN**, not described as a symbolic proof.

Native gradient-sign and gradient-index mutants compile then exit 61; the update-sign mutant exits 63. These intentional nonzero results are **PASS of negative-control rejection**. Sources/recipes, binary/source hashes, statuses and actual output are retained in `evidence/mutants.json` and `commands-final.json`. Earlier compiler/transport failures remain visible, including wrong loop grammar, register pressure, a wrong compiler argv invocation, JSON truncation and malformed Wolfram transport. They are not successful experiment evidence.

### Plots, data and costs

`plots/quantization.svg` comes from exact Wolfram quantization queries. `plots/learned-message.svg` is an XY projection colored by the third coordinate. `plots/message3d.svg` is a labeled isometric 3D view with all sixteen input IDs, blue fit cases and brown held-out cases; `plots/message3d.csv` exposes all three coordinates. These are seeded executable observations, not an invented latent diagram. No cluster/generalization conclusion is inferred from their appearance.

Raw `R` rows log IEEE binary64 values as unsigned decimal integer carriers; `P` rows retain final parameters and `T` rows measured host training nanoseconds. The CSV is an audit representation, not a bandwidth-efficient wire protocol. `experiment.json` documents all fields, modes and thresholds. The final stream SHA-256 is `0245f0d9ba732c75d5a061e88157f8556479f9ff1ecc1405171bfdbfa636d7b0`.

The independently queried `costs.json` ledger is **MODELED**, not measured traffic: continuous payload 24 bytes, parameter values 248 bytes, hypothetical dense eight-row oracle basis 256 bytes, discrete semantic content 4 bits in its actual one-byte carrier; 32000 SGD steps for all-fit and 16000 for parity-fit. The actual fixed basis is computed, not silently charged as a free dense dictionary. Forward affine counts and nonlinear operations are recorded separately. Actual executable/source sizes and hashes are retained in the manifest. Runtime scratch, schemas, code, metadata, all sender/receiver state, dictionary construction, verification, refusals and abstentions cannot be dropped from economics. Training must be charged across a declared deployment lifetime; none is assumed here. No total-energy, end-to-end communication saving, Spark throughput or break-even claim is supported.

The independently rebuilt final experiment executable is **13456 bytes** (SHA-256 `ea8aeebe161bd3319a319de2d7cb836c3014a5d1ae887348ee3e5db768ef50bd`); NL6's separate scorer is **8704 bytes**. These are measured byte sizes, not payload savings or runtime-memory totals. Both sides of the learned channel and all experiment/control code are in that native executable.

## Replay

On an authorized CPU-only AArch64 host, use the exact compiler digest above. No source bootstrap, GPU reservation or network endpoint is needed. From this directory, the compile protocol reads source on stdin and emits the executable on stdout:

```sh
OMC=/path/to/exact/pinned/seed/out/omc
sha256sum "$OMC"
cat hadamard.om io.om check.om > /tmp/nl1-fixed.om
"$OMC" < /tmp/nl1-fixed.om > /tmp/nl1-fixed
chmod +x /tmp/nl1-fixed
/tmp/nl1-fixed
cat hadamard.om io.om affine.om gradientcheck.om > /tmp/nl1-gradient.om
"$OMC" < /tmp/nl1-gradient.om > /tmp/nl1-gradient
chmod +x /tmp/nl1-gradient
/tmp/nl1-gradient
cat hadamard.om io.om affine.om learn.om > /tmp/nl1-learn.om
"$OMC" < /tmp/nl1-learn.om > /tmp/nl1-learn
chmod +x /tmp/nl1-learn
/tmp/nl1-learn
```

Collect compiler and program statuses separately and never execute stale output after a failed compile. Scratch paths must be exclusive. For byte-identical assembly hashes, receipts use a newline between each complete file, including an additional separator after its existing trailing newline; this is semantically the same source as the `cat` example but a different assembly-byte hash.

For Wolfram replay, first execute the two verification calls above, then `oracle.wl` and `gradient.wl`. Every `*-receipt.json` retains the complete submitted Wolfram code with literal bounded raw inputs and the actual returned tool result; replay that `code` verbatim. Template substitution is JSON data transport only, never numerical calculation. `verify.wl`, `metrics.wl`, `sensitivity.wl` and `plot3d.wl` are read-only independent oracles; they are not linked into the native runtime. Tool output truncation is a failed receipt, not a usable JSON fixture.

Independent NL6 [replay receipt](https://github.com/aien-dev/benchmarks-mv/blob/e4ab4afecd62272f86d67043f74ffc8153ae703e/native/neuralese/nl1-independent.json) pins candidate `c5992ca0b015a233bca6f9af7af892a71c558403`, the exact compiler, source assembly, matching native binary and an independently emitted stream. Its stream differs in measured timing fields as expected; every selected raw-derived recovery counter matches the producer stream. The native scorer independently thresholds float carriers and enforces producer prediction/abstention consistency. Separate checker qualification reports 41/41 PASS including six guard-removal mutants; scope is HOST_SYNTHETIC_CHECKER_ONLY. This is algorithmic/checker independence, not a claim that model seats or vendors are independent. This change does not merge or activate an envelope/protocol consumer.

## NL-7: guaranteed structure versus learned structure

This follow-up is frozen separately from NL-1: benchmarks-mv
`989fb7a214d8e0d6c7ed04f5f48c45d68a43d5a3`, posted on BOARD at
2026-10-28T15:21:18Z before running a variant. See
[draft evaluation PR #57](https://github.com/aien-dev/benchmarks-mv/pull/57).
The base here is NL-1 `7edaca90`; the old experiment and its receipts are unchanged.
These are fresh **public** combination splits, not a secret confirmatory campaign.
The seed order below is 101,211,307. Held-out sets are
`{6,7,10,11,12,13,14,15}` and `{3,5,7,9,11,13,14,15}`;
each corresponding training set contains the other eight IDs.

Clean accepted-correct counts at a 192-bit wire budget:

| Variant | Full training fit /16 | Split 1 held-out /8 | Split 2 held-out /8 | Payload / wire bits |
|---|---|---|---|---|
| Exact `((2a+b)/3,c,d)` | 16,16,16 | 8,8,8 | 8,8,8 | 192 /192 here; 6/8,24/24,96/96 also tested |
| RM(1,3) | 16,16,16 | 8,8,8 | 8,8,8 | 8/192 here; padded to every matched budget |
| Learned additive + separation | 15,15,15 | 5,6,6 | 3,4,3 | 192/192 here; 6/8,24/24,96/96 also tested |
| Existing learned softsign | 14,14,14 | 0,0,1 | 2,2,1 | Same continuous budgets |
| Packed semantic control | 16,16,16 | 8,8,8 | 8,8,8 | 4/192 here; minimum actual wire 8 bits |

**EXERCISED:** the exact and RM constructions meet their frozen synthetic
recoverability thresholds. **NOT_SUPPORTED:** the additive learned success
hypothesis. It improves over this existing learned baseline on both fresh splits
at high precision, but neither fits all sixteen nor reaches 8/8 on either split.
An additive sender identity and separation loss do not guarantee a learned
receiver. No post-outcome tuning, checkpoint choice or split substitution occurred.

Wolfram confirms the proposed exact numbers: 16 distinct points, minimum
Euclidean distance `1/3`, strict nearest-codeword radius `<1/6`.
`Reduce[d>=1/3 && r>=0 && r<1/6 && d-r<=r,{d,r},Reals] -> False`
verifies the triangle-inequality margin; equality at `1/6` is not guaranteed.
Every six-axis `+/-1/12` test recovers all sixteen at both IEEE budgets.
The preregistered `1/6` boundary and `1/4` probes are descriptive, not passes
under the strict theorem. RM's independently enumerated parameters are
16 codewords, eight bits and minimum Hamming distance four. At each wire budget
and seed/split, all 128 single-bit flips recover and all 448 double-bit flips
refuse. These repeated exhaustive enumerations are not independent statistical
trials. Packed bits recover cleanly but all 64 single-bit and 96 double-bit
corruptions are accepted wrong.

The additive constraint holds algebraically, and the maximum observed absolute
coordinate composition residual across all nine fits is
`1.7763568394002504646778106689453125e-15`, independently decoded by Wolfram.
The explicit training-pair hinge and its derivative are retained; no held-out
pair enters the loss. The raw `D` summary retains minimum squared separation
and composition residual; `P` lines retain all 31 final parameters, and long
`D` lines retain actual transmitted bytes and decoded coordinate IEEE values.
At 2/8-bit quantization, learned values outside [-4,4] are **refused**, not clipped;
this particularly damages the additive arm. Three floats are not three bits:
IEEE binary32 costs 96 transmitted bits and binary64 costs 192. Fixed codebooks,
weights and schemas are separate receiver-side shared costs, not free compression.

`nl7-structure.om` implements the exact, RM and packed arms. `nl7-learn.om`
extends the existing staged Omega optimizer, using the original receiver and
initialization for both learned arms. Both receivers consume the byte carrier,
not the source truth ID. No Python or new foreign product dependency is used.
Build using the same pinned compiler described above:

```sh
cat io.om nl7-structure.om > "$WORK/structure.om"
"$OMC" < "$WORK/structure.om" > "$WORK/structure"
cat hadamard.om io.om affine.om nl7-learn.om nl7-learn-main.om > "$WORK/learn.om"
"$OMC" < "$WORK/learn.om" > "$WORK/learn"
chmod +x "$WORK/structure" "$WORK/learn"
"$WORK/structure" > "$WORK/structure.csv"
"$WORK/learn" > "$WORK/learn.csv"
cat "$WORK/structure.csv" "$WORK/learn.csv" > "$WORK/all.csv"
```

Run from this directory with an existing writable `$WORK` and pinned `$OMC`.
The native OM2 conductor consumes `nl7-guards.cases` from the repository root;
provide its original `seed/qualification/om2/fixtures` SHA constants read-only
as in NL-1. It passed two original guard programs and killed fourteen source
mutants (16/16 cases): mapping, tie handling, RM radius, finite/length/padding,
additivity, separation sign, actual carrier reads and confidence admission.
The separate NL-6 scorer extension passed 19/19 controls including nine
checker mutants: canonical syntax, completeness, duplicate, payload, split
membership, RM radius/expectation, exact noise guarantee and packed-bit oracle.

[`evidence/nl7-run.json`](evidence/nl7-run.json) records compiler/source/binary
hashes, statuses, full archive hashes and freeze history.
[`evidence/nl7-stream.csv.gz`](evidence/nl7-stream.csv.gz) preserves the actual
82,944-line stream; [`evidence/nl7-score.csv.gz`](evidence/nl7-score.csv.gz)
preserves independent per-seed/split/arm/budget/mode counters, including all
robustness refusals and undetected errors. Raw Wolfram queries/results and both
native mutant logs accompany them. Early compiler/setup/oracle errors are
described rather than counted as successful evidence.

**Verdict:** recoverability can be guaranteed by an explicit finite code;
this experiment does not establish a successful learned language, useful
compression or learned-channel advantage. Johnson-Lindenstrauss projections
for real activations are future work only: **NOT_RUN**. Real models, GPU,
physical-network robustness, authority and deployment remain **NOT_RUN**.
These experiment-local files change no language/compiler/pin/host words,
repository ownership or harness gate, so no omega-harness guide change applies.

Independent numeric replay is additionally implemented by benchmarks-mv
`nl7-wire.om`: it consumes the same raw `P/N/D` stream, reconstructs
coordinates from transmitted bytes, reapplies perturbation, checks emitted
coordinate IEEE values and recomputes the learned receiver and abstention.
Observed result: `NL7WIRE PASS frames=21888 refused=3764`.
Its four qualification cases passed, including three more source mutants
for receiver weight indexing, carrier decoding and decision threshold.
Encode refusals require `pred99/status1`; stale refused carrier bytes do not
independently prove sender-overflow causality. The admitted-carrier replay,
native sender admission tests and refusal counters are separate evidence.
Together the three native conductors executed 39 cases and killed 26 source
mutants. `nl7-wire.stdout`, `nl7-wire-mutants.log` and the transport-equation
Wolfram receipt are retained with the run.
