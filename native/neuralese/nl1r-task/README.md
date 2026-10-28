# NL1R task-conditioned byte communication

Follow-up [aien-train#39](https://github.com/aien-dev/aien-train/issues/39), experiment `NL1R-20261028-task-1`. The original experiment and its negative result remain unchanged. The new NL-6 preregistration is frozen at benchmarks `c8aefbc6cf72ea2ca47fe4308bd867cb1a55e86c`, manifest SHA-256 `2fe5c72edbfe24df38f70b72275bb008ac71779e66455501662ca8e7d911751c`.

## Context and actual channels

The sender observes all four evidence bits plus the shared task. The receiver sees the actual carrier, shared task and frozen decoder; only task2 also receives the first two original bits as environment-provided side information. Sample IDs are evaluator labels, not receiver input. Receiver functions do not read sender scratch inputs or training targets.

Tasks are full reconstruction, parity, and reconstruction with the first two bits already known. Fixed codes use the identity, parity, and upper-two-bit colorings respectively. Their confusion graphs are K16, K8,8, and four disjoint K4; proper-coloring and matching clique witnesses establish the exact logical budgets recorded in the parent Wolfram oracle. All three still serialize to one actual byte.

A task-specific 4-to-4-to-4 softsign model is trained using the qualified capacity module. Inactive residuals are exactly zero. Dense and factorized arms share the same trained checkpoint: these are **inference channel ablations**, not independently optimized discrete models. Dense uses four two-bit coordinate bins; factorized uses four packed binary categories. Neither uses a straight-through gradient. Both carry one byte. Full evidence carries four lowbit-first ASCII digits and is not an equal-payload comparator. Receiver-only carries no payload and uses a zero receiver vector.

Every arm emits correct, fixed-derangement mismatched, and omitted controls. Omission refuses even for receiver-only. Refused records contain zero scores/predictions and abstain; the decoder itself leaves receiver output/state unchanged on refusal. Task2 preserves the original receiver context on a mismatch. Abstention examines only active task outputs, never irrelevant parity outputs or known context bits.

## Qualification and execution

Use only the compiler and source hashes in `config.json`. Concatenate the listed `source_order` paths and compile with the pinned compiler on stdin, binary on stdout. A nonzero compiler result is not executable. Run the resulting program as `task TASK SEED TRAINMASK`. `controls_order` builds a separate native qualification executable without running random training.

Native qualification exercises every task's masked finite-difference gradient and staged update; all fixed/fulltext states and message variants; actual dense/factorized constructive bytes; sender-private scratch destruction before reception; every byte value for fixed/dense/factorized decoding; every byte value at every ASCII position; schema, width, parameter-count, effect, length, context, and omission refusals with unchanged-state sentinels. Independent Wolfram derivatives and updates pass. Deliberately removing the inactive-loss mask, schema/effect/width checks, payload consumption, or dense slot indexing is detected by the native controls. Raw records, exact oracle queries and failed compiler attempts are retained in evidence.

The constructive dense witness has scores 3/8 and 5/8, while factorized has 1/4 and 3/4. Executed Wolfram query: `ss[x_]:=x/(1+Abs[x]); h={-1/2,1/2}; q=Floor[3*(h+1)/2+1/2]; lev=(-3+2*q)/3; {q,lev,Map[(1+ss[#])/2&,lev],Map[(1+ss[#])/2&,{-1,1}],Min[Abs[Map[(1+ss[#])/2&,lev]-1/2]]` returns `{{1,2},{-1/3,1/3},{3/8,5/8},{1/4,3/4},1/8}`. This is capacity evidence, not a training result.

## Cost and authority boundaries

C/P/R/T records follow the frozen judge protocol. The 32 payload columns are logging capacity, not transmitted padding. T measures initialization plus training; row intervals measure encoding and decoding. Record source, binary, model, codebook, metadata, log, whole-process and independent verification costs separately. The fixed baselines have no optimizer; campaign gross costs include the shared harness and learned runs rather than pretending those costs disappeared. No inference retries or best-seed selection are permitted.

The small task carrier is not an execution authorization. The existing NL4 typed proposal/verification path separately consumes its full envelope and verifies identity chunks against independent trusted task references. Wrong accepted decisions must fail that semantic check. Production capability/reservation binding and authorized external execution remain NOT_RUN. No GPU, serving, downloads, deployment, or model capture is part of this experiment.

Random learning and independent hidden confirmation are evaluated only after this source/config freeze. Report every seed, every split, every failed arm, exact recovery, accepted-correct, wrong-accepted, abstention, held-out success and finite 0/1 decision regret separately. No finite diagnostic result establishes general compositionality or economic improvement.
