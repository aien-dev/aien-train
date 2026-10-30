# Independent L720 reference review

Reviewed exact Git source `7e1d0d61755ec1a0bbf20941a18379861b5b9d0d` from
`aien-dev/omega-bridge`, branch `feat/renda8b-ref32-L720`. The remote reference
was independently read back. Snapshot archived with `git archive`, without
editing the author's worktree or restarting its stopped worker. Owner remains
L720. Source files carry SPDX `AGPL-3.0-only`; repository LICENSE is AGPLv3.

Compiler is the independently rebuilt baseline at
`/home/drakestapleton/workspace/omega-ecosystem660-bridge/build/bootstrap/omc`,
SHA256 `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.
Six fresh static AArch64 executables were built using `quiet-build`, after
concatenating the exact source module, existing report module and existing test.
Raw source units, compiler results, outputs and separate process statuses are
retained here; `SHA256SUMS` pins source, historical oracle corpus and executables.

Addition focused16/16 and corpus9505/9505 passed. Multiplication focused15/15
and corpus9505/9505 passed, all native exits0. Both published missing-behavior
controls compiled0 then exited1, each producing13 failed assertions. Existing
corpora use historical external Python expected values; no Python was invoked
or new foreign code introduced by this review. This is independent replay and
source review, not a new independently generated exhaustive arithmetic corpus.

Source inspection checked low32 masking, IEEE exceptional cases, signed zero,
exponent alignment, sticky-bit preservation, tie-to-even rounding, gradual
underflow, multiplication product width and overflow. No blocking defect found
within the declared add/mul bit-pattern reference contract. NaNs canonicalize;
exception flags and alternate rounding modes are not supplied. Multiplication's
wide product fits48 bits; shifts beyond63 cannot round its subnormal result up.
The addition magnitude ordering makes exponent differences nonnegative; exact
cancellation returns positive zero under nearest rounding.

These are real software IEEE binary32 primitives built using integer operations,
not fixed-point arithmetic and not foreign runtime calls. They can support a
bounded native affine forward/VJP/SGD consumer without compiler float syntax.
Division, square root, nonlinear functions, automatic differentiation, tensor
storage/runtime integration, GPU execution and model inference are not provided.
This review does not qualify the full ecosystem or replace L720 ownership.
