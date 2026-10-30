# Bounded native affine numerical consumer

This is an executable prerequisite for aien-train#6 and Omega#663/#664, not a
general trainer, automatic differentiation system, completed ecosystem M1 or
production consumer cutover. It reuses the existing L720 arithmetic and M20
metadata implementations. Their owners remain unchanged. L491 retains trainer
ownership. No compiler, runtime, model, GPU or service was changed.

`native/ecosystem6/affine.om` executes actual IEEE binary32 operations through
the external L720 `add32` and `mul32`. Inputs, intermediate results, gradients
and updated parameters are binary32 bit patterns held in ulong cells. These
carriers are not a new byte-addressed tensor allocator. Descriptor dtype1 means
the arithmetic value format; storage remains the declared fixed local carriers.
No foreign floating runtime, language wrapper or fixed-point product path is used.

The fixed graph has X[2,3], W[3,2], b[2], T[2,2]. It computes Y=XW+b and
L=sum((Y-T)^2), then dX, dW and db from the original W, followed by
W'=W-lr*dW and b'=b-lr*db. Each scalar multiply and addition rounds separately
to nearest, ties to even. This is not fused multiply-add. Reduction order is
the source's ascending row, column and inner-dimension order. The loss and
gradients describe the pre-update parameters. This explicit derivative is a
vector-Jacobian product for one fixed graph, not a general differentiation tape.

The eight slots X,W,b,T,Y,dX,dW,db occupy `AP[t*16+i]`. `AD[t*24+i]` contains
the M20 metadata-v1 descriptor and `AS[t*8+i]` its trusted allocator snapshot.
The consumer validates each with existing `emcheck`, then requires dtype1,
offset0, contiguous fixed dimensions, capacity exactly2/4/6, and distinct slot
identity t+1. The caller must supply trusted snapshots; these records do not
authenticate external memory or callers. Snapshot generations may vary so long
as the descriptor and live allocator snapshot agree. Shared `ED`/`ES` are scratch.
All operations are serialized. Padding cells are not part of a tensor.

`afstep(lr)` returns0 on success. Existing M20 failures retain their codes.
Additional failures are -20 incompatible local storage/dtype, -21 shape/stride,
-22 invalid input or learning rate, -23 nonfinite computed result. Inputs must
be finite canonical-width32 bit patterns. lr must be finite with a clear sign
bit; negative zero is deliberately refused. NaN and infinity inputs are refused.
Overflow and any nonfinite intermediate that reaches a result reject the step.
Subnormals and rounding underflow are supported. `AP` and `AL` remain unchanged
on every refusal; private scratch `AQ`, `ED` and `ES` may change. The final commit
replaces W,b,Y,dX,dW,db and writes the pre-update loss to `AL[0]`, preserving X,T.

## Dependency and build freeze

`dependencies.json` records exact repositories, source commits, files and hashes.
The libraries remain external source dependencies, not copied replacement
libraries. Build order is L720 add32.om, L720 mul32.om, M20 metadata.om, affine.om,
then the chosen caller. Concatenation is existing host build glue and is not
claimed retired. All new executable implementation, tests and mutation tooling
are handwritten `.om`. External dependencies have AGPLv3 licensing; preserve
their existing SPDX declarations and repository licenses when distributing a unit.

On AArch64 Linux, materialize the three dependency files from their exact Git
objects, verify the SHA256 values, then use this existing command recipe with
private paths:

```sh
cat "$ADD32" "$MUL32" "$METADATA" native/ecosystem6/affine.om tests/ecosystem6/step-test.om > "$OUT/unit.om"
quiet-build "$COMPILER" < "$OUT/unit.om" > "$OUT/unit.elf"
chmod +x "$OUT/unit.elf"
"$OUT/unit.elf"
```

Record compiler and program statuses separately and verify a real ELF before
execution. Expected output is `NATIVE_AFFINE_FP32_PASS checks=374708`, exit0.
The independent test oracle represents exactly expressible rational inputs as
integers solely to calculate expected answers: X=x/8,W=w/16,b=b/128,T=t/128,
lr=1/32. It computes integer polynomial derivatives and encodes exact expected
binary32 values without calling add32/mul32. 4096 deterministic cases have all
numerators within24 significant bits, so intermediate rounding is exact.
Additional literal fixtures check a hand-calculated loss36.46875, smallest
subnormal forward/gradient propagation, underflow to zero and midpoint ties.
The oracle's fixed rational representation is test-only, not the implementation.

`tests/ecosystem6/mutate.om` compiles using the same compiler. Its input is one
ASCII mode byte followed by affine.om. Modes0/1/2 invert the gradient sign,
remove dimension admission, or reverse the weight-update direction. It requires
exactly one source match and emits the full altered module. Concatenate the
same dependencies and unchanged test with the resulting module, compile and run.
All three compile0 and fail the test with exit1, at checks31/93/33 respectively.
The tests do not merely look for reduced loss or self-rebuild consistency.

## Qualification boundaries

This host-native test is the first actual downstream numerical consumer of
these pinned primitives and M20 metadata in this slice. It does not replace an
existing production trainer. Arbitrary layouts, dynamic allocation, matmul sizes,
other operators, optimizer state, data loading, checkpoint/resume, evaluation,
export/serve, distributed and GPU execution remain open in their existing issues.
There is no simulator or physical-device numerical result, speedup, quality or
training convergence claim. Existing seeded compiler and L720 host-oracle corpus
evidence remain separately qualified; finite tests are not a proof of all IEEE
inputs. Performance and lifecycle costs are recorded without comparing to a
framework or claiming a win. Initial compiler refusals are preserved in evidence.
