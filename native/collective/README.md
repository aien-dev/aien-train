# SI sparse collective: native .om algebraic pilot

**Status:** Review branch. Seeded omc rejected `bestscore` (error 13: names at most 8 characters). This commit renames that local to `bscore`. Host sandbox execution of the pinned compiler is filed with the PR comment request; it is not repository qualification. Live model training and DGX Spark qualification remain **NOT_RUN**.

**Issue:** [aien-train #26](https://github.com/aien-dev/aien-train/issues/26) under [SI Fusion architecture #306](https://github.com/aien-dev/aien-architecture-mv/issues/306) and [SI Genesis #307](https://github.com/aien-dev/aien-architecture-mv/issues/307).

This is deliberately a small first-party handwritten seeded Omega experiment. The branch does **not** introduce a neural trainer, a new runtime, a model router, another Cortex store, a GPU scheduler or a public endpoint. It does not import, modify or download any model checkpoint. Existing [Omega #694](https://github.com/aien-dev/omega-mv/issues/694) owns native 0.8B/2B inference; [Omega #695](https://github.com/aien-dev/omega-mv/issues/695) owns physical batching; [Core #737](https://github.com/aien-dev/aien-sovereign-core-mv/issues/737) owns Spark admission.

## What the small program actually does

[`shared_bank_p0.om`](shared_bank_p0.om) creates a synthetic common integer matrix B of shape 2x3 and a 3-vector x. The 8 specialist candidates are *rank-one* differences U_i V_i^T where U_i has 2 entries and V_i has 3. Their response is:

    y(i, sign) = B*x + sign*U_i*(V_i^T*x) ; sign in {+1,-1}

- It computes the common B*x **once** into a two-element scratch array.
- It calls the same rank-one adapter evaluator for eight candidate indices in positive and negative phases, checking all **16 outputs**.
- It checks two selected candidate IDs by a deterministic score and allowed mask: [1,3]. It is a toy proposal policy, **not** AEGIS authority or a trained router.
- Denying ID 1 makes the top choice ID 3. Refusing all 8 returns sentinel ID 8 rather than executing an invented model.
- It refuses out-of-range candidate ID, denied candidate and unsupported phase. A nonzero program exit indicates a failed self-test.
- It separately asserts the shared base was computed exactly once, and 16 admitted adapter calls occurred.

Only exact bounded signed integer arithmetic is used. This fixture does not check numerical FP8/GDN/attention, train a LoRA, allocate GPUs or implement multi-tenant security.

## Independent mathematical reference

[`fixtures/golden_v0.json`](fixtures/golden_v0.json) is a frozen independent Wolfram integer-matrix oracle, not calculated by the proposed native .om kernel. The test's expected common response is [16,-8].

| Candidate i | positive output | negative output |
| --- | --- | --- |
| 0 | [-2,-2] | [34,-14] |
| 1 | [10,-2] | [22,-14] |
| 2 | [16,-8] | [16,-8] |
| 3 | [16,-20] | [16,4] |
| 4 | [10,-38] | [22,22] |
| 5 | [-2,-62] | [34,46] |
| 6 | [-20,-92] | [52,76] |
| 7 | [-44,-128] | [76,112] |

Reproduce reference values with an independently obtained Wolfram Language kernel:

```wolfram
x={2,3,5};
b={{2,-1,3},{1,0,-2}};
out[i_,sgn_]:=(b+sgn Outer[Times,{i-3,i+1},{1,3-i,-1}]).x;
Table[{i,out[i,1],out[i,-1]},{i,0,7}]
```

## Native execution instructions (not run by PR author)

In a **claimed isolated AArch64 worktree** with the current owner-approved Omega seed compiler, and a separate checkout of `aien-train` at the PR head:

```sh
cd /path/to/pinned/omega-mv
sh seed/omega-from-seed.sh
mkdir -p build/si-sparse
seed/out/omc < /path/to/aien-train/native/collective/shared_bank_p0.om > build/si-sparse/shared-bank
chmod +x build/si-sparse/shared-bank
timeout 10 build/si-sparse/shared-bank
printf 'native_exit=%s\n' "$?"
sha256sum seed/out/omc build/si-sparse/shared-bank
```

Expected executable exit: 0. Compile and execution are **NOT_RUN**, so the compiler may reject particular grammar details or encounter a machine limitation. A failure must be recorded and fixed by the existing seeded compiler/profile owner or in this branch's source; do not replace it with Python/C/PyTorch product execution. On an x86-64 system without a qualified AArch64 emulator, a native execution is NOT_RUN, not PASS or FAIL.

Do not use this AArch64 smoke to infer an sm_121 GPU kernel already exists. Preserve raw commands, compiler/source hashes, program output and independent checker review when executed.

## Engineering hypothesis, and an intentionally weak example

For **eight full 2x3 expert matrices** the raw dense baseline has 8*6 = 48 scalar weight entries. The shared representation stores 6 common + 8*(2+3) = 46 scalar entries **before** counting routing scores, masks, dictionary metadata, provenance, padding, tensor formats or code. This tiny example is likely *not profitable* in total serialized bytes or runtime work; it exists to check correctness and illustrate reuse.

Only with much larger matching matrix shapes and low-rank learned adapter deltas can a shared representation plausibly reduce external memory bytes. Rank-r factorization is evaluated over its full metadata and low-rank projection operations; quantized formats must be compared at identical quality. The real 35B and 27B have different architectures and cannot simply share parameter coordinates.

**No physical memory-bandwidth saving, token speedup or model quality improvement is claimed.**

## Next owner steps

1. On existing approved seed/host chain, run the exact .om positive/negative cases and independently mutate one factor sign or one expected oracle entry; the gate should fail. Record SHA, command, exit and whether the test ran on host/emulator.
2. Verify the actual current 0.8B staging/model-license/quant/feature profile through [Core #737](https://github.com/aien-dev/aien-sovereign-core-mv/issues/737) and native Omega model requirements [#694](https://github.com/aien-dev/omega-mv/issues/694); do not acquire weights or take a Spark lock without approval.
3. In native aien-train #3/#6 work, train eight real same-base specialist adapters using independent admitted data. Compare eight exact standalone small models as a control, and a truly from-scratch experience model via existing #19 as a distinct origin.
4. Use verified microjob ABI [protocols #50](https://github.com/aien-dev/aien-protocols-mv/issues/50), Allen [#13](https://github.com/aien-dev/allen/issues/13), Omega [#695](https://github.com/aien-dev/omega-mv/issues/695) and current FORGE device owner to route a bounded number of qualified adapters **without duplicating full base weights**.
5. Freeze independent full-task quality/safety/economics in [benchmarks #38](https://github.com/aien-dev/benchmarks-mv/issues/38); only scale the candidate count 8 to 32 to 128 if the completed objective and physical cost improve at matched quality.
6. Final product-owned code/tooling remains seeded .om, and [architecture #300/#316](https://github.com/aien-dev/aien-architecture-mv/issues/300) controls elimination of first-party legacy runtime dependencies after separate verification.

No code in this branch changes the production server or shared compiler. GitHub draft PR does not mean qualified execution.
