# Native distributed metadata admission receipt

Status: bounded host component qualification. Whole aien-train #7 and Architecture
#303 remain OPEN. There is no network collective, floating-point reduction,
distributed trainer, GPU execution, consumer cutover or retirement claim.
The parent coordinator independently reviewed the exact source and accepted
only this serialized metadata-agreement scope. The original report, fresh
265-check replay, all five recompiled mutants and an independent sequence-model
probe are retained in `independent/`. The probe enumerates all 81 four-message
sequences for three ranks, plus deadline/backwards-clock/cancellation controls.
Its initial invalid offset fixture is retained alongside the corrected passing
fixture; no component source change was required.

## Exact inputs

- aien-train base: `ed8b6ea2562efdda7913234ca8f34b443714a987`.
- Functional source/tests/contracts: `ab17ed3234fda9b6b0c4d2bcdcee08e3d501caae`.
- Native component: `native/distributed7/admission.om`, SHA256
  `d92cbbd4480260124d6d60fdba7351b286de3d1074040d7fd7b8de9cb2af2dff`.
- Native tests: `tests/distributed7/admission.om`, SHA256
  `5f7ee38a7bb869b9ae72d78fdd73d779aa21aa59dffb2ca7b25b44667ed4f188`.
- Native source mutator: `tests/distributed7/mutate.om`, SHA256
  `2cbd913b122032da1595509319761a131230d00297f8c8cff8d9a2b6cb9006fc`.
- Consumed tensor implementation: Omega
  `3a995ba8dace538c1532c747d829ca6e4a73582d: native/ecosystem663/metadata.om`,
  SHA256 `f60b6d62c1bf90de42455effae977de66adeca1268669f6144809a3327aec7a4`.
- Primary compiler source-only bootstrap: omega-bridge
  `0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9`, importing Omega seed source
  `29b12760f4c962606411d6bf25032aedef063301`. Rebuilt compiler SHA256
  `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.
- Additional native-host compiler: Omega
  `f0499f415571bdd7496b6de391096915e78f7556`, rebuilt compiler SHA256
  `e5d60db892c8f21cc8c9128af26d8bded0e96d7f92946f5a549397c43a872649`.
- Both final test binaries SHA256
  `5f92ff3c918e9e698bb19c81e29938ad3076568587e4bcf7fc122cb0b6d94390`.
- Model/checkpoint/tokenizer/configuration: NOT_APPLICABLE for this metadata-only
  component. No synthetic fixture is represented as a trained model or gradient.

The seed/compiler lane independently reconstructed the compilers. This worker
verified the hashes, rebuilt this consumer from committed `.om` inputs and ran
the consumer. Identical binaries establish reproducibility of these builds, not
semantic correctness or a diverse compiler trust proof.

## Actual commands and outputs

Run on AArch64 Linux in an isolated directory. `OMEGA_TREE` and `TRAIN_TREE`
below point to source checkouts containing the exact commits. `COMPILER` must
name the already source-rebuilt compiler with one of the hashes above. These
commands use existing host utilities to extract/concatenate source, start the
compiler and inspect its output. This development orchestration has not been
retired into an Omega package or build tool.

```sh
git -C "$OMEGA_TREE" show 3a995ba8dace538c1532c747d829ca6e4a73582d:native/ecosystem663/metadata.om > metadata.om
git -C "$TRAIN_TREE" show ab17ed3234fda9b6b0c4d2bcdcee08e3d501caae:native/distributed7/admission.om > admission.om
git -C "$TRAIN_TREE" show ab17ed3234fda9b6b0c4d2bcdcee08e3d501caae:tests/distributed7/admission.om > tests.om
cat metadata.om admission.om tests.om > test.source.om
quiet-build "$COMPILER" < test.source.om > test.bin
chmod +x test.bin
quiet-build ./test.bin
```

`final/compile.exit` is 0, `final/test.exit` is 0. `final/test.stdout` contains
265 checks and 0 failed checks. The native-host compiler build and execution
also return 0, with identical bytes and 265/0 output. Raw source translation
units and all stdout/stderr/exit files are retained in `final/`.

The test-only native mutator is compiled from the exact committed source. It
requires one ASCII mode byte followed by the exact original translation unit;
it refuses an unexpected number of target occurrences instead of silently
claiming a mutation was made.

```sh
git -C "$TRAIN_TREE" show ab17ed3234fda9b6b0c4d2bcdcee08e3d501caae:tests/distributed7/mutate.om > mutate.om
quiet-build "$COMPILER" < mutate.om > mutate.bin
chmod +x mutate.bin
{ printf 0; cat test.source.om; } | ./mutate.bin > m0.source.om
quiet-build "$COMPILER" < m0.source.om > m0.bin
chmod +x m0.bin
quiet-build ./m0.bin
```

Repeat the explicit mode with 1, 2, 3 and 4. Actual final results:

| Mode | Defect introduced into compiled source | Prepare | Compile | Run | Failed assertions |
|---|---|---|---|---|---|
| 0 | Peer identity binding replaced by an unrelated nonzero tensor-ID check | 0 | 0 | 1 | 66 |
| 1 | A partial group is marked agreed | 0 | 0 | 1 | 49 |
| 2 | Nonzero stale/out-of-order sequences are admitted | 0 | 0 | 1 | 5 |
| 3 | Cross-rank descriptor fields compare to themselves | 0 | 0 | 1 | 9 |
| 4 | Deadline equality is incorrectly accepted | 0 | 0 | 1 | 6 |

`mutants/` preserves the earlier 255-check run, all compiled controls and raw
failures. The final suite adds frozen-roster mutation and unsupported descriptor
checks. Unknown mutator mode, empty source and already-mutated source return 65
and emit no stdout (`mutants/{unknown,empty,stale}.*`).

The first two compilation attempts are retained, not counted as caught behavioral
mutants: `build-first` error 13 for identifiers exceeding the existing eight-byte
limit; `build-second` error 12 for the compiler's fixed register budget. Short
names and bounded helper functions compile on the unchanged compiler.
`build-third` is the first passing 255-check suite. No compiler source was edited.

## Dependency and measurement observations

`readelf -d` reports no dynamic section for the final native executable.
The actual process/file/network trace contains only execution of that test ELF
and its successful exit; no child runtime, file loading or network operation was
observed. `strace` is an external observation tool, not linked into the test.
The filtered trace does not claim to audit unrelated workflows or every syscall.

One `/usr/bin/time` observation reports 192 KiB maximum RSS and elapsed/user/system
time rounded to 0.00 seconds. These durations are below the tool's displayed
resolution. They are not zero execution cost or a performance advantage. No
isolated performance window, throughput, communication cost, model quality or
energy measurement was taken. Raw output remains `final/host-cost.txt`.

Required runtime for these executables: Linux ARM64 syscall ABI. Development
tools used: Git, existing shell source loader/host commands, `cat`, `chmod`,
`sha256sum`, Spark's existing `quiet-build`, GNU readelf, strace and GNU time.
All newly authored component logic, assertions and source mutation logic are
handwritten `.om`. There is no C/Rust/Python/third-party numerical engine in
these test executables. No claim is made that ecosystem-wide foreign code or
existing development orchestration has been retired.

Physical discovery is separately recorded in `discovery/` and
`docs/distributed7/DEVICE_DISCOVERY.md`. Apple M2 is reachable; the seeded model
path is BLOCKED_IMPLEMENTATION. iMac discovery is BLOCKED_HARDWARE because the
configured endpoint is unreachable. Neither host metadata tests nor virtual
AMD/AGX results establish a physical collective or two-vendor model result.

Compiled binaries are retained locally under
`~/handoffs/ecosystem303/distributed7/binaries/`; the repository records their
original command paths and hashes in `BINARY-SHA256SUMS` and sufficient immutable
source to rebuild them. They are not shipped as trusted prebuilt executables.
The source/docs whitespace check passes. The whole-evidence whitespace check
reports the original readelf trailing spaces and SSH carriage return; those raw
observations remain byte-for-byte intact, with the check's failure retained.
