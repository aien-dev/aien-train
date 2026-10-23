# Independent bounded affine numerical review

Reviewer: Architecture #303 portability/distributed worker, separate from the
affine author. Reviewed frozen aien-train source commit
`d79b34352f769687b62cc77ba2e3739445723010` in a new detached worktree at
`/home/drakestapleton/workspace/aien-train-ecosystem6-review`.
No implementation, compiler, dependency or shared runtime changes were made.

Result: PASS within the declared fixed host numerical contract. No actionable
findings in this scope. This is an independent review of one bounded affine
forward/VJP/SGD consumer. It does not approve a complete trainer, general autograd,
GPU workload, real distributed numerical execution, or production cutover.
Architecture #303 and the implementation milestones remain open.

## Exact inputs

- Consumer `native/ecosystem6/affine.om`: SHA256
  `a3eb5b4a0a94adab8e0a2cfbe154280034d1da26bd9f3123fcd524636583af97`.
- Author test: `3c3ca874491bb5f7008b5cd6a324b6550fb9b01448c265a8837b71d56a7846ec`.
- Author mutator: `835b4179df8beb0feb8a4fc161126a76438dbf2f80ac8da199b596f3db18f9f5`.
- External add32/mul32 from omega-bridge commit
  `7e1d0d61755ec1a0bbf20941a18379861b5b9d0d`, SHA256 respectively
  `1de23561202b284e74690034cb07741ebcf9870bc464550b8382c85410478e95` and
  `9027090109e2050046c4b71fa51438c97ad94c23626c2ecc277844996b79c22e`.
- External metadata from omega-mv commit
  `3a995ba8dace538c1532c747d829ca6e4a73582d`, SHA256
  `f60b6d62c1bf90de42455effae977de66adeca1268669f6144809a3327aec7a4`.
- Compiler source omega-bridge `0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9`,
  binary SHA256 `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.
- Handwritten reviewer caller `independent.om`: SHA256
  `fc9318d74ac0af8c34e245601b25cf2b8464dc8bac23aafe210ee0295898e9aa`.

## Source review

Read the complete frozen implementation, author tests and mutation source,
dependency manifest, contract, and both external arithmetic sources. The code
uses the existing metadata validator and original parameter values for the
derivatives. It performs the stated sequential binary32 operations, then applies
the learning rate. The result checks precede all writes to stored tensor values
and loss. The public contract accurately describes fixed ulong carriers, trusted
allocator snapshots, shared scratch, serialization, fixed shape, and finite-input
limits. It explicitly separates the manual derivative from a general tape.

## Executed evidence

Freshly materialized dependency files were hash checked and concatenated with
the frozen consumer and author caller. Compilation and execution both exited0:
`NATIVE_AFFINE_FP32_PASS checks=374708`. The ELF SHA256 is
`de663132f87336a73b9766ea79a5e163e1412a4eb85b199d11aae53be1927973`.
Matching the author's binary is consistency evidence; the independent numerical
checks below supply separate correctness evidence for the tested cases.

The separate handwritten reviewer caller imported no author test helper. A
symbolic algebra kernel independently differentiated the sum of squared errors
and evaluated exact rational answers for X=[[1,-2,3],[2,0,-1]],
W=[[1/2,-1/4],[1,3/4],[-1/2,2]], b=[1/8,-1/2],
T=[[1/4,1],[-1,1/2]], lr=1/32. Its loss is1167/32. All forward
values,14 derivatives and8 updated parameters were checked as literal binary32
words. This test does not rely on lower loss as evidence of correct derivatives.

For each of the14 X/W/b entries, the reviewer executed the actual consumer at
plus and minus1/8 and checked both losses against separate exact symbolic
answers. The symbolic result verifies that4*(Lplus-Lminus) equals the derivative
for every entry. These dyadic inputs and small polynomial results are exactly
representable for the selected fixture. This finite-difference test does not
assert a numerical tolerance or accuracy guarantee for arbitrary ill-conditioned
or rounded cases. The saved request and raw result show the independent oracle.

The reviewer also checked repeated zero-learning-rate execution, preserved X/T
and unused padding, stale/zero generation, a valid-but-incompatible shape,
negative-zero learning rate, NaN input, overflow, and complete AP/AL refusal
atomicity. The final independent caller compiled0 and ran0:
`INDEPENDENT_AFFINE_PASS checks=1041`.

The author mutator was freshly compiled and used to materialize each actual
faulty consumer from the exact frozen source. All mutations generated0 and
compiled0. Both the unchanged author suite and independent caller rejected them:

| Compiled change | Author failure check / exit | Reviewer failure check / exit |
| --- | --- | --- |
| Negate derivative multiplier |31 /1 |7 /1 |
| Remove shape/stride admission |93 /1 |262 /1 |
| Add weight update instead of subtracting |33 /1 |21 /1 |

Raw statuses, output, rebuilt translation units and altered modules are retained.
No compiler refusal is counted as a successful behavioral mutant. No execution
failure occurred in the reviewer positive tests. The preliminary911-check caller
passed before the extra shape refusal was added; the final1041-check source and
raw result are the sealed review artifact.

`readelf` shows a static AArch64 ELF with no dynamic section. An external
file/process/network trace of the independent caller observed only its initial
exec and clean exit; no external file, network or child-process runtime use was
observed. This is a scoped trace, not proof of all possible program paths.

## Reproduction

`commands.txt` records the actual source materialization and build recipe.
`normal-unit.om` and `independent-unit.om` retain the exact combined inputs.
Compile either with the pinned compiler, verify ELF, then execute on AArch64
Linux. Mutant units have explicit names for author and independent callers.
ELF hashes are retained in `raw/elf-sha256.txt`; ELF binaries are local review
artifacts and should not be committed. All new executable reviewer test code is
handwritten `.om`. Shell concatenation is development orchestration only. The
symbolic kernel is an external qualification oracle and is not in the product.

There is no simulator or physical-device result in this packet. It provides no
throughput, model quality, convergence or ecosystem coverage claim. The bounded
test set does not prove IEEE behavior for all possible inputs. Existing owners
retain wider tensor, autograd, training, transport and compiler implementation.
