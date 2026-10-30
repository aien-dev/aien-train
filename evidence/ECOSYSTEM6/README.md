# Native affine FP32 host evidence

Functional source freeze: `d79b34352f769687b62cc77ba2e3739445723010`.
Base train main: `ed8b6ea2562efdda7913234ca8f34b443714a987`.
Consumed dependency source/binary pins and formulas are in
`docs/ecosystem6/dependencies.json` and `CONTRACT.md`.

The final compiled actual native consumer passed374708 assertions, including
4096 independently evaluated exact rational cases. The initial hand-calculated
example has pre-update loss1167/32. Three full altered source units compiled
successfully and failed the unchanged tests: reversed gradient at31, omitted
shape guard at93, reversed weight update at33. All three child exits are1.
The production compiler and child exits are0. Native fault injection runs also
exit0 and require exactly one source match. All final outputs are real static
AArch64 ELF; `frozen/readelf.txt` has no interpreter or dynamic section.

`frozen/unit.om` is the exact concatenated qualification unit, including copies
of the external pinned source solely to preserve this immutable test artifact.
They are not a new vendored product library. `mutant*-unit.om` contains the
exact compiled faulty units. Source snapshots include the original external
SPDX declarations. The standalone original libraries remain externally owned.

`artifacts.sha256` records source and build executable hashes. Generated ELF
files are kept outside Git in the private local build packet; fresh compilation
recreates them. `raw.sha256` seals all raw compiler/runtime results, snapshots
and original refusals. Exact build/run commands are in `commands.json`.

Original compiler refusals are preserved, not counted as behavioral controls:
attempt1 used an unsupported break keyword (9); attempt2 named a function with
a reserved type word (16); attempt3 exceeded the8-character global text name
limit (13); attempts4/5 exhausted temporary registers in the mutation scanner
(12). Extracting a small scanner helper corrected that capacity failure. No
compiler source, gate, numerical tolerance or expected answer was changed.

The one retained final timed complete test run took0.02seconds wall and user
time at0.01second measurement resolution, with192KiB reported maximum RSS.
This measures the combined test and oracle, not kernel-only throughput, training
quality, convergence, or a comparison against a foreign framework. Host is
recorded separately in `host.txt` and `host-cpu.txt`. No GPU or simulator ran.
Build/edit/review energy and total lifecycle time are unmeasured. Intermediate
positive runs were not individually metered or retained; they are not used as
qualification evidence. Eleven author compiler invocations include5 refusals,
and8 author native runs include two positive suite runs, three fault-injector
runs and three rejected faulty builds. Independent review costs are separate.

All aien-train#6 and Omega#663/#664 full acceptance criteria remain open. This
packet does not claim general operators, full autograd, checkpointing, resume,
data pipeline, model training, evaluation/export/serve, hardware execution,
consumer retirement or deletion. Existing L720/L491/L408 owners are preserved.
Independent final-source review is required before the publication desk accepts
the bounded proposed change.

## Independent review addendum

The separate portability/distributed worker completed review of exact functional
source d79b343 with no actionable findings in this bounded scope. Its fresh
rebuild passes374708 and its independently authored caller passes1041 checks,
including literal symbolic results, all14 central finite differences, repeated
zero-learning-rate execution and refusal atomicity. All three compiled faulty
units fail both suites. `independent-review/report.md` and the unmodified raw
packet retain exact inputs, commands, symbolic-oracle responses and hashes.
The original reviewer manifest is preserved verbatim; ELF hashes refer to local
artifacts omitted from Git. Parent review and sole publication-desk acceptance
remain separate. Wider training and ecosystem issues remain open.
