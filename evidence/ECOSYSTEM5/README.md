# PACK5 author receipt

Functional source: `088258d0c76c4e3b6c73ad92c12e20948014cc6f`.
Base: aien-train `ed8b6ea2562efdda7913234ca8f34b443714a987`.
Owner L787 under coordinator L786. Independent final-source review passed:
the seed qualification worker's separate caller passes84378 checks and detects
all three compiled mutants. Their fresh original suite passes282622 and produces
the identical ELF. The report and unchanged native reviewer source/raw records
are in independent-review/. Its original SHA256SUMS manifest is
0bfe741a1abc01d07dd74813cf9d2bb94023249e9d34443e64fb9e716f51e447;
the copied manifest verifies successfully. No ELF binaries were copied.
This review approves only the declared bounded packing contract, not broader
data/training/release gates. The existing desk retains publication ownership.
No hardware, live service, model, private sample, driver or worker restart.

Native actual-consumer suite:282622checksPASS, compile0/run0. Tests compose
existing L775/L776 source and execute cbegin/csubmit/cfinish before packing.
All three native-generated mutated modules compile0 and run1: inverted target
mask fails5163; missing record-attention boundary fails19; split refusal bypass
fails263254. Each mutation matches exactly one source site. A compile failure
is never counted as a detected mutant. All raw exits and output are retained.

Exact dependency identities and input semantics are in
`docs/ecosystem5/CONTRACT.md`; source/tool/ELF SHA256 values are in pins.sha256.
Historical raw source snapshots preserve the failed build translation units.
These snapshots are immutable evidence, not alternative production modules.
No executable binary is committed. The native ELF has no interpreter or dynamic
section; readelf.txt records the actual result. Linux remains the explicit host.

The initial invocation incorrectly passed filenames to a stdin/stdout compiler
and returned16 at line1. The first correctly invoked build rejected reserved
identifier len with16. Subsequent builds rejected an unnecessary bool cast9,
excess temporary register demand12, unsupported mod operation3, bool-to-integer
cast9, and an overlarge oracle function12. All were corrected in owned code;
the compiler and owner modules were not changed. Intermediate build8 passed
274903checks before final additional boundary tests raised the count282622.
The intermediate run was observed in the execution transcript; the final and
cost reruns have their own captured stdout and exits. No failed ELF was run.

One shared-host timing of the complete final test binary reports0.00seconds
at the coarse displayed resolution and192KiB maximum resident set. This is a
whole-suite wrapper measurement, not a zero-cost or per-token throughput claim.
There is no matched baseline, performance win, training quality result or GPU
measurement. Native data I/O, tokenizer parity and full restart remain untested.

## Reproduction commands

The following are disclosed shell development commands, not delivered product
automation. All new implementation, test assertions and defect generation use
handwritten .om. Use isolated paths and fail on any nonzero compile result.

```sh
git -C /path/to/omega show 84d03c5523f39ca1045d1c313c5f62597779462b:src/learning/benchmark/core.om > core.om
git -C /path/to/omega show d8d14c47647e49774f64b58e3c138335656f175d:src/learning/curriculum/admit.om > admit.om
cat core.om admit.om native/ecosystem5/pack.om tests/ecosystem5/pack-test.om > combined.om
/path/to/pinned/omc < combined.om > pack-test 2> compile.log
chmod +x pack-test
./pack-test
/path/to/pinned/omc < tests/ecosystem5/mutate.om > mutate 2> mutator-compile.log
chmod +x mutate
```

The actual compiler used was
`/home/drakestapleton/workspace/omega-ecosystem660-bridge/build/bootstrap/omc`.
Build files remain under
`/home/drakestapleton/handoffs/ecosystem303/data5/build/`.
For each mode ASCII1,2,3, prepend that byte to exact pack.om and feed to mutate.
Compose emitted module in place of pack.om with the same dependencies/test;
compile and require the asserted run1 failure. Keep generation/compile/run
results separate. This tests genuine compiled wrong behavior.

Scope is integer packing and mask/attention isolation for supplied public IDs.
No tokenizer, complete dataset loader, scheduler, durable cursor, transactional
checkpoint or model trainer has been replaced. No dependency or first-party
source is retired. Issues3/5 and the ecosystem parent remain open.
