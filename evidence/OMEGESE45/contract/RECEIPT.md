# OE45 contract suite receipt

Status: EXERCISED on the GB10 Spark Grace CPU host (aarch64), cores 10-14, `nice -n 19 ionice -c3`. No GPU.

- Tree: fresh clone of `aien-dev/aien-train` branch `feat/45-omegese-contract` at `5a765c7973a93cb41bcdddeea0c45a0ebc14a775`
  (`git clone --reference /tmp/at45 -b feat/45-omegese-contract https://github.com/aien-dev/aien-train /tmp/oe45c-fresh`).
- Compiler: omega-mv `bf54b19fa590e8ae925c0e80815516dbf41e7326` `seed/out/omc`, sha256 `62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b`.
- Command (in the clone):
  `taskset -c 10-14 nice -n 19 ionice -c3 env OMC=/tmp/om-main/seed/out/omc OUT=/tmp/oe45c-s/run2 sh tests/omegese45/contract/run-contract.sh`
- rc 0; compile rc unit 0, m1 0, m2 0. Output ([`run-contract.log`](run-contract.log)):

```
unit PASS 4717 checks
m1 KILLED rej-R-bound (want rc=40 out=INVALID 12 grammar)
m2 KILLED gmm-000-ffff (want rc=0 out=VALID G)
OE45_CONTRACT_PASS
```

- Negative control: mutant m1 (R `refused` admits the C set, so `bound`) and m2 (digit `f` dropped from the shared digit
  transition) compile and each fail the suite at the check named above.
- Every check line of the good unit: [`unit.checks`](unit.checks) (4717 lines). Hashes of omc, units, vocab.tsv,
  receipt fixture, concatenated unit, driver ELF and mutants: [`artifacts.sha256`](artifacts.sha256).
- Not exercised: MJR1 `verdict` (272) and policy `zerohash` (240) refusals have no mutated-receipt case; signatures are not
  verified by `mjr1.om` (by design, like `decode_receipt`).
