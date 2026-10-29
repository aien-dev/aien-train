# OE45 contract suite receipt

Status: EXERCISED on the GB10 Spark Grace CPU host (aarch64), cores 10-14, `nice -n 19 ionice -c3`. No GPU.

- Tree: fresh clone of `aien-dev/aien-train` branch `feat/45-omegese-contract` at `076efa37281063fb09cbf1e69c227488e142bc36`
  (`git clone --reference /tmp/oe45-contract -b feat/45-omegese-contract https://github.com/aien-dev/aien-train /tmp/oe45cc-fresh`).
- Compiler: omega-mv `bf54b19fa590e8ae925c0e80815516dbf41e7326` `seed/out/omc`, sha256 `62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b`.
- Command (in the clone):
  `taskset -c 10-14 nice -n 19 ionice -c3 env OMC=/tmp/om-main/seed/out/omc OUT=/tmp/oe45cc-fs sh tests/omegese45/contract/run-contract.sh`
- rc 0; compile rc 0 for the unit and all 21 mutants; wall 3m3.6s. Output: [`run-contract.log`](run-contract.log), ending

```
unit PASS 4722 checks
mutants 21 KILLED 0 survived
OE45_CONTRACT_PASS
```

- Good unit: every check line in [`unit.checks`](unit.checks) (4722 lines). New relative to `59ab7b9`: `mjr1-zero-policy`
  (policy_hash 240..272 zeroed → `MJR1_INVALID 240 zerohash`), `mjr1-zero-verdict` (verifier_verdict 272..304 zeroed →
  `MJR1_INVALID 272 verdict`), `mjr1-refused-zero-verdict` (outcome 2, zero output, zero verdict → accepted),
  `mjr1-attempt2-gen` (attempt 2, generation bytes `01 02 03 04` → 16909060, Wolfram `FromDigits[{1,2,3,4},256]`),
  `mjr1-signer-03` (signer prefix 0x03 → accepted). All are byte edits of the real receipt fixture.
- Negative control: 21 one-line mutants (2 DFA, 6 binding, 5 codec, 8 MJR1). Each must fail first at its named check, or the
  suite fails; table in [`mutants.verdict`](mutants.verdict) and `docs/omegese45/CONTRACT.md` §10.
- The same command in the lane worktree (`/tmp/oe45-contract` at the same commit, OUT `/tmp/oe45cc-w1`) gave the same result:
  `unit.checks` byte-identical (`cmp`), all 51 lines of `artifacts.sha256` equal after removing path prefixes.
- Hashes of omc, units, vocab.tsv, receipt fixture, concatenated unit, driver ELF and every mutant source and ELF:
  [`artifacts.sha256`](artifacts.sha256).
- Not exercised: signatures are not verified by `mjr1.om` (by design, like `decode_receipt`; aienet0-run verifies them).
