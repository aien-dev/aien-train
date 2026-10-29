# evidence/ADTRAIN

Host runs on the DGX Spark (aarch64; `host.txt`), with aien-train code
`17be554dbc12f59522564ca17a4f1df0bae18804`, omega-mv
`965fb7627364274878c35a49e132fae492ff53be` and the pinned `seed/out/omc`
`62775d2a…3712b`. Commands: `commands.json`. Contract: `docs/adtrain/CONTRACT.md`.

| gate | status | evidence |
|---|---|---|
| 1 forward + loss | EXERCISED (host) | `run/T.stdout` 201 `L` rows |
| 2 gradients vs independent reference | EXERCISED (host) | `run/G.stdout` 58 checked, 0 failed; `run/fixture-host.stdout` C M21 tape 75/75 exact |
| 2 exact-rational oracle | NOT_RUN | `oracle.json` (Wolfram tools not mounted) |
| 3 updates reduce loss | EXERCISED (host) | `run/loss-gate.txt` L0=1090364420 > L100=1066232903 > L200=1041634414 (bits) |
| 4 checkpoint/resume identical | EXERCISED (host) | `run/T.tail` = `run/R.tail` (`cmp`); refusals in `run/R-*.rc` 66/65/65/65 |
| 5 CPU/GPU agreement, host arm | EXERCISED (host) | `run/fixture-host.stdout` `exact=75/75 max_ulp=0 verdict=PASS`; mutant-sign control FAIL |
| 5 CPU/GPU agreement, GB10 arm | NOT_RUN | `gb10/NOT_RUN.md` (no GPU window) |
| native tape suite (omega-mv) | EXERCISED (host) | `tape/tape-test.stdout` 302 checks; `tape/mutant*.rc` all 1 |
| mlp mutants | EXERCISED (host) | `run/mutant0-T.rc` 86, `run/mutant1-R-binding.rc` 0, `run/mutant2-R-flip96.rc` 0 |

Nothing here is QUALIFIED; host results are not hardware qualification.
`frozen/` holds the fixture and the step-120 checkpoint image of this run.
`attempts.md` records the compiler refusals met during development.
`raw.sha256` covers every other file in this directory.
