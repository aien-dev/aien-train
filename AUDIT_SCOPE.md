# Audit scope

Written for an outside, independent security audit. Revision described:
`3481e8c84cb95bd3ec0aa3c0171e294c685ed954` on `main`, 2026-10-30 (751 tracked files). Later merges
add units; check `git log` for anything newer. Items marked NOT_RUN were not executed while
writing this document.

## What this repository is

Native training work for AIEN's own engine (Omega, no CUDA, no third-party training
framework). The product code is handwritten seeded Omega (`.om`) compiled by the pinned `omc`
seed compiler; every check and verdict lives in the `.om` units. All results so far are host
CPU (aarch64) runs. No unit has run on a GPU or the DGX Spark (GB10); that work stays NOT_RUN and
needs an authorised hardware window (AGENTS.md).

## Inventory

Native `.om` units and their tests, each with a contract under `docs/<area>/` and receipts under
`evidence/<AREA>/`:

| Area | Units | Tests | Contract | Evidence |
| --- | --- | --- | --- | --- |
| distributed7 (admission over the shared tensor contract, #14) | `native/distributed7/` | `tests/distributed7/admission.om`, `mutate.om` | `docs/distributed7/CONTRACT.md` | `evidence/DISTRIBUTED7/` |
| ecosystem5 (PACK5 training-example packing, #16) | `native/ecosystem5/` | `tests/ecosystem5/pack-test.om`, `mutate.om` | `docs/ecosystem5/CONTRACT.md` | `evidence/ECOSYSTEM5/` |
| ecosystem6 (FP32 forward, backward and SGD step, #15) | `native/ecosystem6/` | `tests/ecosystem6/step-test.om`, `mutate.om` | `docs/ecosystem6/CONTRACT.md` | `evidence/ECOSYSTEM6/` |
| si19 (offline experiential world model, #22) | `native/si19/` | `tests/si19/world-test.om` | `docs/si19/CONTRACT.md` | `evidence/SI19/` |
| collective (shared-base specialist pilot, #27) | `native/collective/` | self-test inside `shared_bank_p0.om` | `native/collective/README.md` | host record in that README |
| neuralese NL1 (numerical channel experiment, #36) | `native/neuralese/*.om`, `mutants/` | `check.om`, `gradientcheck.om` | `native/neuralese/README.md` | `native/neuralese/evidence/` |

Non-`.om` tracked files that execute: `tests/si19/run.sh` (disclosed POSIX host glue that
materialises pinned sources, checks hashes, builds and runs; no verdict logic) and the Wolfram
Language oracles `native/neuralese/*.wl` (independent reference only, never in the product
path).

## Pinned external `.om` dependencies

`docs/ecosystem6/dependencies.json` pins the compiler (omega-bridge `0fc7d866`, binary sha256
`191b776e...98e9`, aarch64 static ELF) and three external sources by commit and sha256:
`native/ref32/add32.om` and `native/ref32/mul32.om` from omega-bridge `7e1d0d61`, and
`native/ecosystem663/metadata.om` from omega-mv `3a995ba8`. `docs/si19/dependencies.json` pins
the same three plus `native/ecosystem6/affine.om` from this repository at `c9782bfd`, with
hashes in `docs/si19/deps.sha256`.

## What an auditor can check here

- That no tracked file carries tokens, credentials or private data.
- That every pinned dependency materialises from its commit with the recorded sha256.
- That the `.om` tests pass and the mutants fail, by replaying the contracts (below).
- That no number is presented as measured without a receipt in `evidence/` or the unit's
  README, and that GPU/Spark results are labelled NOT_RUN.

Cannot be checked here: any GPU kernel, device measurement or real-model training. None exists.

## What to review, in priority order

1. Inputs the `.om` units read (images, records, fixtures) and their refusal paths: damaged or
   truncated images, out-of-range ids, denied candidates.
2. Dependency pinning: the commits and sha256 values above, and that runners refuse a mismatch.
3. `tests/si19/run.sh`: the only shell in the tree; it reads only pinned Git objects and the
   compiler path given in the environment.
4. All tracked files for secrets or private data.
5. `docs/PLAN.md` and `AGENTS.md`: the rules that govern the code (no CUDA or third-party
   training framework in the product path; every number names its chip; GPU work only in an
   authorised hardware window).

There is no `.github/workflows/` directory and no GitHub check runs on pull requests.

## Trust boundaries

- Untrusted: pull requests from outside contributors; any input file given to a `.om` driver.
- Trusted: the maintainer's review; the pinned compiler binary, identified by sha256.
  `main` receives merges and tags only (AGENTS.md).
- Planned GPU work needs an authorised window on the hardware (AGENTS.md).

## Threat model summary

- Assets: the honesty of measurements and receipts; absence of secrets in a public repository;
  integrity of pinned dependencies.
- Attackers considered: accidental commit of private data; unreceipted numbers; a swapped
  dependency or compiler; malformed input images.
- Defended: sha256 checks of every pinned source and of the compiler; refusal paths in the `.om`
  units; mutant runs that must fail; pull-request review.
- Not defended / out of scope: the Omega compiler itself (report to `aien-dev/omega-mv`), the host
  operating system, and the absence of CI.

## Build from source at a pin and run every test

Each contract states its replay recipe: `docs/distributed7/CONTRACT.md` (Consumed interfaces,
Qualification), `docs/ecosystem5/CONTRACT.md` (Frozen dependencies and caller),
`docs/ecosystem6/CONTRACT.md` (Dependency and build freeze), `docs/si19/CONTRACT.md` (Build and
run: `sh tests/si19/run.sh`), and the collective and neuralese READMEs. The recipes need the
pinned `omc` (sha256 in the dependency files), clones of omega-bridge and omega-mv containing the
pinned commits, and an aarch64 Linux host. NOT_RUN while writing this document: a full replay of
every area.

## Known limitations and NOT_RUN

- GPU, DGX Spark and real-model training: NOT_RUN for every area.
- `docs/history/`, `docs/LESSONS.md`, `docs/physics/`: planned, not written.
- `streams/*/`: empty placeholders.
- First-party shell glue `tests/si19/run.sh` remains; its OM2 conductor replacement is recorded
  in `docs/si19/CONTRACT.md`.

## Dependencies and provenance

No lockfile, no vendored third-party code, no package manager dependencies, no GitHub Actions.
External code is limited to the pinned `.om` sources listed above, all first-party AIEN
repositories. PyTorch, JAX and similar are yardsticks and oracles only, never shipped
(AGENTS.md); Wolfram Language files are oracles only.

## Evidence

Receipts live in `evidence/DISTRIBUTED7/`, `evidence/ECOSYSTEM5/`, `evidence/ECOSYSTEM6/`,
`evidence/SI19/` and `native/neuralese/evidence/`. Receipts are append-only and are not edited
after the fact.
