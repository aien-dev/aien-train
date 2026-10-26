# Audit scope

Written for an outside, independent security audit. Revision described:
`b6c33e3a82611361b37c651df70d9a645045773a` on `main`, 2026-10-26. Everything below is taken from the repository's own files at
that revision; items marked NOT_RUN were not executed while writing this document.

## What this repository is

A research plan for training a language model on AIEN's own engine (Omega, no CUDA, no third-party
training framework), starting on one DGX Spark (GB10). AGENTS.md: today the repository holds only
`README.md`, the plan in `docs/PLAN.md`, and empty placeholder directories (`evidence/`,
`streams/*/`). There is no code, build, CI, or written history, lessons, physics or receipts yet.

## What an auditor can and cannot check here

- Can check: that no tracked file carries tokens, credentials or private data; that the documents
  do not present any number as measured.
- Cannot check here: any training code, GPU kernel or measurement. None exists at this revision.

## What to review, in priority order

1. All tracked files for secrets or private data (nine files at this revision).
2. `docs/PLAN.md` and `AGENTS.md`: the rules that will govern later code (no CUDA or third-party
   training framework in the product path; every number names its chip; GPU work only in an
   authorised hardware window).

There is no `.github/workflows/` directory.

## Trust boundaries

- Untrusted: pull requests from outside contributors.
- Trusted: the maintainer's review. `main` receives merges and tags only (AGENTS.md).
- Planned GPU work needs an authorised window on the hardware (AGENTS.md).

## Threat model summary

- Assets: the honesty of future measurements; absence of secrets in a public repository.
- Attackers considered: accidental commit of private data; unreceipted numbers.
- Defended: written rules in AGENTS.md; pull-request review.
- Not defended / explicitly out of scope: there is no code or automated check to defend.

## Build from source at a pin

Nothing to build (AGENTS.md). Future measurements need a DGX Spark (GB10) and run on Omega; that
work is not part of this repository yet.

## Run every test

There is no test suite and no CI. NOT_RUN: nothing to run.

## Known limitations and NOT_RUN

- `docs/history/`, `docs/LESSONS.md`, `docs/physics/`: planned, not written.
- `evidence/` and `streams/*/`: empty placeholders.

## Dependencies and provenance

No lockfile, no vendored code, no dependencies, no GitHub Actions. PyTorch, JAX and similar are
yardsticks and oracles only, never shipped (AGENTS.md).

## Evidence

Receipts for every measurement belong in `evidence/` (empty today). Receipts are append-only and
are not edited after the fact.
