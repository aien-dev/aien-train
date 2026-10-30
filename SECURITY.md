# Security policy

## Reporting a vulnerability

Report privately. Do not open a public issue, pull request or discussion about it.

1. GitHub: open this repository's Security tab and choose "Report a vulnerability". This
   creates a private advisory that only the maintainers can see. If that option is not shown,
   private vulnerability reporting has not been enabled for this repository yet; use email.
2. Email: drake@aienos.com with the subject `[aien-train] security report`.

Include the commit or release you tested, the component (path), steps to reproduce, and what an
attacker gains. If the report involves a secret, say where it was exposed and do not send the
secret itself. Encrypted mail is not required; do not post proof-of-concept payloads publicly.

## Response targets

This project has one maintainer; the targets below are best effort and stated honestly.

- Acknowledgement: within 3 working days.
- Triage and severity: within 7 days of acknowledgement.
- Fix or mitigation for confirmed issues: aim for 30 days for high and critical, 90 days
  otherwise, with a status update at least every 14 days until closed.
- Disclosure: coordinated. An advisory is published with credit (unless you ask otherwise)
  once the fix is on the default branch. Default embargo 90 days, negotiable.

## Scope

The repository holds handwritten seeded Omega (`.om`) training units and their tests
(`native/` and `tests/` for distributed7, ecosystem5, ecosystem6, si19, collective and neuralese),
contracts under `docs/`, receipts under `evidence/`, one disclosed host runner
(`tests/si19/run.sh`) and Wolfram Language oracles. There is no CI. See AUDIT_SCOPE.md for the
inventory.

In scope:

- Tokens, credentials or private data committed to any file (AGENTS.md forbids this).
- A measurement presented as real without its receipt in `evidence/`.
- A `.om` unit that accepts malformed or untrusted input it should refuse.
- A pinned dependency or compiler hash that is not enforced by its runner or recipe.

Out of scope:

- The Omega engine and compiler; report those to `aien-dev/omega-mv`.
- Outside training frameworks used only as yardsticks, and the host operating system.

## Supported versions

Only the default branch `main`; there is no release. Reports against older commits are welcome but fixes land on the default branch only.

## Status

Pre-alpha research repository (AGENTS.md). Host CPU (aarch64) results only; no GPU or DGX Spark
results. No certification, no bug bounty. See AUDIT_SCOPE.md for what to review and how to build
and test it.
