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

At this revision the repository holds only `README.md`, `AGENTS.md`, `docs/PLAN.md`, `LICENSE`
and empty `.keep` placeholders. There is no code, build or CI (AGENTS.md).

In scope:

- Tokens, credentials or private data committed to any file (AGENTS.md forbids this).
- A measurement presented as real without its receipt in `evidence/`.
- Once code lands: anything in the training path that reads untrusted data or runs GPU work.

Out of scope:

- The Omega engine and compiler; report those to `aien-dev/omega-mv`.
- Outside training frameworks used only as yardsticks, and the host operating system.

## Supported versions

Only the default branch `main`; there is no release. Reports against older commits are welcome but fixes land on the default branch only.

## Status

Pre-alpha research repository (AGENTS.md). No code, no measurements, no receipts yet. No
certification, no bug bounty. See AUDIT_SCOPE.md for what to review and how to build and test it.
