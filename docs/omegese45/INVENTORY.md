# Omegese #45: publication-to-implementation inventory

Scope: what the first #45 deliverable takes from outside work, what it reuses from AIEN repos, and what is missing.
Every row names a pin. Anything this lane did not read is marked `[NOT INSPECTED]`. Checked 2026-10-29.

## 1. External publication (Recursive)

| Item | Pin | Status here |
|---|---|---|
| `recursive-org/first-steps-toward-automated-ai-research` | `git ls-remote … HEAD` → `a962ec43e2e3d7c018e59a2ece623fe6e232fdfb` | repository contents `[NOT INSPECTED]` by this lane |
| Recursive blog posts on the method | — | `[NOT INSPECTED]` |
| Methods adopted | — | none claimed by this lane; the method choice (supervised/self-supervised only, no RL) comes from [aien-train#45](https://github.com/aien-dev/aien-train/issues/45) and interface §0 |
| H100/B200 → GB10 changes | — | `[NOT INSPECTED]`: this lane has no source for the publication's hardware assumptions; #45 v1 trains on the GB10 Grace CPU only (pipeline lane), GPU training NOT_RUN by design |

## 2. Reused AIEN components (all read by this lane unless marked)

| Component | Pin | Use |
|---|---|---|
| omc (seeded Omega compiler) | omega-mv `bf54b19fa590e8ae925c0e80815516dbf41e7326`, `seed/out/omc` sha256 `62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b` | builds every contract unit |
| Omega reference | omega-mv `bf54b19f` `docs/one_layer/OMEGESE.md` L114 (types), L668 (OM-2), L691-693 (hex literals); `seed/omc/omc.om` `fn kw` L810 | vocab provenance (CONTRACT.md §8) |
| MJB1/MJR1 schema | aien-protocols-mv `e705fdc9c97394f17f32f1dcbcbf038ca556250b` `crates/aien-microjob/src/lib.rs` L34-40, L103-109, L887-928; `specs/microjob/SPEC.md` L162-190 | `mjr1.om` rules |
| Real MJR1 receipt | aien-protocols-mv `e705fdc9` `evidence/aienet0-20261028/exercised/artifacts/aadc3775…94cb.mjr1` (sha256 `3306d180…4855`) | suite fixture |
| Aienet-0 worker/verifier | open-humanity `53c39bf2031c` `aienet0-serve.rs`, `aienet0-run.rs` | `[NOT INSPECTED]` by this lane; runtime use is the demo lane's |
| Native tape | omega-mv tape `965fb7627364274878c35a49e132fae492ff53be`, pinned by aien-train PR #42 `8307268a` | not used by contract units; PR #42 `run-adtrain.sh` is the suite-script pattern copied |

## 3. Owners and linked issues

| Issue | State | Relation |
|---|---|---|
| [aien-architecture-mv#307](https://github.com/aien-dev/aien-architecture-mv/issues/307) | OPEN | SI Genesis/Experience coordination; owner of the cross-repo pilot plan |
| [aien-protocols-mv#50](https://github.com/aien-dev/aien-protocols-mv/issues/50) | OPEN | MJB1/MJR1 ABI owner (the schema `mjr1.om` follows) |
| [omega-mv#695](https://github.com/aien-dev/omega-mv/issues/695) | OPEN | GB10 microbatch scheduler; not used in v1 (no GPU) |

Issue bodies beyond title and state: `[NOT INSPECTED]`.

## 4. Missing for later deliverables

- MJR1 signature verification in `.om` (P-256) — `mjr1.om` applies the static rules only, like `decode_receipt`; signatures are verified by aienet0-run.
- A transformer core (softmax needs exp; no `.om` exp intrinsic) — interface §6, extension proposal.
- Widening the job allowlist beyond `matmul_u32` 1..256 — cross-repo owner change, out of scope.
