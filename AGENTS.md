# AIEN Train (training our own language model on the Omega engine)

> Pre-alpha research repository. The goal is to train a language model on AIEN's own engine (Omega, no CUDA, no third-party training framework), starting on one DGX Spark (GB10). The repository contains research plans and bounded native components with focused host tests and evidence. These components do not constitute an integrated trainer, distributed transport, or hardware qualification.

Shared rules: read https://www.aienos.com/agents/RULES.md first. This file adds what is specific to this repository; where the two differ, this file wins here.

## Scope

- Applies to the whole repository.
- `docs/PLAN.md`: the research plan. Component contracts live in their own `docs/` subdirectories.
- Planned, not yet written: `docs/history/` (what each lab did right and wrong, cited), `docs/LESSONS.md` (the distilled rules) and `docs/physics/` (GB10 training roofline). `docs/PLAN.md` says which stream writes each.
- `evidence/`: retained source pins, test results and measurement receipts. Preserve historical evidence.
- `streams/`: one working directory per stream (see `docs/PLAN.md`).

## Installation

```bash
git clone https://github.com/aien-dev/aien-train.git
cd aien-train
```

Native components have focused build and test recipes in their contracts and evidence records. For distributed metadata admission, see `docs/distributed7/CONTRACT.md` and `evidence/DISTRIBUTED7/README.md`. Host tests do not establish GPU or distributed execution. Hardware measurements still require an authorized hardware window.

## Usage

Work by stream. `docs/PLAN.md` maps each stream to its directory, branch and output file:

| Stream | Directory | Branch | Writes |
|---|---|---|---|
| history-openai | `streams/history-openai` | `study/openai` | `docs/history/openai.md` |
| history-others | `streams/history-others` | `study/others` | `docs/history/others.md` |
| efficiency | `streams/efficiency` | `study/efficiency` | `docs/history/efficiency.md` |
| physics-gb10 | `streams/physics-gb10` | `physics/gb10` | `docs/physics/gb10-training.md`, `evidence/` |
| synthesis | `docs` | `docs/lessons` | `docs/LESSONS.md` |

`main` receives merges and tags only.

## Conventions

- Every claim in `docs/` links to a source or to a receipt in `evidence/`.
- Every outside claim is cited inline with the URL you actually fetched in that session. Never cite from memory.
- Every number names the chip it was measured on.
- No CUDA and no third-party training framework in the product path. PyTorch, JAX and similar are yardsticks and oracles only, never shipped. GPU work is Omega.
- Stay on your stream's branch and directory; do not write another stream's output file.

## Safety and approvals

- Never write a measurement without its receipt in `evidence/`. No synthetic or estimated numbers presented as measured.
- Do not run GPU work unless you have been given an authorized window on the hardware.
- Never commit tokens, credentials, or private data. Work on a branch and merge through a pull request.

## Further reading

- `README.md`: the thesis and layout.
- `docs/PLAN.md`: streams, directories, branches and rules.
