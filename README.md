<!-- github-native-navigation -->
[Source](https://github.com/aien-dev/aien-train) · [Walkthrough](docs/TEACHING.md) · [Replay receipt](docs/teaching-replay-2026-10-22.txt) · [Issues](https://github.com/aien-dev/aien-train/issues)
<!-- /github-native-navigation -->

# aien-train

## How it works and how to reproduce

I trace one actual source or design boundary in [the worked lesson](docs/TEACHING.md), with a pinned fresh-checkout exercise and [raw replay receipt](docs/teaching-replay-2026-10-22.txt). The lesson states what remains unimplemented or unverified at this revision.

Pre-alpha design for training a language model on AIEN's proposed Omega engine path (no CUDA or third-party training framework in the product path), starting with one DGX Spark (GB10). No trainer or training result is in this branch.

My thesis to test: study prior training approaches and the GB10 memory and compute budget before designing a small-model run. This branch contains a plan for those studies, not measurements or a trained model.

Every claim in `docs/` links to a source or to a receipt in `evidence/`. Numbers carry the chip they were measured on.

Layout:
- `docs/history/` what each lab did right and wrong, cited (OpenAI; Anthropic, xAI, Google, Meta, DeepSeek, Mistral, Microsoft; the efficiency frontier)
- `docs/LESSONS.md` the distilled rules we train by (written after the history lands)
- `docs/physics/` GB10 training roofline and the on-chip ("math cores only") training question
- `evidence/` receipts for every measurement
- `docs/PLAN.md` streams, directories, branches
