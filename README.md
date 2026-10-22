# aien-train

Pre-alpha. Training my own language model on AIEN's own engine (Omega, no CUDA, no third-party training framework), starting on one DGX Spark (GB10).

The thesis under test: the big labs got where they are by pushing mass data and mass compute through a stack they built as they went. Their pitfalls are known now. If I go back through that history first, write the lessons down, and design the training run around the physics of the chip (keep the work on the math cores, stop streaming weights through slow memory every step), a small machine can train a real model the right way the first time.

Every claim in `docs/` links to a source or to a receipt in `evidence/`. Numbers carry the chip they were measured on.

Layout:
- `docs/history/` what each lab did right and wrong, cited (OpenAI; Anthropic, xAI, Google, Meta, DeepSeek, Mistral, Microsoft; the efficiency frontier)
- `docs/LESSONS.md` the distilled rules I train by (written after the history lands)
- `docs/physics/` GB10 training roofline and the on-chip ("math cores only") training question
- `evidence/` receipts for every measurement
- `docs/PLAN.md` streams, directories, branches
