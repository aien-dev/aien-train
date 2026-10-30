# aien-train

Pre-alpha. Training our own language model on AIEN's own engine (Omega, no CUDA, no third-party training framework), starting on one DGX Spark (GB10).

The thesis under test: the big labs got where they are by pushing mass data and mass compute through a stack they built as they went. We know their pitfalls now. If we go back through that history first, write the lessons down, and design the training run around the physics of the chip (keep the work on the math cores, stop streaming weights through slow memory every step), a small machine can train a real model the right way the first time.

Every claim in `docs/` links to a source or to a receipt in `evidence/`. Numbers carry the chip they were measured on.

Layout:
- `docs/history/` what each lab did right and wrong, cited (OpenAI; Anthropic, xAI, Google, Meta, DeepSeek, Mistral, Microsoft; the efficiency frontier)
- `docs/LESSONS.md` the distilled rules we train by (written after the history lands)
- `docs/physics/` GB10 training roofline and the on-chip ("math cores only") training question
- `evidence/` receipts for every measurement
- `docs/PLAN.md` streams, directories, branches

## Human-Directed Agentic Engineering

Development follows Human-Directed Agentic Engineering (HDAE): humans define goals, architecture, constraints, acceptance criteria, and stopping points; agents implement, execute checks, diagnose failures, and revise within those boundaries. Human review retains responsibility for claims, acceptance, and release decisions.

For the documented method and a systems-software case study, see Gregory M. Kurtzer, [*Ginux: Building an Independent Kernel That Runs Linux Userspace through Human-Directed Agentic Engineering* (2026)](https://doi.org/10.5281/zenodo.23682548). This is a development-method citation, not evidence that a training run or earlier experiment has passed.
