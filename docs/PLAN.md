# Plan

| stream | directory | branch | writes | ledger |
|---|---|---|---|---|
| history-openai | streams/history-openai | study/openai | docs/history/openai.md | see ledger |
| history-others | streams/history-others | study/others | docs/history/others.md | see ledger |
| efficiency | streams/efficiency | study/efficiency | docs/history/efficiency.md | see ledger |
| physics-gb10 | streams/physics-gb10 | physics/gb10 | docs/physics/gb10-training.md, evidence/ | see ledger |
| synthesis | docs | docs/lessons | docs/LESSONS.md | coordinator |
| desk | - | main | merges, tags | desk |

Rules: no CUDA, no third-party training framework in the product path (PyTorch, JAX and friends are yardsticks and oracles only, never shipped). GPU work is Omega. Every number names the chip. Every outside claim is cited inline with the URL fetched this session.
