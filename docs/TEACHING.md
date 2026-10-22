# Learn the training budget before a run

At `ed8b6ea2562efdda7913234ca8f34b443714a987`, this repository records a training hypothesis and [stream assignments](PLAN.md). It has no trainer, model configuration, corpus manifest, tokenization code or measured run in `main`; [`evidence/.keep`](../evidence/.keep) is empty. The plan places history, efficiency and GB10 physics notes on separate branches. I cannot turn those pending notes into historical conclusions or a claimed trained model.

The proposed loop is data and tokenization, forward pass, loss, backward pass, optimizer update, checkpoint and evaluation. The memory question is whether weights, gradients, optimizer state and activations fit while arithmetic remains busy. This is a planning calculation, not a GB10 measurement: for 1,000,000,000 parameters, FP16 weights (2 bytes), FP16 gradients (2), two FP32 Adam moments (8) total 12,000,000,000 bytes before master weights, activations, temporary buffers or checkpoint staging. The arithmetic and omissions are explicit so a newcomer can challenge the design before allocating hardware. The no-CUDA and no-third-party-training-framework product rule comes from [PLAN.md](PLAN.md); Omega integration remains a target.

## Fresh checkout exercise (offline, no model)

Prerequisites: Git and Python 3. Checkout `ed8b6ea2562efdda7913234ca8f34b443714a987`, then run from repository root:

```sh
python3 - <<'PYEX'
from pathlib import Path
n=1_000_000_000
print(f'base bytes={n*(2+2+8):,}')
assert not list(Path('.').glob('**/train.py'))
assert not Path('docs/physics/gb10-training.md').exists()
print('TRAINER AND PHYSICS NOTE UNAVAILABLE ON MAIN')
PYEX
```

Expected `base bytes=12,000,000,000` and `TRAINER AND PHYSICS NOTE UNAVAILABLE ON MAIN`, exit 0. This tests arithmetic and missing inputs, not training. A real test needs a pinned corpus, model, optimizer, loss trace, checkpoint and independently replayed evaluation. [Replay receipt](teaching-replay-2026-10-22.txt).


### Parent fresh-checkout replay

A separate coordination replay is retained in [the parent receipt](parent-replay-2026-10-22.txt). Its scope and observed exit are explicit; source review and final desk qualification remain separate. A recorded absence or expected failing baseline is not a passing implementation.
