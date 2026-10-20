# E1 step 1: freeze SmolLM2-1.7B, extract final-norm features X and next tokens Y once, save to .scratch.
# Splits: fit = GSM8K train examples [0, 6000); val = train [6000, 7473) (for lambda / early stop); test = GSM8K test.
import os, time, torch
from common import *

CAP = {"fit": 262144, "val": 32768, "test": 65536}

tok, model = load_model()
train, test = load_split("train"), load_split("test")
parts = {"fit": train[:6000], "val": train[6000:], "test": test}
E = model.model.embed_tokens.weight.detach()  # tied LM head, [V, d]

# Check that last_hidden_state is the post-norm state the LM head reads: logits must match.
ids = encode(tok, parts["test"][:2])
x, m = next(batches(ids, 2))
with torch.no_grad():
    ref = model(input_ids=x.to(DEV), attention_mask=m.long().to(DEV)).logits.float()
    h = model.model(input_ids=x.to(DEV), attention_mask=m.long().to(DEV)).last_hidden_state
    mine = (h @ E.T).float()
md = (ref - mine)[m.to(DEV)].abs().max().item()
print("logit check max abs diff", md)
assert md < 1e-2, md

info = {"logit_check_max_abs_diff": md}
for name, exs in parts.items():
    t0 = time.time()
    X, Y = extract_final(model, encode(tok, exs), CAP[name])
    torch.save({"X": X, "Y": Y}, os.path.join(SCRATCH, f"e1-{name}.pt"))
    info[name] = {"examples_available": len(exs), "tokens": int(X.shape[0]), "d": int(X.shape[1]),
                  "sha256_X": sha_of_tensor(X), "seconds": time.time() - t0}
    print(name, info[name])
torch.save(E.float().cpu(), os.path.join(SCRATCH, "e1-embed.pt"))
info["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
receipt("e1-extract", info)
