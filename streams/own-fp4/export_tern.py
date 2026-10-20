# export_tern.py (L284 own-fp4): write one trained ternary layer + real activations for omega tests/gpu_tern_layer_test.c.
#   ~/.venv/bin/python -I export_tern.py RUN_DIR LAYER(e.g. b2.fc2) M OUT.trn
# Codes follow fp4sim.ternary exactly (absmean, per tensor or per 16x16 tile), activations are the layer's real input on
# held-out text, quantized to int8 per vector by absmax (BitNet's A8). Yref = exact integer W x.
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as TR

run, layer, M, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
meta = json.load(open(f"{run}/meta.json")); a = meta["args"]; cfg = meta["cfg"]
torch.manual_seed(0)
TR.Ctx.dev = "cpu"; TR.Ctx.gen = torch.Generator().manual_seed(1)
TR.Ctx.H16 = TR.Q.hadamard(16, "cpu"); TR.Ctx.R16 = TR.Ctx.H16
model = TR.GPT(32000, a["d"], a["layers"], a["heads"], a["ctx"], cfg)
model.load_state_dict(torch.load(f"{run}/model.pt")); model.eval()
mod = dict(model.named_modules())[layer]
w = mod.weight.detach().float()
if cfg.get("ternary") == "tensor":
    T = (w / w.abs().mean().clamp(min=1e-8)).round().clamp(-1, 1)
else:
    O, I = w.shape; t = w.reshape(O // 16, 16, I // 16, 16)
    s = TR.Q.e4m3(t.abs().mean(dim=(1, 3), keepdim=True).clamp(min=1e-8))
    T = (t / s).round().clamp(-1, 1).reshape(O, I)
cap = {}
mod.register_forward_hook(lambda m_, i_, o_: cap.__setitem__("x", i_[0].detach().float().reshape(-1, i_[0].shape[-1])))
va = np.memmap(os.path.expanduser("~/workspace/aien-train-own-fp4-data/validation.u16"), dtype=np.uint16, mode="r")
idx = torch.from_numpy(va[1000: 1000 + a["ctx"]].astype(np.int64))[None]
with torch.no_grad(): model(idx)
X = cap["x"][:M]
xs = X.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127.0
Xq = (X / xs).round().clamp(-127, 127)
Y = (Xq.double() @ T.double().T).round().to(torch.int64)
assert Y.abs().max() < 2 ** 31
n, k = T.shape
with open(out, "wb") as f:
    f.write(b"TRN1"); f.write(np.array([n, k, M], dtype=np.uint32).tobytes())
    f.write(T.to(torch.int8).numpy().tobytes()); f.write(Xq.to(torch.int8).numpy().tobytes()); f.write(Y.to(torch.int32).numpy().tobytes())
print(json.dumps(dict(run=run, layer=layer, n=n, k=k, m=M, nonzero=float((T != 0).float().mean()), file=out)))
