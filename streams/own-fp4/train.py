# train.py (L284 own-fp4, ORACLE ONLY): one small GPT trained under one numerics recipe. Same seed, same tokens, same
# schedule for every recipe; only the numbers inside the linear layers change.
#   ~/.venv/bin/python -I train.py --recipe nvfp4 --out DIR [--steps N]
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4sim as Q

# recipe -> settings. fwd/bwd formats: "bf16" | "nv" (1x16 UE4M3 + FP32 tensor) | "mx" (1x32 E8M0)
RECIPES = {
    "bf16":     dict(fmt="bf16"),
    # NVIDIA's published recipe (arXiv 2509.25149): W 16x16 2D, X and dY 1x16, RNE fwd, SR on dY (Dgrad + Wgrad),
    # 16x16 random Hadamard (one fixed sign vector) on Wgrad inputs, last ~15% of blocks in BF16, FP32 master.
    "nvfp4":    dict(fmt="nv", w2d=True, sr_dy=True, rot="rht16", hp_last=1),
    # MXFP4 as the OCP spec quantizes it, no recipe extras (control)
    "mxfp4":    dict(fmt="mx", w2d=False, sr_dy=False, rot="none", hp_last=0),
    # (a0) 4-bit weights with no master copy at all (updates below half a step are lost): negative control for (a)
    "w4only":   dict(fmt="nv", w2d=True, sr_dy=True, rot="rht16", hp_last=1, master="none"),
    # (a) error feedback: no FP32 master; stored weight = 4-bit code + 8-bit residual of the last rounding
    "ef8":      dict(fmt="nv", w2d=True, sr_dy=True, rot="rht16", hp_last=1, master="ef8"),
    "ef8sr":    dict(fmt="nv", w2d=True, sr_dy=True, rot="rht16", hp_last=1, master="ef8sr"),
    # (b) NVFP4 recipe with the Wgrad rotation swapped for a cheaper transform chosen by the outlier study
    "b_none":   dict(fmt="nv", w2d=True, sr_dy=True, rot="none", hp_last=1),
    "b_h16":    dict(fmt="nv", w2d=True, sr_dy=True, rot="h16", hp_last=1),
    "b_diag":   dict(fmt="nv", w2d=True, sr_dy=True, rot="diag", hp_last=1),
    # (c) ternary weights {-1,0,+1} x scale (BitNet b1.58 absmean, per tensor), NVFP4 activations and gradients
    "tern":     dict(fmt="nv", w2d=True, sr_dy=True, rot="rht16", hp_last=1, ternary="tensor"),
    "tern16":   dict(fmt="nv", w2d=True, sr_dy=True, rot="rht16", hp_last=1, ternary="block"),
    # (d) literature-inspired: rotate every GEMM's inputs (Quartet-style), per-block MSE-best clip on forward tensors
    "d_rotmse": dict(fmt="nv", w2d=True, sr_dy=True, rot="rht16", hp_last=1, fwd_mse=True, rot_fwd=True),
    # all 4-bit, nothing kept in BF16 (ablation of the 15% rule)
    "nvfp4_all": dict(fmt="nv", w2d=True, sr_dy=True, rot="rht16", hp_last=0),
}

class Ctx:  # numerics state shared by all quantized layers
    gen = None; R16 = None; H16 = None; RF = None; stats = None; dev = None

def qa(x, dim, cfg, sr=False, mse=False):
    if cfg["fmt"] == "mx": return Q.quant_mx(x, dim, sr, Ctx.gen)
    return Q.quant_nv(x, dim, sr, Ctx.gen, scale_mode="mse" if mse else "amax")

def qw(w, cfg, dim):
    if cfg.get("ternary"): return Q.ternary(w, "tensor" if cfg["ternary"] == "tensor" else "block")
    if cfg["w2d"]: return Q.quant_nv2d(w)
    return qa(w, dim, cfg)

def rot_tokens(x, dy, cfg):
    r = cfg["rot"]
    if r == "none": return x, dy
    if r == "diag":  # exact per-token balancing: X_t * s_t, dY_t / s_t with s_t = sqrt(|dY_t| / |X_t|) (dW unchanged)
        s = (dy.float().norm(dim=1) / x.float().norm(dim=1).clamp(min=1e-30)).clamp(min=1e-30).sqrt()
        s = torch.exp2(torch.round(torch.log2(s)))  # power of two: exact on the chip
        return x.float() * s[:, None], dy.float() / s[:, None]
    R = Ctx.R16 if r == "rht16" else Ctx.H16
    return Q.rotate_tokens(x, R), Q.rotate_tokens(dy, R)

def stat(name, a, b):
    if Ctx.stats is not None: Ctx.stats.setdefault(name, []).append(Q.rel_err(a, b))

class QLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, cfg, tag):
        ctx.cfg, ctx.tag = cfg, tag
        ctx.save_for_backward(x, w)
        xin, win = x, w
        if cfg.get("rot_fwd"):  # rotate the contraction (in) dim of both operands: X R^T R W^T = X W^T
            xin = (x.float().reshape(-1, x.shape[-1] // 16, 16) @ Ctx.R16.T).reshape(x.shape)
            win = (w.float().reshape(w.shape[0], -1, 16) @ Ctx.R16.T).reshape(w.shape)
        xs, gx = qa(xin, -1, cfg, mse=cfg.get("fwd_mse", False))
        if cfg.get("rot_fwd") and not cfg.get("ternary"): ws, gw = Q.quant_nv(win, -1, scale_mode="mse" if cfg.get("fwd_mse") else "amax")
        else: ws, gw = qw(win, cfg, 1)
        stat(tag + ".x", xs * gx, xin); stat(tag + ".w", ws * gw, win)
        y = (xs.bfloat16() @ ws.bfloat16().T).float() * (gx * gw)
        return y.bfloat16()

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors; cfg = ctx.cfg
        dy = dy.reshape(-1, dy.shape[-1]); x2 = x.reshape(-1, x.shape[-1])
        # Dgrad: dX = dY W, contraction over out
        dys, gd = qa(dy, -1, cfg, sr=cfg["sr_dy"])
        if cfg.get("ternary") or cfg["w2d"]: ws, gw = qw(w, cfg, 0)
        else: ws, gw = qa(w, 0, cfg)
        stat(ctx.tag + ".dy", dys * gd, dy)
        dx = (dys.bfloat16() @ ws.bfloat16()).float() * (gd * gw)
        # Wgrad: dW = dY^T X, contraction over tokens (rotated per recipe)
        xr, dyr = rot_tokens(x2, dy, cfg)
        xq, g1 = qa(xr, 0, cfg); dq, g2 = qa(dyr, 0, cfg, sr=cfg["sr_dy"])
        stat(ctx.tag + ".wg_x", xq * g1, xr); stat(ctx.tag + ".wg_dy", dq * g2, dyr)
        dw = (dq.bfloat16().T @ xq.bfloat16()).float() * (g1 * g2)
        return dx.reshape(x.shape).bfloat16(), dw, None, None

class Lin(nn.Module):
    def __init__(self, i, o, cfg, tag):
        super().__init__(); self.weight = nn.Parameter(torch.randn(o, i) * 0.02); self.cfg = cfg; self.tag = tag
    def forward(self, x):
        if self.cfg["fmt"] == "bf16": return F.linear(x.bfloat16(), self.weight.bfloat16())
        sh = x.shape; y = QLinearFn.apply(x.reshape(-1, sh[-1]).bfloat16(), self.weight, self.cfg, self.tag)
        return y.reshape(*sh[:-1], -1)

class Block(nn.Module):
    def __init__(self, d, h, cfg, i):
        super().__init__(); self.h = h
        self.n1 = nn.RMSNorm(d); self.n2 = nn.RMSNorm(d)
        self.qkv = Lin(d, 3 * d, cfg, f"b{i}.qkv"); self.proj = Lin(d, d, cfg, f"b{i}.proj")
        self.fc1 = Lin(d, 4 * d, cfg, f"b{i}.fc1"); self.fc2 = Lin(4 * d, d, cfg, f"b{i}.fc2")
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, -1)
        q, k, v = (t.reshape(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a).float()
        return x + self.fc2(F.gelu(self.fc1(self.n2(x)).float()).bfloat16()).float()

class GPT(nn.Module):
    def __init__(self, V, d, L, h, ctx, cfg):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Parameter(torch.randn(ctx, d) * 0.01)
        bf = dict(fmt="bf16")
        self.blocks = nn.ModuleList(Block(d, h, cfg if i < L - cfg.get("hp_last", 0) else bf, i) for i in range(L))
        self.nf = nn.RMSNorm(d)
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, idx):
        x = self.emb(idx) + self.pos[: idx.shape[1]]
        for b in self.blocks: x = b(x)
        return F.linear(self.nf(x).bfloat16(), self.emb.weight.bfloat16()).float()  # tied head, BF16 per recipe

def ef_store(model, mode):
    """master-copy handling after each optimizer step for quantized weights (a0 / a)."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, Lin) and m.cfg["fmt"] != "bf16":
                w = m.weight.data; ws, g = Q.quant_nv2d(w); wq = ws * g
                if mode == "none": m.weight.data.copy_(wq); continue
                r = (w - wq); O, I = r.shape; t = r.reshape(O // 16, 16, I // 16, 16)
                s = torch.exp2(torch.ceil(torch.log2(t.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-30) / 448.0)))
                y = t / s
                if mode == "ef8sr":  # stochastic rounding into E4M3: keeps updates smaller than one residual step on average
                    ulp = torch.exp2(torch.floor(torch.log2(y.abs().clamp(min=2.0 ** -9))).clamp(min=-6) - 3)
                    lo = torch.floor(y / ulp) * ulp
                    yq = lo + ulp * (torch.rand(y.shape, device=y.device, generator=Ctx.gen) < (y - lo) / ulp).float()
                else: yq = Q.e4m3(y)
                rq = (yq * s).reshape(O, I)  # residual in FP8 E4M3, power-of-two scale per 16x16 tile
                m.weight.data.copy_(wq + rq)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/workspace/aien-train-own-fp4-data"))
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=256); ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1.5e-3); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--stats_every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump_steps", default="")  # save Wgrad operands (X, dY) of 4 layers at these steps (outlier study)
    a = ap.parse_args(); cfg = RECIPES[a.recipe]; cfg.setdefault("master", "fp32")
    os.makedirs(a.out, exist_ok=True); dev = "cuda"; Ctx.dev = dev
    torch.manual_seed(a.seed); Ctx.gen = torch.Generator(device=dev); Ctx.gen.manual_seed(a.seed + 7)
    sgn = torch.where(torch.rand(16, generator=torch.Generator().manual_seed(11)) < 0.5, -1.0, 1.0)
    Ctx.H16 = Q.hadamard(16, dev); Ctx.R16 = Ctx.H16 * sgn.to(dev)[None, :]  # one fixed random sign vector (paper)
    tr = np.memmap(f"{a.data}/train.u16", dtype=np.uint16, mode="r"); va = np.memmap(f"{a.data}/validation.u16", dtype=np.uint16, mode="r")
    V = 32000
    model = GPT(V, a.d, a.layers, a.heads, a.ctx, cfg).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() == 2 and "pos" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() == 2 and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    if cfg["master"] != "fp32": ef_store(model, cfg["master"])
    rng = np.random.default_rng(1234)  # identical token order for every recipe
    starts = rng.integers(0, len(tr) - a.ctx - 1, size=(a.steps, a.batch))
    vstarts = np.random.default_rng(99).integers(0, len(va) - a.ctx - 1, size=(100, a.batch))
    def batch(arr, st):
        x = np.stack([arr[s: s + a.ctx + 1] for s in st]).astype(np.int64); t = torch.from_numpy(x).to(dev)
        return t[:, :-1], t[:, 1:]
    def evaluate(nb):
        model.eval(); tot = 0.0
        with torch.no_grad():
            for i in range(nb):
                x, y = batch(va, vstarts[i]); tot += F.cross_entropy(model(x).view(-1, V), y.reshape(-1)).item()
        model.train(); return tot / nb
    log = open(f"{a.out}/log.jsonl", "w"); t0 = time.time(); warm = 200
    meta = dict(recipe=a.recipe, cfg=cfg, params=nparam, args=vars(a), tokens=a.steps * a.batch * a.ctx, label="oracle",
                torch=torch.__version__, device=torch.cuda.get_device_name(0))
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1); print(json.dumps(meta), flush=True)
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
        for gp in opt.param_groups: gp["lr"] = lr
        Ctx.stats = {} if (a.stats_every and step % a.stats_every == 0 and cfg["fmt"] != "bf16") else None
        x, y = batch(tr, starts[step])
        hooks = []; dumps = {}
        if a.dump_steps and step in [int(s) for s in a.dump_steps.split(",")]:
            for m in model.modules():
                if isinstance(m, Lin) and m.tag in ("b0.fc1", "b2.qkv", "b3.proj", "b4.fc2"):
                    def fh(mod, inp, out, tag=m.tag):
                        dumps[tag + ".x"] = inp[0].detach().reshape(-1, inp[0].shape[-1])[:4096].bfloat16().cpu()
                        out.register_hook(lambda g, tag=tag: dumps.__setitem__(tag + ".dy", g.detach().reshape(-1, g.shape[-1])[:4096].bfloat16().cpu()))
                    hooks.append(m.register_forward_hook(fh))
        loss = F.cross_entropy(model(x).view(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        for h in hooks: h.remove()
        if dumps: torch.save(dumps, f"{a.out}/dump_step{step}.pt")
        opt.step()
        if cfg["master"] != "fp32": ef_store(model, cfg["master"])
        rec = dict(step=step, loss=loss.item(), lr=lr, gnorm=gn, t=round(time.time() - t0, 1))
        if Ctx.stats: rec["qerr"] = {k: float(np.mean(v)) for k, v in Ctx.stats.items()}
        if step % a.eval_every == 0 or step == a.steps - 1: rec["val"] = evaluate(20 if step < a.steps - 1 else 100)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 100 == 0 or "val" in rec: print(json.dumps({k: v for k, v in rec.items() if k != "qerr"}), flush=True)
        if not math.isfinite(rec["loss"]): print("DIVERGED"); break
    res = dict(recipe=a.recipe, final_val=rec.get("val"), final_train=rec["loss"], seconds=round(time.time() - t0, 1), steps=step + 1, label="oracle")
    json.dump(res, open(f"{a.out}/result.json", "w")); print("RESULT", json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
