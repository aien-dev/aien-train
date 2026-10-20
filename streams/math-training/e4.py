# E4: how many dimensions does the between-layer state (the residual stream) really use, and what does keeping it small cost?
#   python e4.py   -> writes evidence/math-training/e4-subspace.json
# Fit PCA at every layer boundary on GSM8K train[0:80] (positions >= 1), evaluate on GSM8K test[0:40].
# Truncation: h -> mu + U_k U_k^T (h - mu) at the output of a layer, for positions >= 1 (position 0, the attention-sink token, is kept).
import json, os, math
import torch
from common import *

torch.backends.cuda.matmul.allow_tf32 = False
tok, model = load_model()
layers = model.model.layers
NL, d = len(layers), 2048
tr, te = load_split("train"), load_split("test")
fit_ids, ev_ids = encode(tok, tr[:80]), encode(tok, te[:40])

# ---- capture layer outputs (residual stream after layer l) on fit text
caps = [[] for _ in range(NL)]
def cap(li):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        caps[li].append(h[0, 1:].float())          # batch 1, drop position 0
    return f
hs = [l.register_forward_hook(cap(i)) for i, l in enumerate(layers)]
with torch.no_grad():
    for x, m in batches(fit_ids, 1):
        model.model(input_ids=x.to(DEV))
for h in hs:
    h.remove()

pca, out = [], {"per_layer": []}
for li in range(NL):
    H = torch.cat(caps[li]).double()
    mu = H.mean(0)
    C = (H - mu).T @ (H - mu) / len(H)
    ev, U = torch.linalg.eigh(C)
    ev, U = ev.flip(0).clamp_min(0), U.flip(1)
    c = torch.cumsum(ev, 0) / ev.sum()
    dims = {str(l): int((c < l).sum()) + 1 for l in (0.95, 0.99, 0.999)}
    # outlier ("massive") dims: per-coordinate RMS more than 20x the median coordinate RMS
    rms = H.pow(2).mean(0).sqrt()
    outl = torch.nonzero(rms > 20 * rms.median()).flatten()
    keep = torch.ones(d, dtype=torch.bool, device=H.device); keep[outl] = False
    Hk = H[:, keep]
    Ck = (Hk - Hk.mean(0)).T @ (Hk - Hk.mean(0)) / len(Hk)
    evk = torch.linalg.eigvalsh(Ck).flip(0).clamp_min(0)
    ck = torch.cumsum(evk, 0) / evk.sum()
    dims_k = {str(l): int((ck < l).sum()) + 1 for l in (0.95, 0.99, 0.999)}
    pca.append((mu.float(), U.float()))
    out["per_layer"].append({"layer": li, "tokens": len(H), "energy_dims": dims, "outlier_dims": outl.tolist(),
                             "energy_dims_without_outlier_dims": dims_k,
                             "share_of_uncentered_energy_in_outlier_dims": float(H[:, outl].pow(2).sum() / H.pow(2).sum())})
    caps[li] = None
print(json.dumps(out["per_layer"][::6])[:1500], flush=True)

# ---- subspace stability: mean cos^2 of principal angles between top-k subspaces of consecutive boundaries
for k in (64, 256, 1024):
    out[f"consecutive_overlap_k{k}"] = [float((pca[l][1][:, :k].T @ pca[l + 1][1][:, :k]).pow(2).sum() / k) for l in range(NL - 1)]
    out[f"first_last_overlap_k{k}"] = float((pca[0][1][:, :k].T @ pca[-1][1][:, :k]).pow(2).sum() / k)

# ---- truncation experiments
state = {"k": None, "only": None, "rand": None}
def trunc(li):
    def f(mod, inp, outp):
        if state["k"] is None or (state["only"] is not None and state["only"] != li):
            return None
        h = outp[0] if isinstance(outp, tuple) else outp
        mu, U = pca[li]
        Uk = state["rand"][li] if state["rand"] is not None else U[:, :state["k"]]
        hf = h.float()
        p = mu + ((hf - mu) @ Uk) @ Uk.T
        p[:, 0] = hf[:, 0]
        p = p.to(h.dtype)
        return (p,) + tuple(outp[1:]) if isinstance(outp, tuple) else p
    return f
hs = [l.register_forward_hook(trunc(i)) for i, l in enumerate(layers)]

ref_logits = []
@torch.no_grad()
def run(collect=False):
    tot, n, agree, i = 0.0, 0, 0, 0
    for x, m in batches(ev_ids, 1):
        x = x.to(DEV)
        lg = model(input_ids=x).logits[0, :-1].float()
        tot += float(torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum")); n += lg.shape[0]
        a = lg.argmax(1)
        if collect:
            ref_logits.append(a)
        else:
            agree += int((a == ref_logits[i]).sum())
        i += 1
    return tot / n, (agree / n if not collect else 1.0)

base_nll, _ = run(collect=True)
out["heldout_nll_full"] = base_nll
res = []
for k in (64, 128, 256, 512, 1024, 1536, 2048):
    state.update(k=k, only=None, rand=None)
    nll, ag = run()
    res.append({"k": k, "nll": nll, "nll_increase": nll - base_nll, "top1_agreement_with_full": ag})
    print(res[-1], flush=True)
out["all_boundaries_pca"] = res
g = torch.Generator(device="cpu").manual_seed(0)
rr = []
for k in (512, 1024, 1536):
    state.update(k=k, only=None, rand=[torch.linalg.qr(torch.randn(d, k, generator=g))[0].to(DEV) for _ in range(NL)])
    nll, ag = run()
    rr.append({"k": k, "nll_increase": nll - base_nll, "top1_agreement_with_full": ag})
    print("rand", rr[-1], flush=True)
out["all_boundaries_random_subspace_control"] = rr
one = []
for li in range(NL):
    state.update(k=256, only=li, rand=None)
    nll, ag = run()
    one.append({"layer": li, "k": 256, "nll_increase": nll - base_nll, "top1_agreement_with_full": ag})
out["one_boundary_at_k256"] = one
print("one", [round(o["nll_increase"], 3) for o in one], flush=True)
state.update(k=None, only=None, rand=None)
for h in hs:
    h.remove()

# ---- the fold: with h = mu + U z (U orthonormal, d x k), q = W diag(g) (h / rms(h)) needs only z:
#   rms(h)^2 = (|mu|^2 + 2 mu^T U z + |z|^2) / d,   W diag(g) h = (W diag(g) mu) + (W diag(g) U) z
li, k = 12, 512
mu, U = pca[li - 1]
Uk = U[:, :k]
lay = layers[li]
g_attn = lay.input_layernorm.weight.float()
Wq = lay.self_attn.q_proj.weight.float()
with torch.no_grad():
    x, _ = next(batches(ev_ids[:1], 1))
    hcap = []
    hk = layers[li - 1].register_forward_hook(lambda m_, i_, o_: hcap.append((o_[0] if isinstance(o_, tuple) else o_)[0, 1:].float()))
    model.model(input_ids=x.to(DEV)); hk.remove()
    h = mu + ((hcap[0] - mu) @ Uk) @ Uk.T                  # a state that lies exactly in the subspace
    z = (h - mu) @ Uk
    eps = model.config.rms_norm_eps
    direct = (h / torch.sqrt(h.pow(2).mean(1, keepdim=True) + eps) * g_attn) @ Wq.T
    r2 = (mu @ mu + 2 * z @ (Uk.T @ mu) + z.pow(2).sum(1)) / d
    folded = ((Wq * g_attn) @ mu + z @ ((Wq * g_attn) @ Uk).T) / torch.sqrt(r2 + eps)[:, None]
out["fold_check"] = {"layer": li, "k": k, "rel_err_folded_vs_direct_fp32": float((folded - direct).norm() / direct.norm())}
print(out["fold_check"])

# ---- bytes: decode weight reads per token at k vs d (bf16), at the measured 245.1 GB/s (L279 / L151)
cfg = model.config
F = cfg.intermediate_size
per_layer = 4 * d * d + 3 * d * F
lm_head = cfg.vocab_size * d
full = (NL * per_layer + lm_head) * 2
kv_per_ctx_token = NL * 2 * d * 2
bw = 245.1e9
tab = []
for k in (256, 512, 1024, 1536, 2048):
    w = full * k / d + NL * k * k * 2                         # every matrix shrinks by k/d; plus one k x k change of basis per boundary
    tab.append({"k": k, "weight_bytes_per_token": w, "ms_at_245GBps_weights_only": w / bw * 1e3,
                "kv_bytes_per_token_at_ctx_1024_unchanged": kv_per_ctx_token * 1024})
out["decode_bytes"] = {"full_weight_bytes_per_token": full, "table": tab,
                       "note": "weights only, bf16, one decode token; KV-cache reads do not shrink with k"}
receipt("e4-subspace", out)
