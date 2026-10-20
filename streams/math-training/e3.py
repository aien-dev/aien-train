# E3: do real weight matrices live in a small closed algebra (structure constants)?
#   python e3.py toy      -> exact checks of the paste's toy claims (rational arithmetic), writes e3-toy.json
#   python e3.py algebra  -> per-head OV/QK spans, closure of virtual-head products, growth, controls, writes e3-algebra.json
#   python e3.py nonlin   -> linear fit of each gated MLP on real activations, and the NLL cost of swapping it in, writes e3-nonlin.json
# Rank-64 head matrices are kept factored, M = L R^T (L, R are d x 64), so inner products and products never build d x d:
#   <L1 R1^T, L2 R2^T> = sum((L1^T L2) * (R1^T R2)),   (L1 R1^T)(L2 R2^T) = (L1 (R1^T L2)) R2^T.
import json, os, sys, random
from fractions import Fraction as Fr
import torch
from common import *

MODE = sys.argv[1]
HD, NH = 64, 32


def weights(layer):
    from safetensors import safe_open
    out = {}
    with safe_open(os.path.join(MODEL_DIR, "model.safetensors"), "pt") as f:
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[n] = f.get_tensor(f"model.layers.{layer}.self_attn.{n}.weight").float()
        for n in ("gate_proj", "up_proj", "down_proj"):
            out[n] = f.get_tensor(f"model.layers.{layer}.mlp.{n}.weight").float()
    return out


def heads_OV(w):
    """OV_h = W_O[:, h] W_V[h, :]  (x -> contribution of head h, ignoring the attention mixing)."""
    return [(w["o_proj"][:, h * HD:(h + 1) * HD], w["v_proj"][h * HD:(h + 1) * HD].T) for h in range(NH)]


def heads_QK(w):
    """QK_h = W_Q[h]^T W_K[h]  (bilinear score at relative position 0; RoPE rotates it at other offsets)."""
    return [(w["q_proj"][h * HD:(h + 1) * HD].T, w["k_proj"][h * HD:(h + 1) * HD].T) for h in range(NH)]


def gram(items, others=None):
    others = items if others is None else others
    G = torch.empty(len(items), len(others), dtype=torch.float64, device=items[0][0].device)
    for a, (La, Ra) in enumerate(items):
        LL = torch.stack([La.T @ Lb for Lb, _ in others])   # [n, r, r]
        RR = torch.stack([Ra.T @ Rb for _, Rb in others])
        G[a] = (LL * RR).sum((1, 2)).double()
    return G


def normalize(items):
    out = []
    for L, R in items:
        n = float(((L.T @ L) * (R.T @ R)).sum()) ** 0.5
        out.append((L / n ** 0.5, R / n ** 0.5))
    return out


def energy_dims(G, levels=(0.95, 0.99, 0.999)):
    ev = torch.linalg.eigvalsh(G).flip(0).clamp_min(0)
    c = torch.cumsum(ev, 0) / ev.sum()
    return {str(l): int((c < l).sum()) + 1 for l in levels}, ev


def matched_random(items, seed):
    """Control: same singular values per matrix, random orthonormal singular vectors."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for L, R in items:
        # exact SVD of L R^T from the factors: L = Ql Rl, R = Qr Rr, core = Rl Rr^T
        Ql, Rl = torch.linalg.qr(L)
        Qr, Rr = torch.linalg.qr(R)
        s = torch.linalg.svdvals(Rl @ Rr.T)
        d, r = L.shape
        A = torch.linalg.qr(torch.randn(d, r, generator=g).to(L.device))[0]
        B = torch.linalg.qr(torch.randn(d, r, generator=g).to(L.device))[0]
        out.append((A * s, B))
    return out


def closure_residual(B_items, P_items):
    """Relative residual of each P after orthogonal projection onto span(B): sqrt(1 - b^T G^-1 b / |P|^2)."""
    G = gram(B_items)
    b = gram(P_items, B_items)                       # [nP, nB]
    nP = torch.stack([((L.T @ L) * (R.T @ R)).sum() for L, R in P_items]).double()
    w = torch.linalg.solve(G + 1e-12 * torch.eye(len(B_items), dtype=G.dtype, device=G.device) * G.diagonal().mean(), b.T)
    proj = (b.T * w).sum(0)
    return torch.sqrt((1 - proj / nP).clamp_min(0))


def products(A_items, B_items, pairs):
    return [(A_items[i][0] @ (A_items[i][1].T @ B_items[j][0]), B_items[j][1]) for i, j in pairs]


if MODE == "toy":
    out = {}
    I = [[Fr(1), Fr(0)], [Fr(0), Fr(1)]]
    Q = [[Fr(1), Fr(1)], [Fr(1), Fr(0)]]
    mm = lambda A, B: [[sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    add = lambda A, B, a=1, b=1: [[a * A[i][j] + b * B[i][j] for j in range(2)] for i in range(2)]
    out["Q2_equals_Q_plus_I_exact"] = mm(Q, Q) == add(Q, I)
    # eigenvalues are roots of t^2 - t - 1 (trace 1, det -1); check numerically too
    import numpy as np
    ev = sorted(np.linalg.eigvalsh(np.array([[1.0, 1.0], [1.0, 0.0]])).tolist())
    out["eigenvalues"] = ev
    out["eigen_match_golden"] = abs(ev[1] - (1 + 5 ** 0.5) / 2) < 1e-14 and abs(ev[0] - (1 - 5 ** 0.5) / 2) < 1e-14
    # (aI + bQ)(cI + dQ) = (ac + bd) I + (ad + bc + bd) Q, exact on 200 random rationals
    rnd = random.Random(0)
    ok = True
    for _ in range(200):
        a, b, c, d = (Fr(rnd.randint(-50, 50), rnd.randint(1, 20)) for _ in range(4))
        lhs = mm(add(I, Q, a, b), add(I, Q, c, d))
        ok &= lhs == add(I, Q, a * c + b * d, a * d + b * c + b * d)
    out["product_rule_exact_200_random_rationals"] = ok
    # ReLU does not commute with Q: one witness
    relu = lambda v: [max(x, 0) for x in v]
    mv = lambda A, v: [sum(A[i][k] * v[k] for k in range(2)) for i in range(2)]
    v = [Fr(1), Fr(-2)]
    out["relu_witness"] = {"v": [str(x) for x in v], "Q_relu_v": [str(x) for x in mv(Q, relu(v))], "relu_Q_v": [str(x) for x in relu(mv(Q, v))]}
    out["relu_breaks_commutation"] = mv(Q, relu(v)) != relu(mv(Q, v))
    # chunked sufficient statistics are exactly additive (integer toy, exact rationals), and the ridge gradient is 0
    rows = [[Fr(rnd.randint(-5, 5)) for _ in range(3)] for _ in range(12)]
    ys = [Fr(rnd.randint(-5, 5)) for _ in range(12)]
    def stats(R, Y):
        G = [[sum(r[i] * r[j] for r in R) for j in range(3)] for i in range(3)]
        H = [sum(r[i] * y for r, y in zip(R, Y)) for i in range(3)]
        return G, H
    G1, H1 = stats(rows[:5], ys[:5]); G2, H2 = stats(rows[5:], ys[5:]); G, H = stats(rows, ys)
    out["chunked_G_H_exact"] = all(G1[i][j] + G2[i][j] == G[i][j] for i in range(3) for j in range(3)) and all(H1[i] + H2[i] == H[i] for i in range(3))
    lam = Fr(1, 2)
    A = [[G[i][j] + (lam if i == j else 0) for j in range(3)] + [H[i]] for i in range(3)]
    for c in range(3):  # exact Gauss-Jordan
        p = next(r for r in range(c, 3) if A[r][c] != 0); A[c], A[p] = A[p], A[c]
        A[c] = [x / A[c][c] for x in A[c]]
        for r in range(3):
            if r != c:
                A[r] = [x - A[r][c] * y for x, y in zip(A[r], A[c])]
    W = [A[i][3] for i in range(3)]
    grad = [sum(G[i][j] * W[j] for j in range(3)) + lam * W[i] - H[i] for i in range(3)]
    out["ridge_solution_exact"] = [str(x) for x in W]
    out["ridge_gradient_exact_zero"] = all(g == 0 for g in grad)
    print(out)
    receipt("e3-toy", out)

elif MODE == "algebra":
    dev = DEV if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {"per_layer": [], "notes": "items normalized to unit Frobenius norm before every Gram; energy = eigenvalue mass of the Gram"}
    rng = random.Random(0)
    # positive control: matrices drawn from the exact 2-dim algebra span{I, Q} kron I_1024 close perfectly
    d = 2048
    Qd = torch.kron(torch.tensor([[1.0, 1.0], [1.0, 0.0]]), torch.eye(d // 2)).to(dev)
    Id = torch.eye(d, device=dev)
    pc = [(a * Id + b * Qd, Id) for a, b in ((1.0, 0.3), (0.2, 1.0), (-0.5, 0.7))]
    pp = products(pc, pc, [(i, j) for i in range(3) for j in range(3)])
    out["positive_control_IQ_closure_residual_max"] = float(closure_residual(normalize(pc), normalize(pp)).max())
    prev = None
    for layer in range(24):
        w = {k: v.to(dev) for k, v in weights(layer).items()}
        ov, qk = normalize(heads_OV(w)), normalize(heads_QK(w))
        row = {"layer": layer}
        for name, items in (("OV", ov), ("QK", qk)):
            dims, _ = energy_dims(gram(items))
            dims_r, _ = energy_dims(gram(normalize(matched_random(items, layer * 7 + len(name)))))
            row[f"{name}_heads_energy_dims"] = dims
            row[f"{name}_heads_energy_dims_random_control"] = dims_r
        if prev is not None:
            # virtual heads: OV of this layer applied after OV of the previous layer (products the model actually forms)
            pairs = [(i, j) for i in range(NH) for j in range(NH)]
            sub = rng.sample(pairs, 256)
            P = normalize(products(ov, prev, sub))
            basis = ov + prev
            res = closure_residual(basis, P)
            ovr, prevr = normalize(matched_random(ov, layer)), normalize(matched_random(prev, layer + 100))
            res_r = closure_residual(ovr + prevr, normalize(products(ovr, prevr, sub)))
            row["virtual_head_closure_residual"] = {"median": float(res.median()), "min": float(res.min()), "max": float(res.max())}
            row["virtual_head_closure_residual_random_control"] = {"median": float(res_r.median()), "min": float(res_r.min()), "max": float(res_r.max())}
            if layer in (1, 6, 12, 18, 23):
                # growth: dims of span(B) and span(B + B.B) at 99% energy, B = 64 OV heads of two layers, 256 sampled products
                g1, _ = energy_dims(gram(basis))
                g2, _ = energy_dims(gram(basis + P))
                g2r, _ = energy_dims(gram(ovr + prevr + normalize(products(ovr, prevr, sub))))
                row["growth"] = {"n_basis": len(basis), "n_basis_plus_products": len(basis) + len(P),
                                 "dims_basis": g1, "dims_with_products": g2, "dims_with_products_random_control": g2r}
        prev = ov
        out["per_layer"].append(row)
        print(json.dumps(row)[:600], flush=True)
    # across depth: the 24 dense matrices of each kind
    depth = {}
    from safetensors import safe_open
    with safe_open(os.path.join(MODEL_DIR, "model.safetensors"), "pt") as f:
        for n, sec in (("q_proj", "self_attn"), ("k_proj", "self_attn"), ("v_proj", "self_attn"), ("o_proj", "self_attn"),
                       ("gate_proj", "mlp"), ("up_proj", "mlp"), ("down_proj", "mlp")):
            M = torch.stack([f.get_tensor(f"model.layers.{l}.{sec}.{n}.weight").float().reshape(-1) for l in range(24)]).to(dev)
            M = M / M.norm(dim=1, keepdim=True)
            G = (M @ M.T).double()
            depth[n] = {"energy_dims_of_24": energy_dims(G)[0], "mean_abs_cosine_offdiag": float((G - torch.eye(24, device=dev, dtype=G.dtype)).abs().sum() / (24 * 23))}
            del M
    out["across_depth"] = depth
    out["apply_cost_per_token_flops"] = {
        "dense_d_by_d": 2 * 2048 * 2048, "one_OV_head_factored": 2 * 2 * 2048 * 64,
        "virtual_head_factored": 2 * (2048 * 64 + 64 * 64 + 64 * 2048),
        "coefficients_over_r_dense_basis_elements": "r * 2 * d^2 (worse than dense for r > 1 unless each B_k is structured)"}
    print(json.dumps(depth))
    receipt("e3-algebra", out)

elif MODE == "nonlin":
    # Collect MLP input u = post_attention_layernorm(h) and output y = mlp(u) on real text; fit y ~ [u, 1] W per layer.
    tok, model = load_model()
    tr, te = load_split("train"), load_split("test")
    fit_ids, ev_ids = encode(tok, tr[:80]), encode(tok, te[:40])
    caps = {}

    def hook(li):
        def f(mod, inp, outp):
            caps.setdefault(li, []).append((inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu(), outp.detach().float().reshape(-1, outp.shape[-1]).cpu()))
        return f

    hs = [l.mlp.register_forward_hook(hook(i)) for i, l in enumerate(model.model.layers)]
    with torch.no_grad():
        for x, m in batches(fit_ids, 1):  # batch 1: no padding positions in the captured activations
            model.model(input_ids=x.to(DEV), attention_mask=m.long().to(DEV))
    fitcap = caps
    caps = {}
    with torch.no_grad():
        for x, m in batches(ev_ids, 1):
            model.model(input_ids=x.to(DEV), attention_mask=m.long().to(DEV))
    evcap = caps
    for h in hs:
        h.remove()
    Wfit, rows = {}, []
    for li in range(24):
        U = torch.cat([a for a, _ in fitcap[li]]); Y = torch.cat([b for _, b in fitcap[li]])
        Ue = torch.cat([a for a, _ in evcap[li]]); Ye = torch.cat([b for _, b in evcap[li]])
        Ua = torch.cat([U, torch.ones(len(U), 1)], 1).double()
        G = Ua.T @ Ua
        W = torch.linalg.solve(G + 1e-6 * G.diagonal().mean() * torch.eye(G.shape[0], dtype=G.dtype), Ua.T @ Y.double()).float()
        pe = torch.cat([Ue, torch.ones(len(Ue), 1)], 1) @ W
        rows.append({"layer": li, "tokens_fit": len(U), "tokens_eval": len(Ue),
                     "linear_fit_rel_residual_heldout": float((Ye - pe).norm() / Ye.norm()),
                     "linear_fit_rel_residual_fit": float((Y - torch.cat([U, torch.ones(len(U), 1)], 1) @ W).norm() / Y.norm())})
        Wfit[li] = W
        print(rows[-1], flush=True)
    del fitcap, evcap

    # NLL on held-out when one MLP at a time is replaced by its linear fit
    @torch.no_grad()
    def nll():
        tot, n = 0.0, 0
        for x, m in batches(ev_ids, 8):
            x, m = x.to(DEV), m.to(DEV)
            lg = model(input_ids=x, attention_mask=m.long()).logits[:, :-1].float()
            v = m[:, 1:] & m[:, :-1]
            tot += float(torch.nn.functional.cross_entropy(lg[v], x[:, 1:][v], reduction="sum")); n += int(v.sum())
        return tot / n
    base = nll()
    for li in range(24):
        W = Wfit[li].to(DEV)
        orig = model.model.layers[li].mlp.forward
        model.model.layers[li].mlp.forward = lambda u, W=W: (torch.cat([u.float(), torch.ones(*u.shape[:-1], 1, device=DEV)], -1) @ W).to(u.dtype)
        rows[li]["heldout_nll_with_this_mlp_linearized"] = nll()
        rows[li]["nll_increase"] = rows[li]["heldout_nll_with_this_mlp_linearized"] - base
        model.model.layers[li].mlp.forward = orig
        print(li, rows[li]["nll_increase"], flush=True)
    receipt("e3-nonlin", {"heldout_nll_full_model": base, "per_layer": rows,
                          "fit": "GSM8K train[0:80] all positions; eval GSM8K test[0:40]; least squares with bias, ridge 1e-6 * mean diag"})
