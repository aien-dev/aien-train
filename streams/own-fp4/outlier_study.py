# outlier_study.py (L284 own-fp4 Part 2b, ORACLE): on real Wgrad operands dumped from the BF16 baseline (X and dY of four
# layers at three points of training), how much error does each outlier treatment leave in the weight gradient
# dW = dY^T X when both operands are quantized to NVFP4 (1x16 along the token dimension, X nearest, dY stochastic)?
# Metric: relative Frobenius error of dW against the FP32 product (the GEMM output, not per-element error).
#   ~/.venv/bin/python -I outlier_study.py DUMP_DIR OUT.json
import glob, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4sim as Q

dev = "cuda" if torch.cuda.is_available() else "cpu"
gen = torch.Generator(device=dev); gen.manual_seed(5)
sgn = torch.where(torch.rand(16, generator=torch.Generator().manual_seed(11)) < 0.5, -1.0, 1.0).to(dev)
H16 = Q.hadamard(16, dev); R16 = H16 * sgn[None, :]
sgn128 = torch.where(torch.rand(128, generator=torch.Generator().manual_seed(12)) < 0.5, -1.0, 1.0).to(dev)
R128 = Q.hadamard(128, dev) * sgn128[None, :]

def diag(x, dy):  # exact per-token balancing with power-of-two factors (dW unchanged in exact arithmetic)
    s = (dy.norm(dim=1) / x.norm(dim=1).clamp(min=1e-30)).clamp(min=1e-30).sqrt()
    s = torch.exp2(torch.round(torch.log2(s)))
    return x * s[:, None], dy / s[:, None]

T = {
    "none":       lambda x, d: (x, d),
    "signs_only": lambda x, d: (x * sgn.repeat(x.shape[0] // 16)[:, None], d * sgn.repeat(d.shape[0] // 16)[:, None]),
    "h16_fixed":  lambda x, d: (Q.rotate_tokens(x, H16), Q.rotate_tokens(d, H16)),
    "rht16":      lambda x, d: (Q.rotate_tokens(x, R16), Q.rotate_tokens(d, R16)),
    "rht128":     lambda x, d: (Q.rotate_tokens(x, R128), Q.rotate_tokens(d, R128)),
    "diag_pow2":  diag,
    "diag+rht16": lambda x, d: T["rht16"](*diag(x, d)),
}
# cost per operand element on the chip (multiply-adds), for the table
COST = {"none": 0, "signs_only": 0, "h16_fixed": 16, "rht16": 16, "rht128": 128, "diag_pow2": 1, "diag+rht16": 17}

def run(dump_dir, out):
    rows = []
    for f in sorted(glob.glob(f"{dump_dir}/dump_step*.pt")):
        step = int(f.split("dump_step")[1].split(".")[0]); d = torch.load(f)
        for tag in sorted({k.rsplit(".", 1)[0] for k in d}):
            x = d[tag + ".x"].float().to(dev); dy = d[tag + ".dy"].float().to(dev)
            ref = dy.T @ x
            for name, fn in T.items():
                for fmt in ("nv", "mx"):
                    xr, dr = fn(x, dy)
                    if fmt == "nv": xq, g1 = Q.quant_nv(xr, 0); dq, g2 = Q.quant_nv(dr, 0, True, gen)
                    else: xq, g1 = Q.quant_mx(xr, 0); dq, g2 = Q.quant_mx(dr, 0, True, gen)
                    dw = (dq.T @ xq) * (g1 * g2)
                    rows.append(dict(step=step, layer=tag, transform=name, fmt=fmt, dw_rel_err=Q.rel_err(dw, ref),
                                     x_rel_err=Q.rel_err(xq * g1, xr), dy_rel_err=Q.rel_err(dq * g2, dr), cost_mac_per_elem=COST[name]))
    json.dump(rows, open(out, "w"), indent=0)
    # summary: mean dW error per transform and format over all layers and steps
    summ = {}
    for r in rows: summ.setdefault((r["fmt"], r["transform"]), []).append(r["dw_rel_err"])
    for (fmt, name), v in sorted(summ.items()):
        print(f"{fmt} {name:12s} mean dW rel err {sum(v)/len(v):.4f}  max {max(v):.4f}  n {len(v)}  cost {COST[name]} MAC/elem")

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
