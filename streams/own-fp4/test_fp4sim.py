# test_fp4sim.py (L284): host checks of the oracle quantizers. ~/.venv/bin/python -I test_fp4sim.py
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, fp4sim as Q
y = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.9, 7.0, -0.25, -2.5, -9.0])
want = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 0.0, -2.0, -6.0])
assert torch.equal(Q.round_e2m1(y), want), Q.round_e2m1(y)
g = torch.Generator().manual_seed(0); v = torch.full((200000,), 2.6)
m = Q.round_e2m1(v, True, g).mean().item(); assert abs(m - 2.6) < 0.01, m          # SR unbiased between 2 and 3
x = torch.randn(64, 256) * torch.exp(torch.randn(64, 1))
for dim in (0, 1):
    qs, gg = Q.quant_nv(x, dim); e = Q.rel_err(qs * gg, x); assert 0.05 < e < 0.2, e
    qs, gg = Q.quant_mx(x, dim); e2 = Q.rel_err(qs * gg, x); assert 0.05 < e2 < 0.3, e2
w = torch.randn(64, 128); qs, gg = Q.quant_nv2d(w); assert Q.rel_err(qs * gg, w) < 0.2
# bf16-exactness of q * blockscale (the claim the GEMM simulation rests on)
qs, gg = Q.quant_nv(x, 1); assert torch.equal(qs.bfloat16().float(), qs)
qs, gg = Q.quant_nv2d(w); assert torch.equal(qs.bfloat16().float(), qs)
H = Q.hadamard(16, "cpu"); assert torch.allclose(H @ H.T, torch.eye(16), atol=1e-6)
t, a = Q.ternary(w); assert set((t / t.abs().max()).unique().tolist()) <= {-1.0, 0.0, 1.0}
print("fp4sim host checks PASS")
