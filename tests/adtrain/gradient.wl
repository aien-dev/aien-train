(* SPDX-License-Identifier: AGPL-3.0-only *)
(* adtrain exact-rational gradient oracle (Wolfram Language; verification only).
   Inputs, prepended by the caller from one run of the suite:
     gbits  = {58 integers}  bit patterns of the `G <i> <bits>` rows (run/G.stdout)
     kref   = {10 integers}  the `K <i> <k>` rows (run/D.stdout)
     l0bits = integer        the `L 0 <bits>` row (run/T.stdout)
   Regenerates the seed-17 data and initial parameters exactly, fixes the relu
   pattern (x > 0, the tape's rule) from the exact forward, differentiates the
   exact loss symbolically, and compares every tape value against the declared
   bound |om - exact| <= 2^-16 Max[1, |exact|]. *)
mask64 = 2^64 - 1;
xs[x_] := Module[{y = x},
  y = BitXor[y, BitAnd[BitShiftLeft[y, 13], mask64]];
  y = BitXor[y, BitShiftRight[y, 7]];
  BitXor[y, BitAnd[BitShiftLeft[y, 17], mask64]]];
ks = Mod[Rest[NestList[xs, 17, 96]], 17] - 8;
xv = Partition[ks[[1 ;; 32]]/8, 4];
tv = Partition[ks[[33 ;; 48]]/8, 2];
w1v = Partition[ks[[49 ;; 80]]/16, 8];
w2v = Partition[ks[[81 ;; 96]]/16, 2];
w1s = Array[a, {4, 8}]; b1s = Array[b, 8]; w2s = Array[c, {8, 2}]; b2s = Array[d, 2];
params = Join[Flatten[w1s], b1s, Flatten[w2s], b2s];
point = Thread[params -> Join[Flatten[w1v], ConstantArray[0, 8], Flatten[w2v], ConstantArray[0, 2]]];
pre = xv . w1s + ConstantArray[b1s, 8];
preV = pre /. point;
relu = Map[Boole[# > 0] &, preV, {2}];
y = (relu*pre) . w2s + ConstantArray[b2s, 8];
loss = Total[(y - tv)^2, 2];
gex = D[loss, {params}] /. point;
lex = loss /. point;
f32[u_] := Module[{s = BitShiftRight[u, 31], e = BitAnd[BitShiftRight[u, 23], 255],
    m = BitAnd[u, 2^23 - 1]},
  (-1)^s If[e == 0, m 2^-149, (2^23 + m) 2^(e - 150)]];
gom = f32 /@ gbits;
bound = 2^-16 Max[1, Abs[#]] & /@ gex;
diff = Abs[gom - gex];
lom = f32[l0bits];
<|"kMatch" -> (ks[[1 ;; 10]] === kref),
  "zeroPreActivations" -> Count[Flatten[preV], 0],
  "reluActive" -> Total[Flatten[relu]],
  "n" -> Length[gex],
  "allWithin" -> And @@ Thread[diff <= bound],
  "exactEqual" -> Count[diff, 0],
  "maxDiffOverBound" -> N[Max[diff/bound]],
  "maxAbsDiff" -> N[Max[diff]],
  "lossExact" -> lex, "lossOm" -> lom,
  "lossWithin" -> (Abs[lom - lex] <= 2^-16 Max[1, Abs[lex]]),
  "lossRelErr" -> N[Abs[lom - lex]/Abs[lex]]|>
