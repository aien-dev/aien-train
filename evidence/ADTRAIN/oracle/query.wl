gbits = {1012531200,3200499712,1038315520,1057377920,3192803328,1044633088,1072657216,1059866112,3210915840,1051499520,3214650560,3215344960,3179126784,3206203264,3214120640,3221337216,3191130112,1026220032,3196256768,3213141312,3167682560,3209066880,3211721344,3216259840,3208305408,3193594880,3211001984,3221852992,3206473216,3220845952,3205072128,3224286976,1066432512,1050025984,1068422656,1077000192,3213832192,1074026496,3190120448,1082103808,3210985600,3208587392,3212141952,3213871040,3214739904,3190289408,3222396032,1060899584,3205229568,1056598016,3222220288,1052957696,3223806656,1039037440,3227447552,1067773184,3239437312,1054441472};
kref = {5,7,-3,5,-8,2,0,-4,7,0};
l0bits = 1090364420;
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
