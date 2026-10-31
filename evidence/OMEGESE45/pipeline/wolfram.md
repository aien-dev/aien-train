# Wolfram queries used by docs/omegese45/PIPELINE.md

All of these were evaluated with the mounted WolframLanguageEvaluator on 2026-10-29. The inputs come
from the receipts in this directory. Numeric results below are rounded for readability.

## 1. Train outcome distribution (seed 4501, N 20000)

See [`wolfram-distribution.txt`](wolfram-distribution.txt) for the complete query and unmodified
output, evaluating both N=20000 and N=1000. The train values were:

- adm = 0.778688;
- pj = 0.6325;
- expected counts:
  - oob 4426.24;
  - okep 8040.39;
  - retry 3074.84;
  - stop 7533.37;
  - attempts 23573.81;
  - transport 3536.07;
  - other, per pair, 471.476.

Against the observed counts (`data/gen-4501-20000.stdout`), the z-scores (obs − ex)/√ex were:

| Counter | z |
|---|---|
| oob 4461 | 0.5225 |
| okep 8059 | 0.2075 |
| retry 3042 | −0.5923 |
| stop 7480 | −0.6149 |
| transport 3495 | −0.6907 |

## 2. The 15 other outcome pairs

```
others = {481,456,538,456,459,482,473,472,448,474,430,474,459,493,432}; e = 471.4761168; x2 = Total[(others - e)^2/e]; {Total[others], N[x2, 5], N[1 - CDF[ChiSquareDistribution[14], Total[(others - Mean[others])^2/Mean[others]]], 4], N[1 - CDF[ChiSquareDistribution[15], x2], 4]}
-> {7027, 20.628, 0.1160, 0.1491}
```

## 3. Dev outcome distribution (seed 4502, N 1000)

See the N=1000 row of `wolfram-distribution.txt`. The expected counts are oob 221.312, okep 402.020,
retry 153.742 and stop 376.668. Against observed counts 210, 401, 158 and 389, the z-scores are
−0.7604, −0.0509, 0.3434 and 0.6354.

## 4. Dev and timing figures

```
{N[43793/44233, 6], N[1564/44233, 6], N[2116/2410, 6], N[159/2410, 6], N[148/1200, 4], N[139/401, 4], N[7/210, 4], N[1000*1429.84/400000, 5], N[(23*60 + 52.05)/60, 5], bits2f[1178283946], bits2f[1178697656], N[bits2f[1178283946]/874210, 6], bits2f[1006632960], 2^-7}
```

with `bits2f[b_] := First[ImportByteArray[ByteArray[Reverse[IntegerDigits[b, 256, 4]]], "Real32"]]`, returned:

```
{0.990053, 0.0353582, 0.878008, 0.0659751, 0.1233, 0.3466, 0.03333, 3.5746, 23.8675, 11979.916015625, 12383.9296875, 0.0137037, 0.0078125, 1/128}
```

## 5. Loss and sizes

```
{bits2f[1221734527], N[bits2f[1221734527]/1496027, 6], bits2f[1198743734], N[bits2f[1198743734]/874210, 6], N[bits2f[1178283946]/874210, 6], 3854*256 + 256 + 256*41 + 41, 4*997417 + 240, (30 + 64)*41, N[1 - 92^3/100^3, 6]}
-> {430467.96875, 0.287741, 62304.7109375, 0.0712697, 0.0137037, 997417, 3989908, 3854, 0.221312}
```
