(* Independent Wolfram oracle, not an Omega product runtime. Synthetic data only. *)
s = {{1,1,1,0},{1,0,0,1},{0,1,1,1}};
w = Nest[KroneckerProduct[{{1,1},{1,-1}},#]&,{{1}},3];
e = w[[All,{2,3,5,8}]]/Sqrt[8];
h = s.Transpose[e];
hq = Round[16 h]/16;
err[a_,b_] := Max[Abs[Flatten[a-b]]];
bits[a_] := Map[Boole[#>=1/2]&,a,{2}];
exact[a_] := ToString[a,InputForm];
all = Tuples[{0,1},4];
noise = Table[{d, Max[Table[err[(hq+ConstantArray[v,3]).e,s],{v,Tuples[{-d,d},8]}]]},{d,{0,1/128,1/64,1/32,1/16}}];
perm = {2,1,3,4,5,6,7,8};
mutants = <|"wrongScale"->err[h.(2 e),s],"wrongSign"->err[h.(-e),s],"wrongColumns"->err[h.e[[All,{2,1,3,4}]],s]|>;
shrink = Table[ek=e[[1;;k,All]]; messages=all.Transpose[ek]; <|"coordinates"->k,"rank"->MatrixRank[ek],"distinctMessages"->Length[DeleteDuplicates[messages]],"maxProjectionError"->exact[err[messages.ek,all]],"exactReconstruction"->TrueQ[messages.ek==all]|>,{k,1,8}];
qcurve = Table[<|"denominator"->q,"maxErrorExact"->exact[err[(Round[q h]/q).e,s]],"maxError"->N[err[(Round[q h]/q).e,s],17],"thresholdCorrect"->Count[Flatten[bits[(Round[q h]/q).e]-s],0]|>,{q,{1,2,4,8,16,32,64}}];
result = <|"status"->"PASS","scope"->"MODELED exact synthetic fixed encoder, not learned communication", "dimensions"-><|"S"->Dimensions[s],"W"->Dimensions[w],"E"->Dimensions[e],"H"->Dimensions[h]|>,"ranks"-><|"S"->MatrixRank[s],"W"->MatrixRank[w],"E"->MatrixRank[e],"H"->MatrixRank[h]|>,"orthogonality"->TrueQ[Transpose[e].e==IdentityMatrix[4]],"reconstruction"->TrueQ[Simplify[h.e-s]==ConstantArray[0,{3,4}]],"HExact"->exact[h],"quantized"->hq,"decodedExact"->exact[Simplify[hq.e]],"maxErrorExact"->exact[err[hq.e,s]],"maxError"->N[err[hq.e,s],17],"thresholdCorrect"->Count[Flatten[bits[hq.e]-s],0],"noiseWorstCase"->Map[<|"epsilon"->exact[#[[1]]],"maxErrorExact"->exact[#[[2]]],"maxError"->N[#[[2]],17]|>&,noise],"basisMatched"->TrueQ[h[[All,perm]].e[[perm,All]]==s],"basisUnmatchedMaxError"->exact[err[h[[All,perm]].e,s]],"negativeControls"->Map[exact,mutants],"quantizationCurve"->qcurve,"shrinkingChannel"->shrink,"malformed"-><|"wrongShape"->"refuse before Dot","nonfinite"->"refuse before Dot","outsideDeclaredRange"->"refuse before Dot","executedNativeAdmission"->"NOT_RUN"|>|>;
ExportString[result,"RawJSON"]
