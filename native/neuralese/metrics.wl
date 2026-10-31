(* Read-only Wolfram oracle. The data placeholder is replaced by untouched CSV fields as a JSON string. *)
data = Map[ToExpression,ImportString[__DATA__,"RawJSON"],{2}];
fromBits[n_Integer] := Module[{sgn,e,m},sgn=If[BitAnd[BitShiftRight[n,63],1]==1,-1,1]; e=BitAnd[BitShiftRight[n,52],2047]; m=BitAnd[n,2^52-1]; If[e==0,sgn m 2^-1074,sgn (2^52+m) 2^(e-1075)]];
truth[r_] := Reverse[IntegerDigits[r[[4]],2,4]];
probs[r_] := N[fromBits /@ r[[10;;13]],18];
pred[r_] := r[[14;;17]];
correct[r_] := r[[21]]==0 && pred[r]===truth[r];
part[r_] := If[r[[2]]==0 || EvenQ[Total[truth[r]]],"train","heldout"];
key[r_] := Append[r[[{1,2,3,5,6}]],part[r]];
groups=GatherBy[data,key];
summary[g_] := Module[{ps,ys,pairs,ece,bins,accepted,messages},
 ps=probs/@g; ys=truth/@g;
 pairs=Flatten[Map[Function[r,Transpose[{(Max[#,1-#]&/@probs[r]),MapThread[Boole[#1==#2]&,{pred[r],truth[r]}]}]],g],1];
 bins=Table[Select[pairs,(i/5<=#[[1]] && If[i==4,#[[1]]<=1,#[[1]]<(i+1)/5])&],{i,0,4}];
 ece=Total[(If[Length[#]==0,0,Length[#]/Length[pairs] Abs[Mean[#[[All,1]]]-Mean[#[[All,2]]]]]&)/@bins];
 accepted=Select[g,#[[18]]==0 && #[[21]]==0&]; messages=g[[All,7;;9]];
 <|"key"->key[First[g]],"n"->Length[g],"exact"->Count[correct/@g,True],"featureMSE"->Mean[Flatten[(ps-ys)^2]],"featureECE5"->ece,"abstained"->Count[g[[All,18]],1],"accepted"->Length[accepted],"acceptedExact"->Count[correct/@accepted,True],"collisionPairs"->Total[Binomial[#,2]&/@Values[Counts[messages]]],"distinctMessages"->Length[DeleteDuplicates[messages]],"encoderMedianNs"->Median[g[[All,19]]],"decoderMedianNs"->Median[g[[All,20]]],"encoderP95Ns"->Quantile[g[[All,19]],19/20],"decoderP95Ns"->Quantile[g[[All,20]],19/20]|>];
ExportString[<|"status"->"PASS","scope"->"EXERCISED synthetic aggregation, not campaign qualification","columns"->Keys[summary[First[groups]]],"metrics"->(Values[summary[#]]&/@groups)|>,"RawJSON","Compact"->True]
