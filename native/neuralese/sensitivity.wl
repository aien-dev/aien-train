(* Independent exploratory sensitivity aggregation; not the frozen primary endpoint. *)
data=Map[ToExpression,ImportString[__DATA__,"RawJSON"],{2}];
fromBits[n_Integer] := Module[{sgn,e,m},sgn=If[BitAnd[BitShiftRight[n,63],1]==1,-1,1]; e=BitAnd[BitShiftRight[n,52],2047]; m=BitAnd[n,2^52-1]; If[e==0,sgn m 2^-1074,sgn (2^52+m) 2^(e-1075)]];
truth[r_]:=Reverse[IntegerDigits[r[[4]],2,4]];
probs[r_]:=N[fromBits/@r[[10;;13]],18];
correct[r_]:=r[[21]]==0 && (Boole[#>=1/2]&/@probs[r])===truth[r];
part[r_]:=If[r[[2]]==0 || EvenQ[Total[truth[r]]],"train","heldout"];
mode[r_]:=If[5<=r[[5]]<=12,"eight-noise-corners",r[[5]]];
key[r_]:=Join[r[[1;;3]],{mode[r],part[r]}];
summary[g_]:=Module[{bycase,accepted,distinct},bycase=GatherBy[g,#[[4]]&]; accepted=Select[g,#[[21]]==0&]; distinct=Length[DeleteDuplicates[#[[All,7;;9]]]]&/@GatherBy[g,#[[5]]&];
 {key[First[g]],Length[bycase],Count[(And@@(correct/@#))&/@bycase,True],Count[(Or@@Map[Function[r,r[[18]]==1],#])&/@bycase,True],Count[(Or@@Map[Function[r,r[[21]]!=0],#])&/@bycase,True],If[Length[accepted]==0,Null,Max[Flatten[Abs[(probs[#]-truth[#])&/@accepted]]]],Min[distinct]}];
ExportString[<|"status"->"PASS","scope"->"EXERCISED exploratory sensitivity; eight fixed sign corners, not random corruption distribution","columns"->{"key","cases","worstCornerExact","casesAnyAbstain","casesAnyRefusal","maxFeatureErrorAmongAcceptedFrames","minimumDistinctMessagesPerCorner"},"metrics"->(summary/@GatherBy[data,key])|>,"RawJSON","Compact"->True]
