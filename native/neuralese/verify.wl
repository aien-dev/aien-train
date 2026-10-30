(* The inserted exact-gradient oracle is independent, not a host trainer. *)
__GRAD_SOURCE__;
fromBits[n_Integer] := Module[{sgn,e,m},sgn=If[BitAnd[BitShiftRight[n,63],1]==1,-1,1]; e=BitAnd[BitShiftRight[n,52],2047]; m=BitAnd[n,2^52-1]; If[e==0,sgn m 2^-1074,sgn (2^52+m) 2^(e-1075)]];
gdata=Map[ToExpression,ImportString[__GDATA__,"RawJSON"],{2}]; fdata=Map[ToExpression,ImportString[__FDATA__,"RawJSON"],{2}]; pdata=Map[ToExpression,ImportString[__PDATA__,"RawJSON"],{2}];
ngrad=fromBits/@gdata[[All,3]]; nupdate=fromBits/@gdata[[All,4]];
gerr=Max[Abs[ngrad-grad]]; uerr=Max[Abs[nupdate-(p0-grad/10)]];
hw=Nest[KroneckerProduct[{{1,1},{1,-1}},#]&,{{1}},3]; he=hw[[All,{2,3,5,8}]]/Sqrt[8];
ferr[r_] := Module[{xx,hh,expected},xx=Reverse[IntegerDigits[r[[1]],2,4]]; hh=xx.Transpose[he]; If[r[[2]]!=0,hh=Round[r[[2]] hh]/r[[2]]]; expected=hh.he; Abs[fromBits[r[[4]]]-expected[[r[[3]]+1]]]];
fe=Max[ferr/@fdata]; predictions=And@@Map[(#[[5]]==Reverse[IntegerDigits[#[[1]],2,4]][[#[[3]]+1]])&,fdata];
s={{1,1,1,0},{1,0,0,1},{0,1,1,1}}; sh=s.Transpose[he];
qcurve=Table[{q,N[Max[Abs[Flatten[(Round[q sh]/q).he-s]]]]},{q,{1,2,4,8,16,32,64}}];
coords=Map[(fromBits/@#[[7;;9]])&,pdata];
num[x_] := ToString[N[x],InputForm];
qpoints=StringRiffle[(num[40+65 Log[2,#[[1]]]]<>","<>num[240-240 #[[2]]])&/@qcurve," "];
qplot="<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"480\" height=\"290\" viewBox=\"0 0 480 290\"><title>Exact Hadamard fixture quantization error</title><rect width=\"480\" height=\"290\" fill=\"white\"/><path d=\"M40 20V240H450\" fill=\"none\" stroke=\"black\"/><polyline fill=\"none\" stroke=\"#2456ad\" points=\""<>qpoints<>"\"/><text x=\"40\" y=\"270\">denominator q: log2 scale, 1 to 64</text><text x=\"48\" y=\"25\">max feature error (y=0 to 1)</text></svg>";
pdots=StringJoin[MapThread[("<circle cx=\""<>num[240+160 #1[[1]]]<> "\" cy=\""<>num[140-100 #1[[2]]]<> "\" r=\"4\" fill=\"hsl("<>num[180 (#1[[3]]+1)]<>",70%,40%)\"/><text x=\""<>num[246+160 #1[[1]]]<> "\" y=\""<>num[140-100 #1[[2]]]<> "\" font-size=\"10\">"<>ToString[#2]<>"</text>")&,{coords,pdata[[All,4]]}]];
pplot="<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"480\" height=\"290\" viewBox=\"0 0 480 290\"><title>Actual learned three-coordinate messages; seed17 parity fit; xy projection, hue coordinate3</title><rect width=\"480\" height=\"290\" fill=\"white\"/><path d=\"M80 140H410M240 40V240\" fill=\"none\" stroke=\"#888\"/>"<>pdots<>"<text x=\"20\" y=\"265\">message coordinates 1/2; hue=coordinate 3; labels=input IDs</text><text x=\"20\" y=\"20\">Actual native learned messages: seed 17, parity fit</text></svg>";
ExportString[<|"status"->If[gerr<10^-12&&uerr<10^-12&&fe<10^-12&&predictions,"PASS","FAIL"],"scope"->"EXERCISED independent comparison of native binary64 carriers with exact Wolfram oracle","gradientCount"->Length[gdata],"gradientMaxError"->N[gerr,18],"updateMaxError"->N[uerr,18],"fixedFeatureCount"->Length[fdata],"fixedMaxError"->N[fe,18],"fixedPredictionsCorrect"->predictions,"quantizationPlotSVG"->qplot,"learnedMessagePlotSVG"->pplot|>,"RawJSON"]
