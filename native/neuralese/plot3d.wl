(* Read-only Wolfram plotting oracle. __DATA__ is literal raw CSV payload JSON, not executable product code. *)
rows=Map[ToExpression,ImportString[__INPUT__,"RawJSON"],{2}];
fromBits[b_]:=Module[{s,e,m},s=If[BitShiftRight[b,63]==0,1,-1];e=BitAnd[BitShiftRight[b,52],2047];m=BitAnd[b,2^52-1];If[e==0,s*m*2^-1074,s*(2^52+m)*2^(e-1075)]];
coords=(N[fromBits/@#[[7;;9]]]&/@rows);
project[p_]:={240+100*(p[[1]]-p[[2]])*Sqrt[3]/2,160+50*(p[[1]]+p[[2]])-100*p[[3]]};
num[x_]:=ToString[N[x],InputForm];
xy[p_]:=StringRiffle[num/@project[p],","];
origin=project[{0,0,0}];
axes=StringJoin[MapThread[Function[{p,label},With[{q=project[p]},"<line x1=\""<>num[origin[[1]]]<>"\" y1=\""<>num[origin[[2]]]<>"\" x2=\""<>num[q[[1]]]<>"\" y2=\""<>num[q[[2]]]<>"\" stroke=\"black\"/><text x=\""<>num[q[[1]]]<>"\" y=\""<>num[q[[2]]]<>"\">"<>label<>"</text>"]],{{{1,0,0},{0,1,0},{0,0,1}},{"h0","h1","h2"}}]];
points=StringJoin[MapThread[Function[{p,r},With[{q=project[p],color=If[EvenQ[Total[IntegerDigits[r[[4]],2,4]]],"#2468a2","#b35718"]},"<circle cx=\""<>num[q[[1]]]<>"\" cy=\""<>num[q[[2]]]<>"\" r=\"3\" fill=\""<>color<>"\"/><text x=\""<>num[q[[1]]+4]<>"\" y=\""<>num[q[[2]]-4]<>"\">"<>ToString[r[[4]]]<>"</text>"]],{coords,rows}]];
svg="<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 480 320\"><rect width=\"480\" height=\"320\" fill=\"white\"/><g font-size=\"12\"><text x=\"12\" y=\"20\">Learned 3D messages: seed17 parity split</text><text x=\"12\" y=\"302\">ID labels; blue=train, brown=heldout; isometric view</text>"<>axes<>points<>"</g></svg>";
csv=ExportString[Prepend[MapThread[Join[{#2[[4]],If[EvenQ[Total[IntegerDigits[#2[[4]],2,4]]],"train","heldout"]},#1]&,{coords,rows}],{"input_id","partition","h0","h1","h2"}],"CSV"];
ExportString[<|"status"->"PASS","sourceCommit"->"c5992ca","seed"->17,"split"->1,"arm"->0,"mode"->0,"context"->1,"points"->Length[coords],"projection"->"isometric; no separation or density claim","svg"->svg,"csv"->csv|>,"RawJSON","Compact"->True]
