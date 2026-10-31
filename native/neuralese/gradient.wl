(* Independent mathematical verification of bounded ecosystem6 extension. *)
(* Symbolic D/Simplify over 31 parameters timed out (HTTP 504): symbolic path NOT_RUN.
   Direct exact rational explicit derivative plus central difference is used instead. *)
soft[x_] := x/(1+Abs[x]);
x = {1,0,1,0}; t = {1,0,1,0};
lossF[q_List] := Module[{w,b,v,c,a,h,z,o},
 w = Partition[Take[q,12],3]; b = q[[13;;15]]; v = Partition[q[[16;;27]],4]; c = q[[28;;31]];
 a = x.w+b; h = soft/@a; z = h.v+c; o = (1+soft[#])/2&/@z; Total[(o-t)^2]];
p0 = Table[(-1)^i i/64,{i,1,31}];
w = Partition[Take[p0,12],3]; b = p0[[13;;15]]; v = Partition[p0[[16;;27]],4]; c = p0[[28;;31]];
a = x.w+b; h = soft/@a; z = h.v+c; o = (1+soft[#])/2&/@z;
dz = (o-t)/(1+Abs[z])^2;
dh = v.dz;
da = dh/(1+Abs[a])^2;
grad = Join[Flatten[Outer[Times,x,da]],da,Flatten[Outer[Times,h,dz]],dz];
eps = 1/1000000;
fd = Table[(lossF[ReplacePart[p0,i->p0[[i]]+eps]]-lossF[ReplacePart[p0,i->p0[[i]]-eps]])/(2 eps),{i,31}];
err = Max[Abs[fd-grad]];
ExportString[<|"status"->If[Length[grad]==31&&err<1/100000000,"PASS","FAIL"],"profile"->"MODELED exact rational softsign network; explicit derivative plus central difference; symbolic D/Simplify NOT_RUN (504 timeout)","parameters"->31,"input"->x,"target"->t,"initial"->N[p0,17],"output"->N[o,17],"loss"->N[lossF[p0],17],"gradient"->N[grad,17],"updated"->N[p0-grad/10,17],"manualDerivativeExact"->"NOT_RUN","centralDifferenceMaxError"->N[err,17],"signMutantRejected"->TrueQ[Max[Abs[-grad-grad]]>1/100000000],"indexMutantRejected"->TrueQ[Max[Abs[RotateLeft[grad]-grad]]>1/100000000],"updateMutantRejected"->TrueQ[Max[Abs[(p0+grad/10)-(p0-grad/10)]]>1/100000000]|>,"RawJSON"]
