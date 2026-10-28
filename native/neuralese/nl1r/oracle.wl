(* Independent exact-rational derivative and native-carrier oracle; no product runtime. *)
soft[x_] := x/(1+Abs[x]);
fromBits[n_Integer] := Module[{sgn,e,m},sgn=If[BitAnd[BitShiftRight[n,63],1]==1,-1,1];e=BitAnd[BitShiftRight[n,52],2047];m=BitAnd[n,2^52-1];If[e==0,sgn m 2^-1074,sgn (2^52+m) 2^(e-1075)]];
lossF[q_List,d_,non_] := Module[{a,h,u,r,z,o},a={1,0,1,0}.Partition[Take[q,4d],d]+q[[4d+1;;5d]];h=soft/@a;If[non,u=h.Partition[q[[16;;39]],8]+q[[40;;47]];r=Max[0,#]&/@u;z=r.Partition[q[[48;;79]],4]+q[[80;;83]],z=h.Partition[q[[5d+1;;9d]],4]+q[[9d+1;;9d+4]]];o=(1+soft[#])/2&/@z;Total[(o-{1,0,1,0})^2]];
check[rows_,d_,non_] := Module[{n,p0,a,h,u,r,z,o,dz,dh,du,da,g,fd,ng,nu},
n=If[non,83,9d+4];p0=Range[n]/128;
a={1,0,1,0}.Partition[Take[p0,4d],d]+p0[[4d+1;;5d]];h=soft/@a;
If[non,u=h.Partition[p0[[16;;39]],8]+p0[[40;;47]];r=Max[0,#]&/@u;z=r.Partition[p0[[48;;79]],4]+p0[[80;;83]],z=h.Partition[p0[[5d+1;;9d]],4]+p0[[9d+1;;9d+4]]];
o=(1+soft[#])/2&/@z;dz=(o-{1,0,1,0})/(1+Abs[z])^2;
If[non,du=(Partition[p0[[48;;79]],4].dz)*(Boole[#>0]&/@u);dh=Partition[p0[[16;;39]],8].du,dh=Partition[p0[[5d+1;;9d]],4].dz];
da=dh/(1+Abs[a])^2;
g=If[non,Join[Flatten[Outer[Times,{1,0,1,0},da]],da,Flatten[Outer[Times,h,du]],du,Flatten[Outer[Times,r,dz]],dz],Join[Flatten[Outer[Times,{1,0,1,0},da]],da,Flatten[Outer[Times,h,dz]],dz]];
fd=Table[(lossF[ReplacePart[p0,i->p0[[i]]+1/1000000],d,non]-lossF[ReplacePart[p0,i->p0[[i]]-1/1000000],d,non])/(2/1000000),{i,n}];
ng=fromBits/@rows[[All,4]];nu=fromBits/@rows[[All,5]];
<|"parameters"->n,"gradientMaxError"->N[Max[Abs[ng-g]],18],"updateMaxError"->N[Max[Abs[nu-(p0-g/10)]],18],"finiteDifferenceMaxError"->N[Max[Abs[fd-g]],18],"PASS"->TrueQ[Max[Abs[ng-g]]<10^-12&&Max[Abs[nu-(p0-g/10)]]<10^-12&&Max[Abs[fd-g]]<10^-8],"signMutantRejected"->TrueQ[Max[Abs[-g-g]]>10^-8],"indexMutantRejected"->TrueQ[Max[Abs[RotateLeft[g]-g]]>10^-8]|>];
