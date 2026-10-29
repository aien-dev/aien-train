(* Verification oracle only. Actual native G records are appended by the recorded query. *)
ss[x_] := x/(1+Abs[x]);
fromBits[n_Integer] := Module[{s,e,m},s=If[BitAnd[BitShiftRight[n,63],1]==1,-1,1];e=BitAnd[BitShiftRight[n,52],2047];m=BitAnd[n,2^52-1];If[e==0,s*m*2^-1074,s*(2^52+m)*2^(e-1075)]];
active[t_] := {{1,1,1,1},{1,0,0,0},{0,0,1,1}}[[t+1]];
target[t_] := If[t==1,{0,0,0,0},{1,0,1,0}];
loss[p_,t_] := Module[{h,z,o},h=ss/@({1,0,1,0}.Partition[p[[1;;16]],4]+p[[17;;20]]);z=h.Partition[p[[21;;36]],4]+p[[37;;40]];o=(1+ss[#])/2&/@z;Total[active[t]*(o-target[t])^2]];
check[rows_,t_] := Module[{p,a,h,z,o,dz,da,g,fd,ng,nu},p=Range[40]/128;a={1,0,1,0}.Partition[p[[1;;16]],4]+p[[17;;20]];h=ss/@a;z=h.Partition[p[[21;;36]],4]+p[[37;;40]];o=(1+ss[#])/2&/@z;dz=active[t]*(o-target[t])/(1+Abs[z])^2;da=(Partition[p[[21;;36]],4].dz)/(1+Abs[a])^2;g=Join[Flatten[Outer[Times,{1,0,1,0},da]],da,Flatten[Outer[Times,h,dz]],dz];fd=Table[(loss[ReplacePart[p,i->p[[i]]+1/1000000],t]-loss[ReplacePart[p,i->p[[i]]-1/1000000],t])/(2/1000000),{i,40}];ng=fromBits/@rows[[All,4]];nu=fromBits/@rows[[All,5]];<|"task"->t,"gradientMaxError"->N[Max[Abs[ng-g]],18],"updateMaxError"->N[Max[Abs[nu-(p-g/10)]],18],"finiteDifferenceMaxError"->N[Max[Abs[fd-g]],18],"PASS"->TrueQ[Max[Abs[ng-g]]<10^-12&&Max[Abs[nu-(p-g/10)]]<10^-12&&Max[Abs[fd-g]]<10^-8]|>];
