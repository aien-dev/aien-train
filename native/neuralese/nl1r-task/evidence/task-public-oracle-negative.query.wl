
Clear[bits,target,active,metric];
bits[id_]:=Table[BitAnd[BitShiftRight[id,j],1],{j,0,3}];
target[t_,id_]:=If[t==1,{Mod[Total[bits[id]],2],0,0,0},bits[id]];
active[t_]:=Switch[t,0,{1,2,3,4},1,{1},2,{3,4}];
rat[a_,b_]:=If[b==0,Null,ToString[a/b,InputForm]];
metric[rr_,t_,arm_,v_,split_]:=Module[{n,exact,accepted,ac,wa,ab,ae,refusals,payload,en,de,ids,pred,want,er,ok},
n=Length[rr];ids=rr[[All,1]];pred=rr[[All,5;;8]];want=target[t,#]&/@ids;
exact=MapThread[Boole[#1===#2]&,{pred,want}];accepted=(Boole[#[[9]]==0&&#[[12]]==0]&/@rr);
ac=Total[exact*accepted];wa=Total[(1-exact)*accepted];ab=Count[rr[[All,9]],1];
ae=Total[MapThread[Count[MapThread[Unequal,{#1[[active[t]]],#2[[active[t]]]}],True]&,{pred,want}]];
refusals=Count[rr[[All,12]],Except[0]];payload=Total[rr[[All,4]]];en=Total[rr[[All,10]]];de=Total[rr[[All,11]]];
<|"task"->t,"arm"->arm,"variant"->v,"subset"->split,"ids"->ids,"n"->n,"exact_ignoring_abstention"->Total[exact],"exact_rate"->rat[Total[exact],n],"accepted"->Total[accepted],"accepted_correct"->ac,"wrong_accepted"->wa,"abstentions"->ab,"decoder_refusals"->refusals,"active_feature_errors"->ae,"active_feature_comparisons"->n Length[active[t]],"active_feature_error_rate"->rat[ae,n Length[active[t]]],"forced_decision_01_regret_sum"->n-Total[exact],"forced_decision_01_regret_mean"->rat[n-Total[exact],n],"accepted_decision_01_regret_sum"->wa,"accepted_decision_01_regret_mean"->rat[wa,Total[accepted]],"abstention_regret"->Null,"actual_payload_bytes"->payload,"encode_ns_sum"->en,"decode_ns_sum"->de,"encode_ns_min"->Min[rr[[All,10]]],"encode_ns_max"->Max[rr[[All,10]]],"decode_ns_min"->Min[rr[[All,11]]],"decode_ns_max"->Max[rr[[All,11]]]|>
];

r={{0,0,0,1,0,0,0,0,0,320,832,0,0,0}};
a=metric[r,0,0,0,"control"];bad=ReplacePart[r,{1,5}->1];b=metric[bad,0,0,0,"control"];
ExportString[<|"positive_exact"->a["exact_ignoring_abstention"],"positive_wrong_accepted"->a["wrong_accepted"],"mutant_exact"->b["exact_ignoring_abstention"],"mutant_wrong_accepted"->b["wrong_accepted"],"positive_pass"->(a["exact_ignoring_abstention"]==1&&a["wrong_accepted"]==0),"negative_rejected"->!(b["exact_ignoring_abstention"]==1&&b["wrong_accepted"]==0)|>,"RawJSON","Compact"->True]

