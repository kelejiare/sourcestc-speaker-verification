#!/usr/bin/env python3
"""Evaluate a protocol patch with actual calibrated_fusion output geometry and corrected gates."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.audit_protocol import pairs
from tools.evaluate_distillation_gates import compute_eer_percent
from tools.relation_distillation import sha256_file
from tools.evaluate import data
from tools.evaluate_teacher_gates import DURATIONS, extract_patch

RECORD=ROOT/"records/2026-09-16/155_protocol_protocol_hard_negative"
LORA=ROOT/"records/2026-09-13/126_lora_lora_duration_experts"
PARENT=LORA/"lora_duration_s5201/merged_patch.pt"
SOURCE1={"long3":.45,"short05":.32,"short10":.25}
# Reaching 92.5 from the online 91.71295 anchor requires a 0.78705-point
# reduction in the 15-condition macro EER.  This candidate changes source one
# only (approximately 9/15 conditions), so the source-one proxy must improve by
# at least 0.78705 * 15 / 9 = 1.31175 points.  Use 1.30 as the predeclared gate
# after rounding, rather than silently accepting a statistically tiny change.
TARGET_SOURCE1_MACRO_DELTA = -1.30


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp");tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n");tmp.replace(path)


def compose(anchor,teacher,trial,duration):
    left,right,labels=trial.T
    a=np.sum(anchor["long3"][left].astype(np.float64)*anchor[duration][right],axis=1)
    t=np.sum(teacher["long3"][left].astype(np.float64)*teacher[duration][right],axis=1)
    score=np.sqrt((1-SOURCE1["long3"])*(1-SOURCE1[duration]))*a+np.sqrt(SOURCE1["long3"]*SOURCE1[duration])*t
    return labels,score


def localize(rows,arrays,indices):
    return [rows[i] for i in indices],{k:v[indices] for k,v in arrays.items()}


def bootstrap_correct(rows,trial,before,after,base_weight,repeats=400,seed=7041):
    names=sorted({str(row["speaker"]) for row in rows})
    sid=np.asarray([names.index(str(row["speaker"])) for row in rows])
    left,right,labels=trial.T;sl,sr=sid[left],sid[right]
    rng=np.random.RandomState(seed);values=[]
    for _ in range(repeats):
        count=np.bincount(rng.randint(len(names),size=len(names)),minlength=len(names))
        cluster=np.where(sl==sr,count[sl],count[sl]*count[sr]).astype(np.float64)
        weight=cluster*base_weight
        if weight[labels==0].sum()>0 and weight[labels==1].sum()>0:
            values.append(float(np.mean([compute_eer_percent(labels,n,weight)-compute_eer_percent(labels,o,weight)
                                         for o,n in zip(before,after)])))
    if len(values)<int(.95*repeats):raise RuntimeError("too few valid bootstrap replicates")
    return {"mean_delta":float(np.mean(values)),"ci95":np.quantile(values,[.025,.975]).tolist(),
            "valid_repeats":len(values)}


def protocol_result(rows,anchor,parent,candidate,enroll_domains,test_domains,name,weighted=None):
    trial=pairs(rows,set(enroll_domains),set(test_domains)); left,right,labels=trial.T
    weight=np.ones(len(trial),np.float64)
    if weighted:
        test_domain=np.asarray([rows[i]["domain"] for i in right])
        for domain,mass in weighted.items():
            mask=test_domain==domain
            if mask.any(): weight[mask]=float(mass)/int(mask.sum())
    cells=[];before=[];after=[]
    for duration,_ in DURATIONS:
        y,old=compose(anchor,parent,trial,duration);_,new=compose(anchor,candidate,trial,duration)
        old_e=compute_eer_percent(y,old,weight);new_e=compute_eer_percent(y,new,weight)
        cells.append({"duration":duration,"calibrated_fusion_eer":old_e,"candidate_eer":new_e,"delta":new_e-old_e})
        before.append(old);after.append(new)
    used=sorted(set(left.tolist()+right.tolist())); mapping={old:new for new,old in enumerate(used)}
    local_trial=np.asarray([(mapping[e],mapping[t],y) for e,t,y in trial],np.int64)
    boot=bootstrap_correct([rows[i] for i in used],local_trial,before,after,weight,repeats=400,seed=7041)
    return {"name":name,"trials":int(len(trial)),"positive":int(labels.sum()),"cells":cells,
            "macro_delta":float(np.mean([x["delta"] for x in cells])),
            "speaker_cluster_bootstrap":boot}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--patch",required=True);parser.add_argument("--tag",required=True)
    parser.add_argument("--candidate-cache")
    parser.add_argument("--candidate-durations",default="long3,short05,short10")
    parser.add_argument("--device",default="cpu");args=parser.parse_args()
    patch=Path(args.patch).resolve()
    if not patch.is_file():raise FileNotFoundError(patch)
    output=RECORD/"evaluations"/args.tag/"GATE_REPORT.json"
    if output.exists():raise FileExistsError(output)
    datasets={}; all_rows=[]; waves=[]; offsets={}; cursor=0
    parent_parts={key:[] for key,_ in DURATIONS}
    for name in ("dev","confirm_child"):
        rows,value=data(name);datasets[name]=rows;all_rows.extend(rows);waves.append(value)
        cached=np.load(LORA/"evaluations"/name/"lora_duration_s5201/embeddings.npz",allow_pickle=False)
        if list(cached["ids"])!=[r["id"] for r in rows]:raise RuntimeError("parent cache order mismatch")
        for key,_ in DURATIONS:parent_parts[key].append(cached[key])
        offsets[name]=np.arange(cursor,cursor+len(rows));cursor+=len(rows)
    waves=np.concatenate(waves,axis=0)
    parent={key:np.concatenate(value,axis=0) for key,value in parent_parts.items()}
    allowed={value.strip() for value in args.candidate_durations.split(",") if value.strip()}
    known={key for key,_ in DURATIONS}
    if not allowed or not allowed.issubset(known):
        raise ValueError("invalid candidate duration set: " + repr(sorted(allowed)))
    if args.candidate_cache:
        cached=dict(np.load(Path(args.candidate_cache).resolve(),allow_pickle=False))
        if list(cached["ids"]) != [r["id"] for r in all_rows]:
            raise RuntimeError("candidate cache order mismatch")
        candidate={key:np.asarray(cached[key]) for key,_ in DURATIONS}
    else:
        candidate=extract_patch(patch,waves,args.device)
    for key in known-allowed:
        candidate[key]=parent[key].copy()
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.parent/"candidate_embeddings.npz",ids=np.asarray([r["id"] for r in all_rows]),**candidate)
    protocols=[]
    for dataset in ("dev","confirm_child"):
        rows=datasets[dataset];ix=offsets[dataset]
        old={k:v[ix] for k,v in parent.items()};new={k:v[ix] for k,v in candidate.items()}
        anchor_file=LORA/"evaluations"/dataset/"plus_online_anchor/embeddings.npz"
        a=dict(np.load(anchor_file,allow_pickle=False))
        if list(a["ids"])!=[r["id"] for r in rows]:raise RuntimeError("anchor order mismatch")
        if dataset=="confirm_child":
            protocols.append(protocol_result(rows,a,old,new,{"child"},{"child"},"confirm_child_within"))
            continue
        for domain in ("child","youth"):
            indices=np.asarray([i for i,r in enumerate(rows) if r["domain"]==domain])
            rr,aa=localize(rows,a,indices);_,oo=localize(rows,old,indices);_,nn=localize(rows,new,indices)
            protocols.append(protocol_result(rr,aa,oo,nn,{domain},{domain},"dev_"+domain+"_within"))
        indices=np.asarray([i for i,r in enumerate(rows) if r["domain"] in ("child","youth")])
        rr,aa=localize(rows,a,indices);_,oo=localize(rows,old,indices);_,nn=localize(rows,new,indices)
        protocols.append(protocol_result(rr,aa,oo,nn,{"child"},{"child","youth"},
                                         "dev_child_enroll_child_youth_7_6",{"child":7,"youth":6}))
    macro=float(np.mean([p["macro_delta"] for p in protocols]))
    regressions=[{"protocol":p["name"],**c} for p in protocols for c in p["cells"] if c["delta"]>.2]
    ci_ok=all(p["speaker_cluster_bootstrap"]["ci95"][1]<0 for p in protocols)
    cross=next(p for p in protocols if p["name"].endswith("7_6"))
    authorized=bool(macro<=TARGET_SOURCE1_MACRO_DELTA and not regressions and ci_ok and cross["macro_delta"]<0)
    result={"schema":"protocol_corrected_protocol_gate_v1","candidate_patch":str(patch),
            "candidate_sha256":sha256_file(patch),"parent_sha256":sha256_file(PARENT),
            "candidate_durations":sorted(allowed),
            "protocols":protocols,"reliable_macro_delta":macro,"regressions_gt_0p2":regressions,
            "gate":{"target_macro_delta_at_most":TARGET_SOURCE1_MACRO_DELTA,"macro_pass":macro<=TARGET_SOURCE1_MACRO_DELTA,
                    "no_cell_regression_gt_0p2":not regressions,"all_protocol_ci_upper_below_zero":ci_ok,
                    "cross_age_protocol_improves":cross["macro_delta"]<0,"authorized_for_packaging":authorized},
            "limitations":["Proxy corpora do not reproduce hidden sources","Youth female coverage is insufficient",
                           "Repeated crops are not new speakers","Passing is necessary, not a hidden-score guarantee"],
            "hidden_score_guarantee":False}
    write_json(output,result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":torch.set_num_threads(4);main()
