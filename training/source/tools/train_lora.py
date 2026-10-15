#!/usr/bin/env python3
"""Predeclared matched-budget full-vector vs duration-prefix LoRA pilots."""
import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_data import RECORD,load_cache,TEACHER,CONFIG,write_json,sha256_file
from tools.train_layer_adapters import load_patch
from tools.train_redimnet2_aam_domain_expert import aam_logits,augment_channel,paired_info_nce
from native_w2vbert2 import NativeFeatures,load_native_teacher
from lora_duration_model import install_lora,export_merged_patch,TARGETS

PARENT=ROOT/'records/2026-09-13/125_relation_chinese_relation_distillation/w2v_layer_adapters_s5101/adapter_patch.pt'
PARENT_SHA='def352ff89821fc16719250745a3660b74aeb16b988db4ae032ceeb0c8cc914b'


def speech_window(x,n,rng):
    starts=[rng.randrange(len(x)-n+1) for _ in range(5)]
    reference=float(np.mean(x.astype(np.float64)**2))
    windows=[x[s:s+n] for s in starts]
    energies=[float(np.mean(w.astype(np.float64)**2)) for w in windows]
    ix=next((i for i,e in enumerate(energies) if e>=max(reference*.25,1e-9)),int(np.argmax(energies)))
    return windows[ix]


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('full','duration'),required=True)
    p.add_argument('--seed',type=int,default=5201);p.add_argument('--steps',type=int,default=180);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(4)
    out=RECORD/('lora_'+a.mode+'_s'+str(a.seed))
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    rows,cache=load_cache();waves=cache['waves'];targets=cache['teacher']
    assert sha256_file(PARENT)==PARENT_SHA
    rng=random.Random(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for i,r in enumerate(rows):
        if r['split']=='train':groups[r['domain']][r['speaker']].append(i)
    keys=[(d,s) for d in sorted(groups) for s in sorted(groups[d])];lookup={k:i for i,k in enumerate(keys)}
    model=load_native_teacher(TEACHER,CONFIG);load_patch(model,PARENT)
    head,lora=install_lora(model);model.to(a.device).eval()
    # Param objects survive Module.to, verified below for the active runtime.
    assert all(p.device.type==a.device for p in head+lora)
    weights=np.stack([targets[groups[d][s]].mean(0) for d,s in keys])
    weights/=np.maximum(np.linalg.norm(weights,axis=1,keepdims=True),1e-8)
    classifier=torch.nn.Parameter(torch.from_numpy(weights).to(a.device))
    optimizer=torch.optim.AdamW([{'params':head,'lr':1e-5},{'params':lora,'lr':1e-4},
                               {'params':[classifier],'lr':5e-4}],weight_decay=1e-4)
    features=NativeFeatures();trace=[];started=time.time();seen=collections.Counter()
    for step in range(1,a.steps+1):
        left,right,labels=[],[],[]
        for domain,count in [('child',4),('youth',2),('meeting',2)]:
            for speaker in rng.sample(sorted(groups[domain]),count):
                i,j=rng.sample(groups[domain][speaker],2)
                assert rows[i]['event']!=rows[j]['event']
                left.append(i);right.append(j);labels.append(lookup[(domain,speaker)]);seen[domain+':'+speaker]+=1
        duration=(8000,16000,32000)[(step-1)%3]
        short=[augment_channel(speech_window(waves[j],duration,rng),rng,.25) for j in right]
        sa=model(features(waves[left]).to(a.device));sb=model(features(short).to(a.device))
        label=torch.tensor(labels,device=a.device);reference=torch.from_numpy(targets[left]).to(a.device)
        prefix={8000:64,16000:128,32000:192}[duration] if a.mode=='duration' else 256
        margin=({8000:0.,16000:.05,32000:.10}[duration] if a.mode=='duration' else .12)*min(1.,step/30.)
        cls_long=F.cross_entropy(aam_logits(sa,classifier,label,.12*min(1.,step/30.),24),label)
        # Both vectors and shared classifier prefix are normalized in aam_logits.
        cls_short=F.cross_entropy(aam_logits(sb[:,:prefix],classifier[:,:prefix],label,margin,24),label)
        classification=.5*(cls_long+cls_short)
        preserve=(1-F.cosine_similarity(sa,reference)).mean()
        relation=paired_info_nce(sa,sb,.1)
        loss=classification+.25*relation+.1*preserve
        assert torch.isfinite(loss),'nonfinite training loss'
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(head+lora+[classifier],5.)
        assert torch.isfinite(grad),'nonfinite training gradient'
        optimizer.step()
        if step==1 or step%20==0 or step==a.steps:
            row=dict(step=step,loss=float(loss.detach().cpu()),classification=float(classification.detach().cpu()),
                     preserve=float(preserve.detach().cpu()),relation=float(relation.detach().cpu()),
                     gradient_norm=float(grad.detach().cpu()),seconds=time.time()-started)
            trace.append(row);write_json(out/'PROGRESS.json',dict(arguments=vars(a),trace=trace))
            print(json.dumps(dict(run=out.name,**row)),flush=True)
    patch=export_merged_patch(model,PARENT_SHA)
    changes={name:float(torch.linalg.norm(model.get_submodule(name).scale*(model.get_submodule(name).lora_b @ model.get_submodule(name).lora_a)).detach().cpu()) for name in TARGETS}
    assert all(v>0 for v in changes.values()),'LoRA did not change every intended projection'
    # Merged native graph output must equal the trained wrapper output.
    with torch.inference_mode():
        x=features(waves[left[:2],:16000]).to(a.device);before=model(x).cpu().numpy()
        for name in TARGETS:
            parent,leaf=name.rsplit('.',1);wrapper=model.get_submodule(name)
            wrapper.base.weight.copy_(wrapper.merged_weight())
            setattr(model.get_submodule(parent),leaf,wrapper.base)
        after=model(x).cpu().numpy()
    error=float(np.max(np.abs(before-after)));assert error<1e-4,error
    torch.save(patch,out/'merged_patch.pt')
    write_json(out/'TRAINING_RECEIPT.json',dict(arguments=vars(a),trace=trace,
        trainable_head_parameters=sum(p.numel() for p in head),trainable_lora_parameters=sum(p.numel() for p in lora),
        observed_train_speakers=len(seen),unique_train_speakers=len(keys),lora_weight_delta_norms=changes,
        merged_equivalence_max_abs=error,checkpoint_sha256=sha256_file(out/'merged_patch.pt'),
        manifest_sha256=sha256_file(RECORD/'DATA_MANIFEST.json'),parent_sha256=PARENT_SHA,
        release_allowed=False,hidden_score_guarantee=False,seconds=time.time()-started))
    print('LORA_TRAIN_COMPLETE '+out.name,flush=True)


if __name__=='__main__':main()
