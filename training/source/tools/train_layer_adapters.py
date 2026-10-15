#!/usr/bin/env python3
"""Second research route: real intermediate-layer w2v-BERT adapter training.

The 600M encoder stays frozen. Adapter outputs from layers 8/12/16/20/24,
attentive pooling and the 256D speaker projection receive gradients. Saved
checkpoints are strict, small patches against the hash-bound native base, not
final embedding-space linear maps.
"""
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
from tools.relation_distillation import (RECORD,read_cache,TEACHER,CONFIG,write_json,normalize,paired_trials)
from tools.train_redimnet2_aam_domain_expert import aam_logits,augment_channel,paired_info_nce
from tools.evaluate_distillation_gates import compute_eer_percent
from native_w2vbert2 import NativeFeatures,load_native_teacher
from multidomain_kd_inference import sha256_file

LAYERS=(8,12,16,20,24)
PREFIXES=tuple('adapter_layers.'+str(i)+'.' for i in LAYERS)+('pooling.','bottleneck.')


def load_patch(model,path):
    patch=torch.load(path,map_location='cpu',weights_only=True)
    assert patch['base_sha256']==sha256_file(TEACHER)
    state=model.state_dict()
    allowed={k for k in state if k.startswith(PREFIXES)}
    assert set(patch['state_dict'])==allowed,'adapter patch keys mismatch'
    for k,v in patch['state_dict'].items():
        assert v.shape==state[k].shape and torch.isfinite(v).all()
    state.update(patch['state_dict']);model.load_state_dict(state,strict=True)


def train(a):
    rows,cache=read_cache();waves=cache['waves'];targets=cache['teacher']
    rng=random.Random(a.seed);torch.manual_seed(a.seed);np.random.seed(a.seed)
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for i,r in enumerate(rows):
        if r['split']=='train':groups[r['domain']][r['speaker']].append(i)
    keys=[(d,s) for d in sorted(groups) for s in sorted(groups[d])];lookup={k:i for i,k in enumerate(keys)}
    model=load_native_teacher(TEACHER,CONFIG).to(a.device).eval()
    params=[]
    for name,param in model.named_parameters():
        if name.startswith(PREFIXES):param.requires_grad_(True);params.append(param)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(PREFIXES)}
    classifier=torch.nn.Parameter(torch.from_numpy(np.stack([normalize(targets[groups[d][s]].mean(0)) for d,s in keys])).to(a.device))
    optimizer=torch.optim.AdamW([{'params':params,'lr':1e-5},{'params':[classifier],'lr':5e-4}],weight_decay=1e-4)
    suffix = ('_' + a.tag) if a.tag else ''
    assert not a.tag or all(c.isalnum() or c=='_' for c in a.tag)
    features=NativeFeatures();out=RECORD/('w2v_layer_adapters_s'+str(a.seed)+suffix)
    if (out/'adapter_patch.pt').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);started=time.time();trace=[]
    for step in range(1,a.steps+1):
        left,right,labels=[],[],[]
        for d in sorted(groups):
            for s in rng.sample(sorted(groups[d]),min(2,len(groups[d]))):
                i,j=rng.sample(groups[d][s],2)
                assert rows[i]['event']!=rows[j]['event']
                left.append(i);right.append(j);labels.append(lookup[(d,s)])
        length=(8000,16000,32000)[(step-1)%3];short=[]
        for j in right:
            start=rng.randrange(48000-length+1)
            short.append(augment_channel(waves[j,start:start+length],rng,.25))
        sa=model(features(waves[left]).to(a.device));sb=model(features(short).to(a.device))
        label=torch.tensor(labels,device=a.device);reference=torch.from_numpy(targets[left]).to(a.device)
        margin=.12*min(1.,step/30.)
        classification=.5*(F.cross_entropy(aam_logits(sa,classifier,label,margin,24),label)+F.cross_entropy(aam_logits(sb,classifier,label,margin,24),label))
        preserve=(1-F.cosine_similarity(sa,reference)).mean()
        loss=classification+.25*paired_info_nce(sa,sb,.1)+.1*preserve
        assert torch.isfinite(loss),'nonfinite adapter loss'
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step()
        if step==1 or step%20==0 or step==a.steps:
            row=dict(step=step,loss=float(loss.detach().cpu()),classification=float(classification.detach().cpu()),preserve=float(preserve.detach().cpu()),seconds=time.time()-started)
            trace.append(row);write_json(out/'PROGRESS.json',dict(trace=trace,arguments=vars(a)))
            print(json.dumps(dict(run=out.name,**row)),flush=True)
    patch={k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(PREFIXES)}
    changed=[k for k in patch if not torch.equal(patch[k],initial[k])]
    assert any(k.startswith('adapter_layers.12.') for k in changed),'intermediate adapter did not learn'
    torch.save(dict(schema_version=1,base_sha256=sha256_file(TEACHER),state_dict=patch),out/'adapter_patch.pt')
    write_json(out/'TRAINING_RECEIPT.json',dict(arguments=vars(a),trace=trace,changed_tensors=changed,
        trainable_parameters=sum(p.numel() for p in params),frozen_encoder=True,layer_ids=LAYERS,
        checkpoint_sha256=sha256_file(out/'adapter_patch.pt'),manifest_sha256=sha256_file(RECORD/'DATA_MANIFEST.json'),
        release_allowed=False,hidden_score_guarantee=False,seconds=time.time()-started))
    print('ADAPTER_TRAIN_COMPLETE '+out.name,flush=True)


def evaluate(a):
    rows,cache=read_cache();indices=[i for i,r in enumerate(rows) if r['split']==a.split]
    rows=[rows[i] for i in indices];waves=cache['waves'][indices]
    model=load_native_teacher(TEACHER,CONFIG).to(a.device).eval()
    name='w2v_original'
    if a.patch:load_patch(model,a.patch);name=Path(a.patch).parent.name
    features=NativeFeatures();values={};started=time.time()
    with torch.inference_mode():
        for duration,n in [('long3',48000),('short05',8000),('short10',16000)]:
            if not a.patch and duration=='long3':
                # Exact same frozen teacher/crop already computed and hash-bound.
                values[duration]=cache['teacher'][indices].copy()
                continue
            output=[]
            for offset in range(0,len(rows),4):
                x=waves[offset:offset+4,(48000-n)//2:(48000+n)//2]
                output.extend(model(features(x).to(a.device)).cpu().numpy())
            values[duration]=np.asarray(output,np.float32)
            assert np.isfinite(values[duration]).all()
            print(json.dumps(dict(phase='adapter_eval',name=name,duration=duration,seconds=time.time()-started)),flush=True)
    out=RECORD/'evaluations'/a.split/name;out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'embeddings.npz',ids=np.asarray([r['id'] for r in rows]),**values)
    cells=[]
    for d in sorted({r['domain'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['domain']==d];p=paired_trials([rows[i] for i in ix])
        for duration,v in values.items():
            scores=np.sum(values['long3'][ix][p[:,0]]*v[ix][p[:,1]],axis=1)
            cells.append(dict(domain=d,duration=duration,eer=compute_eer_percent(p[:,2],scores)))
    write_json(out/'METRICS.json',dict(name=name,cells=cells,release_allowed=False))
    print(json.dumps(dict(name=name,cells=cells)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('train','evaluate'));p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=5101);p.add_argument('--steps',type=int,default=120);p.add_argument('--patch');p.add_argument('--split',default='dev');p.add_argument('--tag',default='')
    a=p.parse_args();torch.set_num_threads(4)
    if a.command=='train':train(a)
    else:evaluate(a)


if __name__=='__main__':main()
