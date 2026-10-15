#!/usr/bin/env python3
"""Frozen protocols, independent-embedding fusion, speaker-level uncertainty."""
import argparse
import collections
import json
import sys
import tempfile
import time
import wave
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.prepare_data import RECORD,PREVIOUS,load_cache,TEACHER,CONFIG,write_json,sha256_file,stable
from tools.relation_distillation import paired_trials
from tools.cache_online_anchor import build_anchor
from tools.evaluate_distillation_gates import compute_eer_percent
from native_w2vbert2 import NativeFeatures,load_native_teacher
from lora_duration_model import apply_merged_patch
from audio_features import load_audio
DURATIONS=(('long3',48000),('short05',8000),('short10',16000))


def prepare_far():
    folder=RECORD/'confirm_nearfar';path=folder/'MANIFEST.json'
    if path.exists():raise FileExistsError(path)
    root=ROOT/'data/distillation_sources/alimeeting/prepared_eval/audio/test/eval'
    groups=collections.defaultdict(list)
    for role in ('near','far'):
        for p in sorted((root/role).glob('*/*.wav')):
            with wave.open(str(p)) as w:seconds=w.getnframes()/w.getframerate()
            if seconds<3:continue
            origin=p.stem.split('__far_ch')[0].split('__near')[0]
            groups[p.parent.name].append(dict(path=str(p),speaker=p.parent.name,event=origin,
                id=stable(str(p)),sha256=sha256_file(p),domain='meeting',kind=role))
    old=json.loads((PREVIOUS/'external_alimeeting/MANIFEST.json').read_text())['rows']
    old_speakers={r['speaker_stable_id'].split(':')[-1] for r in old}
    rows=[]
    for speaker,items in sorted(groups.items()):
        assert speaker not in old_speakers,'AliMeeting eval/test speaker overlap'
        near=sorted([r for r in items if r['kind']=='near'],key=lambda r:stable(r['id']))
        if not near:continue
        enroll=near[0];tests={}
        for r in sorted(items,key=lambda r:stable(r['id'])):
            if r['kind']=='far' and r['event']!=enroll['event']:tests.setdefault(r['event'],r)
        if not tests:continue
        rows.append(dict(enroll,role='enroll'));rows.extend(dict(r,role='test') for r in list(tests.values())[:2])
    assert len({r['speaker'] for r in rows})>=8
    waves=[]
    for r in rows:
        x=load_audio(r['path'],16000,False);start=(len(x)-48000)//2;waves.append(x[start:start+48000])
    folder.mkdir(parents=True,exist_ok=True)
    write_json(path,dict(rows=rows,protocol='AliMeeting eval, near enrollment vs different-event far tests',
        speakers=len({r['speaker'] for r in rows}),disjoint_from_relation_40_speakers=True,
        historical_pre_relation_reuse_possible=True))
    np.savez(folder/'waves.npz',waves=np.stack(waves),manifest_sha256=np.asarray(sha256_file(path)))
    print(json.dumps(dict(files=len(rows),speakers=len({r['speaker'] for r in rows}))))


def data(dataset):
    if dataset=='nearfar_dev':
        folder=PREVIOUS/'external_alimeeting';m=json.loads((folder/'MANIFEST.json').read_text())
        rows=[dict(r,speaker=r['speaker_stable_id'],event=r['origin_stable_id'],domain='meeting',id=r['utterance_stable_id']) for r in m['rows']]
        n=np.load(folder/'waves.npz',allow_pickle=False)
        assert str(n['manifest_sha256'])==sha256_file(folder/'MANIFEST.json')
        return rows,n['waves']
    if dataset=='confirm_nearfar':
        folder=RECORD/dataset;rows=json.loads((folder/'MANIFEST.json').read_text())['rows']
        n=np.load(folder/'waves.npz',allow_pickle=False)
        assert str(n['manifest_sha256'])==sha256_file(folder/'MANIFEST.json')
        return rows,n['waves']
    rows,cache=load_cache();ix=[i for i,r in enumerate(rows) if r['split']==dataset]
    return [rows[i] for i in ix],cache['waves'][ix]


def pairs_for(rows):
    if 'role' not in rows[0]:return paired_trials(rows)
    pairs=[]
    for t,r in enumerate(rows):
        if r['role']!='test':continue
        for e,s in enumerate(rows):
            if s['role']!='enroll':continue
            assert r['event']!=s['event']
            pairs.append((e,t,int(r['speaker']==s['speaker'])))
    return np.asarray(pairs,np.int64)


def evaluate(a):
    rows,waves=data(a.dataset)
    name=Path(a.patch).parent.name if a.patch else 'plus_online_anchor'
    dataset_key=a.dataset+('' if a.crop=='center' else '_'+a.crop)
    out=RECORD/'evaluations'/dataset_key/name
    if (out/'embeddings.npz').exists():raise FileExistsError(out)
    # Reuse exactly hash/order-checked, completed plus development vectors.
    if not a.patch and a.dataset in ('dev','nearfar_dev') and a.crop=='center':
        old=PREVIOUS/('evaluations/dev/plus_online_anchor/embeddings.npz' if a.dataset=='dev' else 'external_alimeeting/plus_online_anchor/embeddings.npz')
        values=np.load(old,allow_pickle=False)
        if 'ids' in values:assert list(values['ids'])==[r['id'] for r in rows]
        out.mkdir(parents=True,exist_ok=True)
        np.savez(out/'embeddings.npz',ids=np.asarray([r['id'] for r in rows]),**{k:values[k] for k,n in DURATIONS})
        write_json(out/'RECEIPT.json',dict(reused_from=str(old),sha256=sha256_file(old)))
        print('REUSED_EXACT_ANCHOR '+a.dataset,flush=True);return
    if a.patch:
        m=load_native_teacher(TEACHER,CONFIG)
        apply_merged_patch(m,a.patch,sha256_file(a.patch));m.to(a.device).eval();features=NativeFeatures()
    else:m=build_anchor(a.device)
    values={};started=time.time()
    with tempfile.TemporaryDirectory(prefix='game823-lora-eval-') as tmp,torch.inference_mode():
        for duration,n in DURATIONS:
            start=0 if a.crop=='start' else 48000-n if a.crop=='end' else (48000-n)//2
            output=[]
            if a.patch:
                for offset in range(0,len(rows),4):
                    x=waves[offset:offset+4,start:start+n]
                    output.extend(m(features(x).to(a.device)).cpu().numpy())
            else:
                for i,row in enumerate(rows):
                    x=waves[i,start:start+n]
                    path=Path(tmp)/('data_scenery2' if row['domain']=='meeting' else 'data_scenery1')/'audio'/('sample_'+str(i)+'.wav')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    pcm=np.clip(np.round(x.astype(np.float64)*2147483648),-2147483648,2147483647).astype('<i4')
                    with wave.open(str(path),'wb') as w:
                        w.setnchannels(1);w.setsampwidth(4);w.setframerate(16000);w.writeframes(pcm.tobytes())
                    assert np.max(np.abs(load_audio(path,16000,False)-x))<1e-8
                    vec,meta=m.extract_file(path)
                    assert meta.get('fallback_mode') in ('none','',None)
                    assert meta.get('auxiliary_mode')!='file_safe_bypass'
                    output.append(vec)
            values[duration]=np.asarray(output,np.float32)
            assert np.isfinite(values[duration]).all() and np.max(np.abs(np.linalg.norm(values[duration],axis=1)-1))<1e-4
            print(json.dumps(dict(dataset=a.dataset,name=name,duration=duration,seconds=time.time()-started)),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'embeddings.npz',ids=np.asarray([r['id'] for r in rows]),**values)
    write_json(out/'RECEIPT.json',dict(patch_sha256=sha256_file(a.patch) if a.patch else None,
        device=a.device,seconds=time.time()-started,embedding_sha256=sha256_file(out/'embeddings.npz')))


def compare(a):
    dataset_key=a.dataset+('' if a.crop=='center' else '_'+a.crop)
    rows,_=data(a.dataset);folder=RECORD/'evaluations'/dataset_key
    anchor=np.load(folder/'plus_online_anchor/embeddings.npz',allow_pickle=False)
    assert list(anchor['ids'])==[r['id'] for r in rows]
    result=[]
    for path in sorted(folder.glob('lora_*/embeddings.npz')):
        new=np.load(path,allow_pickle=False);assert list(new['ids'])==list(anchor['ids'])
        cells=[];intervals={}
        for domain in sorted({r['domain'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['domain']==domain];subset=[rows[i] for i in ix];pairs=pairs_for(subset)
            scorepairs=[]
            for duration,n in DURATIONS:
                old=np.sum(anchor['long3'][ix][pairs[:,0]]*anchor[duration][ix][pairs[:,1]],axis=1)
                candidate=np.sum(new['long3'][ix][pairs[:,0]]*new[duration][ix][pairs[:,1]],axis=1)
                fused=.75*old+.25*candidate;labels=pairs[:,2]
                old_eer=compute_eer_percent(labels,old);new_eer=compute_eer_percent(labels,fused)
                cells.append(dict(domain=domain,duration=duration,anchor_eer=old_eer,new_eer=new_eer,delta=new_eer-old_eer))
                scorepairs.append((old,fused))
            speakers=sorted({r['speaker'] for r in subset});sid=np.asarray([speakers.index(r['speaker']) for r in subset])
            if len(speakers)<10:
                intervals[domain]=dict(speakers=len(speakers),ci95=None,
                    reason='too_few_speakers_for_stable_cluster_bootstrap')
                continue
            e,t=sid[pairs[:,0]],sid[pairs[:,1]];rng=np.random.RandomState(2026091352);boot=[]
            for _ in range(300):
                counts=np.bincount(rng.randint(len(speakers),size=len(speakers)),minlength=len(speakers))
                weights=np.where(e==t,counts[e],counts[e]*counts[t]).astype(float)
                if weights[labels==0].sum()==0 or weights[labels==1].sum()==0:continue
                boot.append(float(np.mean([compute_eer_percent(labels,nw,weights)-compute_eer_percent(labels,od,weights) for od,nw in scorepairs])))
            intervals[domain]=dict(speakers=len(speakers),ci95=np.quantile(boot,[.025,.975]).tolist(),valid_replicates=len(boot))
        reliable=[r for r in cells if intervals[r['domain']]['speakers']>=10]
        result.append(dict(name=path.parent.name,alpha=.25,cells=cells,bootstrap=intervals,
            macro_delta=float(np.mean([r['delta'] for r in cells])),
            reliable_macro_delta=float(np.mean([r['delta'] for r in reliable])),
            release_allowed=False,hidden_score_guarantee=False))
    write_json(RECORD/('COMPARISON_'+dataset_key+'.json'),dict(candidates=result,selection_or_confirmation=dataset_key))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare_far','evaluate','compare'))
    p.add_argument('--dataset',choices=('dev','nearfar_dev','confirm_child','confirm_nearfar'),default='dev')
    p.add_argument('--patch');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--crop',choices=('center','start','end'),default='center')
    a=p.parse_args();torch.set_num_threads(a.threads)
    if a.command=='prepare_far':prepare_far()
    elif a.command=='evaluate':evaluate(a)
    else:compare(a)
