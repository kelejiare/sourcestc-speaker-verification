#!/usr/bin/env python3
"""Expand training speakers, then freeze a distinct confirmation subset."""
import argparse
import collections
import json
import sys
import wave
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.relation_distillation import (RECORD as PREVIOUS,TEACHER,CONFIG,
    resolve,event_id,stable,write_json,sha256_file,read_cache)
from tools.train_redimnet2_aam_domain_expert import MANIFESTS,read_rows
from audio_features import load_audio
RECORD=ROOT/'records/2026-09-13/126_lora_lora_duration_experts'


def prepare():
    path=RECORD/'DATA_MANIFEST.json'
    if path.exists():raise FileExistsError(path)
    prior=json.loads((PREVIOUS/'DATA_MANIFEST.json').read_text())['rows']
    old_child={r['speaker'] for r in prior if r['split']=='train' and r['domain']=='child'}
    grouped=collections.defaultdict(list)
    for domain in ('child','youth','meeting'):
        for r in read_rows(MANIFESTS[domain]):
            if r['split']!='train':continue
            p=resolve(r,domain)
            if not p.is_file():continue
            with wave.open(str(p)) as w:seconds=w.getnframes()/w.getframerate()
            if seconds<3:continue
            grouped[(domain,r['speaker_id'])].append(dict(path=str(p),domain=domain,speaker=r['speaker_id'],
                event=event_id(r,domain,p),gender=str(r.get('gender','unknown')).lower(),
                source_seconds=seconds,id=stable(str(p)),sha256=sha256_file(p)))
    unique={}
    for key,items in grouped.items():
        events={}
        for row in sorted(items,key=lambda r:stable(r['path'])):events.setdefault(row['event'],row)
        if len(events)>=4:unique[key]=list(events.values())
    eligible=sorted([s for d,s in unique if d=='child' and s not in old_child],key=lambda s:stable(bytes.fromhex("7635322d686f6c646f75747c").decode("utf-8")+s))
    holdout=set(eligible[:24]);assert len(holdout)==24
    rows=[]
    for (domain,speaker),items in sorted(unique.items()):
        split='confirm_child' if domain=='child' and speaker in holdout else 'train'
        for row in items[:4 if split=='confirm_child' else 6]:rows.append(dict(row,split=split))
    # relation dev/test stays fixed; old test is diagnostic only, not new selection.
    rows.extend(r for r in prior if r['split'] in ('dev','test'))
    speaker_splits=collections.defaultdict(set)
    for r in rows:speaker_splits[(r['domain'],r['speaker'])].add(r['split'])
    assert all(len(s)==1 for s in speaker_splits.values())
    assert len({r['sha256'] for r in rows})==len(rows)
    assert not holdout.intersection(old_child)
    rows.sort(key=lambda r:(r['split'],r['domain'],r['speaker'],r['id']))
    counts={}
    for split,domain in sorted({(r['split'],r['domain']) for r in rows}):
        selected=[r for r in rows if r['split']==split and r['domain']==domain]
        counts[split+'/'+domain]=dict(files=len(selected),speakers=len({r['speaker'] for r in selected}))
    write_json(path,dict(rows=rows,counts=counts,confirmation_speakers=sorted(holdout),
        old_relation_train_speakers_retained=True,historical_pre_relation_reuse_possible=True,
        previous_manifest_sha256=sha256_file(PREVIOUS/'DATA_MANIFEST.json')))
    print(json.dumps(counts),flush=True)


def cache(device):
    from native_w2vbert2 import NativeFeatures,load_native_teacher
    path=RECORD/'acoustic_cache.npz'
    if path.exists():raise FileExistsError(path)
    rows=json.loads((RECORD/'DATA_MANIFEST.json').read_text())['rows']
    oldrows,old=read_cache();lookup={r['id']:i for i,r in enumerate(oldrows)}
    # NPZ arrays are decompressed on every indexing access. Materialize once,
    # not one full old array retained behind each row view (RAM explosion).
    old_waves=old['waves'];old_targets=old['teacher']
    waves=[];targets=np.zeros((len(rows),256),np.float32);missing=[]
    for index,row in enumerate(rows):
        if row['id'] in lookup:
            j=lookup[row['id']];assert oldrows[j]['sha256']==row['sha256']
            waves.append(old_waves[j].copy());targets[index]=old_targets[j]
        else:
            assert sha256_file(row['path'])==row['sha256']
            x=load_audio(row['path'],16000,remove_dc=False);start=(len(x)-48000)//2
            waves.append(x[start:start+48000]);missing.append(index)
    waves=np.stack(waves).astype(np.float32,copy=False)
    del old_waves,old_targets
    old.close()
    model=load_native_teacher(TEACHER,CONFIG).to(device).eval();features=NativeFeatures()
    with torch.inference_mode():
        for offset in range(0,len(missing),4):
            ix=missing[offset:offset+4];targets[ix]=model(features(waves[ix]).to(device)).cpu().numpy()
            if offset%120==0:print(json.dumps(dict(phase='new_teacher_cache',files=offset+len(ix),total=len(missing))),flush=True)
    assert np.isfinite(targets).all() and np.max(np.abs(np.linalg.norm(targets,axis=1)-1))<1e-4
    np.savez(path,waves=waves,teacher=targets,manifest_sha256=np.asarray(sha256_file(RECORD/'DATA_MANIFEST.json')))
    write_json(RECORD/'CACHE_RECEIPT.json',dict(cache_sha256=sha256_file(path),
        manifest_sha256=sha256_file(RECORD/'DATA_MANIFEST.json'),teacher_sha256=sha256_file(TEACHER),
        new_teacher_files=len(missing),reused_files=len(rows)-len(missing),device=device))
    print('LORA_CACHE_COMPLETE',flush=True)


def load_cache():
    manifest=RECORD/'DATA_MANIFEST.json';path=RECORD/'acoustic_cache.npz'
    receipt=json.loads((RECORD/'CACHE_RECEIPT.json').read_text())
    assert sha256_file(path)==receipt['cache_sha256'] and sha256_file(manifest)==receipt['manifest_sha256']
    data=np.load(path,allow_pickle=False)
    assert str(data['manifest_sha256'])==sha256_file(manifest)
    return json.loads(manifest.read_text())['rows'],data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','cache'));parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.command=='prepare':prepare()
    else:cache(args.device)
