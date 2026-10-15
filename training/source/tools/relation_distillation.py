#!/usr/bin/env python3
"""Audited Chinese long/short acoustic KD laboratory. Never auto-submits.

Training changes ReDimNet2 stage5/pooling, not a post-hoc output matrix.
All evaluation pairs use different speech events. Existing public corpora are
development proxies, not a pristine or official 15-scenario hidden test.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_features import load_audio
from multidomain_kd_inference import _load_redim, sha256_file
from tools.train_redimnet2_aam_domain_expert import (
    MANIFESTS, configure_student, aam_logits, augment_channel, paired_info_nce,
)
from tools.evaluate_distillation_gates import compute_eer_percent

RECORD = ROOT / 'records/2026-09-13/125_relation_chinese_relation_distillation'
SOURCE = ROOT / 'model/redimnet2-b6-vb2-vox2-cnc2-lm.pt'
TEACHER = ROOT / 'model/w2vbert2-multidomain-speaker-fp16.pt'
CONFIG = ROOT / 'records/2026-09-12/120_teacher_w2vbert2_teacher_screen/base_config/config.json'
SEEDS = (5101, 5102, 5103)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def stable(value):
    return hashlib.sha256(('relation-frozen-20260913|' + value).encode()).hexdigest()


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def resolve(row, domain):
    path = Path(row['path'])
    if path.is_file():
        return path
    if domain == 'child':
        suffix = str(path).split('/extracted/' + row['split'] + '/')[-1]
        roots = {'train': Path('/private/tmp/game823-child-train/train'),
                 'dev': ROOT / 'data/distillation_sources/childmandarin/holdout_dev_2026-09-04',
                 'test': ROOT / 'data/distillation_sources/childmandarin/teacher_test_confirmation/test'}
        return roots[row['split']] / suffix
    return path


def event_id(row, domain, path):
    if domain == 'meeting':
        return re.sub(r'__ch\d+$', '', path.stem)
    if domain == 'child':
        # Same text/session recorded by multiple phones is one speech event.
        return row['speaker_id'] + ':' + '_'.join(path.stem.split('_')[-2:])
    return path.stem


def prepare():
    destination = RECORD / 'DATA_MANIFEST.json'
    if destination.exists():
        raise FileExistsError('frozen manifest already exists')
    rows, excluded, splits = [], collections.Counter(), collections.defaultdict(set)
    for domain in ('child', 'youth', 'meeting', 'senior'):
        path = MANIFESTS[domain]
        source = list(csv.DictReader(path.open())) if path.suffix == '.csv' else [json.loads(l) for l in path.open()]
        grouped = collections.defaultdict(list)
        for r in source:
            if domain == 'senior' and r['split'] != 'test':
                continue  # Entire senior corpus excluded from student fitting/selection.
            p = resolve(r, domain)
            if not p.is_file():
                excluded[domain + ':missing'] += 1
                continue
            with wave.open(str(p), 'rb') as w:
                seconds = w.getnframes() / w.getframerate()
            if seconds < 3.0:
                excluded[domain + ':shorter_than_3s'] += 1
                continue
            item = dict(path=str(p), domain=domain, speaker=r['speaker_id'],
                        split=r['split'], gender=str(r.get('gender', 'unknown')).lower(),
                        event=event_id(r, domain, p), device=r.get('device', 'unknown'),
                        source_seconds=seconds)
            grouped[(r['split'], r['speaker_id'])].append(item)
        chosen = collections.defaultdict(list)
        for (split, speaker), items in grouped.items():
            by_event = {}
            for item in sorted(items, key=lambda x: stable(x['path'])):
                by_event.setdefault(item['event'], item)
            items = list(by_event.values())
            if len(items) >= 4:
                chosen[split].append((speaker, items))
        for split, groups in sorted(chosen.items()):
            cap = 96 if split == 'train' and domain == 'child' else 10000
            for speaker, items in sorted(groups, key=lambda x: stable(x[0]))[:cap]:
                for item in items[:(6 if split == 'train' else 4)]:
                    item['sha256'] = sha256_file(item['path'])
                    item['id'] = stable(item['path'])
                    item['crop_seconds'] = 3.0
                    rows.append(item)
                    splits[speaker].add(split)
    assert all(len(v) == 1 for v in splits.values()), 'speaker split overlap'
    assert len({x['sha256'] for x in rows}) == len(rows), 'duplicate audio across manifest'
    rows.sort(key=lambda x: (x['split'], x['domain'], x['speaker'], x['id']))
    counts = {}
    for split in ('train', 'dev', 'test'):
        for d in ('child', 'youth', 'meeting', 'senior'):
            subset = [r for r in rows if r['split'] == split and r['domain'] == d]
            if subset:
                counts[split + '/' + d] = dict(files=len(subset), speakers=len({r['speaker'] for r in subset}))
    write_json(destination, dict(rows=rows, counts=counts, excluded=dict(excluded),
        split_audit='speaker disjoint within corpus, distinct speech event pairs',
        caveats=['Public dev/test sets were used by earlier versions. Not pristine.',
                 'Teacher pretraining data overlap with some public corpora cannot be excluded.',
                 'Fixed three-second proxy does not establish full utterance/official 15-scenario performance.',
                 'Speaker identities across unrelated corpora are namespaced, not acoustically verified.']))
    print(json.dumps(counts), flush=True)


def load_manifest():
    path = RECORD / 'DATA_MANIFEST.json'
    return json.loads(path.read_text())['rows'], sha256_file(path)


def cache(device):
    from native_w2vbert2 import NativeFeatures, load_native_teacher
    rows, digest = load_manifest()
    path = RECORD / 'acoustic_cache.npz'
    if path.exists():
        raise FileExistsError(path)
    started = time.time()
    waves = []
    for r in rows:
        assert sha256_file(r['path']) == r['sha256'], 'source changed'
        x = load_audio(r['path'], 16000, remove_dc=False)
        start = (len(x) - 48000) // 2
        waves.append(np.ascontiguousarray(x[start:start + 48000], dtype=np.float32))
    waves = np.stack(waves)
    teacher = load_native_teacher(TEACHER, CONFIG).to(device).eval()
    frontend = NativeFeatures()
    targets = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 4):
            value = teacher(frontend(waves[offset:offset+4]).to(device)).cpu().numpy()
            assert np.isfinite(value).all()
            targets.extend(value)
            if offset % 120 == 0:
                print(json.dumps(dict(phase='teacher_cache', files=offset+len(value), total=len(rows), seconds=time.time()-started)), flush=True)
    del teacher
    if device == 'mps':
        torch.mps.empty_cache()
    model, _ = _load_redim(SOURCE, 'lm', device)
    baseline = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 16):
            baseline.extend(F.normalize(model(torch.from_numpy(waves[offset:offset+16]).to(device)), dim=1).cpu().numpy())
    np.savez(path, waves=waves, teacher=np.asarray(targets, np.float32), base_long=np.asarray(baseline, np.float32), manifest_sha256=np.asarray(digest))
    write_json(RECORD / 'CACHE_RECEIPT.json', dict(manifest_sha256=digest, cache_sha256=sha256_file(path),
        teacher_sha256=sha256_file(TEACHER), source_sha256=sha256_file(SOURCE),
        source_count=len(rows), seconds=time.time()-started, device=device))
    print('CACHE_COMPLETE', flush=True)


def read_cache():
    rows, digest = load_manifest()
    cache_path = RECORD / 'acoustic_cache.npz'
    receipt = json.loads((RECORD / 'CACHE_RECEIPT.json').read_text())
    assert receipt['cache_sha256'] == sha256_file(cache_path), 'cache bytes changed'
    assert receipt['teacher_sha256'] == sha256_file(TEACHER), 'teacher changed'
    assert receipt['source_sha256'] == sha256_file(SOURCE), 'student initialization changed'
    p = np.load(cache_path, allow_pickle=False)
    assert str(p['manifest_sha256']) == digest
    assert p['waves'].shape == (len(rows), 48000)
    for name,dim in [('teacher',256),('base_long',192)]:
        values = p[name]
        assert values.shape == (len(rows),dim) and np.isfinite(values).all()
        assert np.max(np.abs(np.linalg.norm(values,axis=1)-1)) < 1e-4
    return rows, p


def informative_relation(student_a, student_b, teacher_a, teacher_b, labels, domains):
    """Cross-utterance positives + top-3 hard negatives, balanced by domain.

    Domain balance stops numerous cross-corpus pairs from overwhelming the
    difficult within-corpus identities. Selection uses teacher scores only.
    This is a local extension inspired by relational KD, not paper reproduction.
    """
    s = F.normalize(student_a, dim=1) @ F.normalize(student_b, dim=1).T
    t = F.normalize(teacher_a, dim=1) @ F.normalize(teacher_b, dim=1).T
    positive = labels[:, None] == labels[None, :]
    terms = [(s[positive] - t[positive]).square().mean()]
    for within in (True, False):
        allowed = (~positive) & ((domains[:, None] == domains[None, :]) == within)
        hard = t.masked_fill(~allowed, -100).topk(min(3, t.shape[1]), dim=1).indices
        mask = torch.zeros_like(allowed).scatter(1, hard, True) & allowed
        if mask.any():
            terms.append((s[mask] - t[mask]).square().mean())
    return torch.stack(terms).mean()


def train(args):
    rows, payload = read_cache()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, row in enumerate(rows):
        if row['split'] == 'train' and row['domain'] != args.leave_out:
            groups[row['domain']][row['speaker']].append(i)
    keys = [(d, s) for d in sorted(groups) for s in sorted(groups[d])]
    lookup = {key: i for i, key in enumerate(keys)}
    waves, targets, base = payload['waves'], payload['teacher'], payload['base_long']
    student, _ = _load_redim(SOURCE, 'lm', device)
    parameters = configure_student(student, False)
    student.eval()  # Frozen BN and inference-style feature extraction in every route.
    prototypes = np.stack([normalize(base[groups[d][s]].mean(0)) for d, s in keys])
    classifier = torch.nn.Parameter(torch.from_numpy(prototypes).to(device))
    # Linear alignment trained on TRAIN ONLY, used only for vector-KD control.
    indices = [i for d, s in keys for i in groups[d][s]]
    a = base[indices].astype(np.float64)
    b = targets[indices].astype(np.float64)
    # Explicit contractions avoid Apple Accelerate spurious floating-point
    # flags observed in NumPy's BLAS matmul. Never accept a nonfinite mapping.
    gram = np.einsum('ni,nj->ij', a, a, optimize=False)
    cross = np.einsum('ni,nj->ij', a, b, optimize=False)
    projection = np.linalg.solve(gram + .1*np.eye(192), cross).astype(np.float32)
    assert np.isfinite(projection).all(), 'nonfinite TRAIN-only alignment'
    projection = torch.from_numpy(projection).to(device)
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr':2e-5},
                                  {'params':[classifier], 'lr':5e-4}], weight_decay=1e-4)
    started, trace = time.time(), []
    out = RECORD / ('train_' + args.mode + '_s' + str(args.seed) + ('_without_' + args.leave_out if args.leave_out else ''))
    if (out / 'student.pt').exists():
        raise FileExistsError('training output already exists')
    out.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps+1):
        left, right, labels, domains = [], [], [], []
        for di, d in enumerate(sorted(groups)):
            speakers = rng.sample(sorted(groups[d]), min(4 if d != 'meeting' else 2, len(groups[d])))
            for s in speakers:
                i, j = rng.sample(groups[d][s], 2)
                assert rows[i]['event'] != rows[j]['event']
                left.append(i); right.append(j); labels.append(lookup[(d,s)]); domains.append(di)
        length = (8000, 16000, 32000)[(step-1)%3]
        short = []
        for j in right:
            start = rng.randrange(48000-length+1)
            short.append(augment_channel(waves[j, start:start+length], rng, .25))
        x = torch.from_numpy(waves[left]).to(device)
        y = torch.from_numpy(np.stack(short)).to(device)
        label = torch.tensor(labels, device=device)
        domain = torch.tensor(domains, device=device)
        ta = torch.from_numpy(targets[left]).to(device)
        tb = torch.from_numpy(targets[right]).to(device)
        sa, sb = student(x), student(y)
        margin = .15 * min(1., step/30.)
        classification = .5*(F.cross_entropy(aam_logits(sa,classifier,label,margin,24),label)+F.cross_entropy(aam_logits(sb,classifier,label,margin,24),label))
        contrast = paired_info_nce(sa,sb,.1)
        preserve = (1-F.cosine_similarity(sa,torch.from_numpy(base[left]).to(device))).mean()
        kd = sa.new_zeros(())
        if args.mode == 'vector':
            kd = .5*((1-F.cosine_similarity(sa@projection,ta)).mean()+(1-F.cosine_similarity(sb@projection,tb)).mean())
        elif args.mode == 'relation':
            kd = informative_relation(sa,sb,ta,tb,label,domain)
        loss = classification + .25*contrast + .1*preserve + (2.0 if args.mode=='relation' else .5)*kd
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.)
        optimizer.step()
        if step==1 or step%20==0 or step==args.steps:
            row=dict(step=step, seconds=time.time()-started, loss=float(loss.detach().cpu()),
                     classification=float(classification.detach().cpu()), kd=float(kd.detach().cpu()),
                     duration=length/16000, preserve=float(preserve.detach().cpu()))
            trace.append(row)
            print(json.dumps(dict(run=out.name, **row)),flush=True)
            write_json(out/'PROGRESS.json', dict(arguments=vars(args), trace=trace))
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    state = {k:v.detach().cpu() for k,v in student.state_dict().items()}
    changed = [k for k in state if not torch.equal(state[k], source['state_dict'][k])]
    assert changed and any(k.startswith('backbone.stage5') for k in changed), 'encoder not trained'
    torch.save(dict(model_config=source['model_config'], state_dict=state), out/'student.pt')
    write_json(out/'TRAINING_RECEIPT.json', dict(arguments=vars(args), trace=trace,
        checkpoint_sha256=sha256_file(out/'student.pt'), source_sha256=sha256_file(SOURCE),
        manifest_sha256=sha256_file(RECORD/'DATA_MANIFEST.json'),
        changed_tensors=changed, trainable_parameters=sum(p.numel() for p in parameters),
        train_speakers=len(keys), train_domains=sorted(groups), seconds=time.time()-started,
        release_status='RESEARCH_ONLY_PENDING_GATES', hidden_score_guarantee=False))
    print('TRAIN_COMPLETE ' + out.name, flush=True)


def embeddings(model, waves, device):
    output = {}
    with torch.inference_mode():
        for duration, length in [('long3',48000),('short05',8000),('short10',16000)]:
            values = []
            for offset in range(0,len(waves),16):
                x = waves[offset:offset+16, (48000-length)//2:(48000+length)//2]
                e = F.normalize(model(torch.from_numpy(np.ascontiguousarray(x)).to(device)), dim=1).cpu().numpy()
                assert np.isfinite(e).all()
                values.extend(e)
            output[duration] = np.asarray(values,np.float32)
    return output


def paired_trials(rows):
    # Enrollment: first frozen event, tests: remaining three distinct events.
    groups = collections.defaultdict(list)
    for i,r in enumerate(rows):
        groups[r['speaker']].append(i)
    pairs=[]
    for speaker, indices in sorted(groups.items()):
        for t in indices[1:]:
            for other, enrolls in sorted(groups.items()):
                e = enrolls[0]
                assert e!=t
                pairs.append((e,t,int(speaker==other)))
    return np.asarray(pairs,dtype=np.int64)


def evaluate(args):
    from redimnet2.redimnet2 import ReDimNet2Wrap
    rows,payload=read_cache()
    chosen=[i for i,r in enumerate(rows) if r['split']==args.split]
    selected=[rows[i] for i in chosen]
    if args.checkpoint:
        p=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        model=ReDimNet2Wrap(**p['model_config'])
        model.load_state_dict(p['state_dict'],strict=True)
        model.to(args.device).eval()
        name=Path(args.checkpoint).parent.name
    else:
        model,_=_load_redim(SOURCE,'lm',args.device)
        name='untrained_redim'
    values=embeddings(model,payload['waves'][chosen],args.device)
    out=RECORD/'evaluations'/args.split/name
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'embeddings.npz',ids=np.asarray([r['id'] for r in selected]),**values)
    metrics=[]
    for domain in sorted({r['domain'] for r in selected}):
        ix=[i for i,r in enumerate(selected) if r['domain']==domain]
        rr=[selected[i] for i in ix]
        if len({r['speaker'] for r in rr})<2:
            continue
        pairs=paired_trials(rr)
        for duration in values:
            scores=np.sum(values['long3'][ix][pairs[:,0]]*values[duration][ix][pairs[:,1]],axis=1)
            eer=compute_eer_percent(pairs[:,2].tolist(),scores.tolist())
            metrics.append(dict(domain=domain,duration=duration,eer=eer, speakers=len({r['speaker'] for r in rr}),trials=len(pairs)))
    write_json(out/'METRICS.json',dict(name=name,split=args.split,cells=metrics,
        macro_eer=float(np.mean([r['eer'] for r in metrics])),
        checkpoint_sha256=sha256_file(args.checkpoint or SOURCE),
        proxy_only=True,official_15_scenario_coverage=False))
    print(json.dumps(dict(name=name,split=args.split,cells=metrics)),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('prepare','cache','train','evaluate'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--mode',choices=('supervised','vector','relation'),default='relation')
    p.add_argument('--seed',type=int,default=5101)
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--leave-out',default='')
    p.add_argument('--checkpoint')
    p.add_argument('--split',choices=('dev','test'),default='dev')
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.command=='prepare':prepare()
    elif args.command=='cache':cache(args.device)
    elif args.command=='train':train(args)
    else:evaluate(args)


if __name__=='__main__':
    main()
