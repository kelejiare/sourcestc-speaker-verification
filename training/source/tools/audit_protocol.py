#!/usr/bin/env python3
"""Read-only model audit: fixed calibrated_fusion/calibration caches, no fitting or package writes.

The added cross-age trial protocol is diagnostic, not an independent test set.
It intentionally makes missing official-scene coverage visible.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_distillation_gates import compute_eer_percent

RECORD = ROOT / "records/2026-09-16/protocol_audit"
CACHE = ROOT / "records/2026-09-13/126_lora_lora_duration_experts"
DURATIONS = ("long3", "short05", "short10")


def norm(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def eer(labels, scores, weights=None):
    """Equivalent to the existing interpolated, tie-grouped EER, vectorized."""
    labels = np.asarray(labels)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights)
    order = np.argsort(-scores, kind="stable")
    y, s, w = labels[order], scores[order], weights[order]
    if w[y == 1].sum() <= 0 or w[y == 0].sum() <= 0:
        raise ValueError("bootstrap lost a trial class")
    ends = np.r_[np.where(s[:-1] != s[1:])[0], len(s) - 1]
    far = np.r_[0., np.cumsum(w * (1-y))[ends] / w[y == 0].sum()]
    frr = np.r_[1., 1. - np.cumsum(w * y)[ends] / w[y == 1].sum()]
    k = np.where(frr-far <= 0)[0][0]
    d0, d1 = frr[k-1]-far[k-1], frr[k]-far[k]
    return float(100 * (far[k-1] + d0/(d0-d1)*(far[k]-far[k-1])))


def pairs(rows, enroll_domains, test_domains):
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["speaker"]].append(i)
    enroll = [ix[0] for ix in groups.values() if rows[ix[0]]["domain"] in enroll_domains]
    test = [i for ix in groups.values() for i in ix[1:] if rows[i]["domain"] in test_domains]
    return np.asarray([(e,t,int(rows[e]["speaker"] == rows[t]["speaker"]))
                       for e in enroll for t in test], dtype=np.int64)


def bootstrap(rows, trial, before, after, repeats=400):
    names = sorted({row["speaker"] for row in rows})
    sid = np.asarray([names.index(row["speaker"]) for row in rows])
    left, right, labels = trial.T
    sl, sr = sid[left], sid[right]
    rng = np.random.RandomState(7001)
    values = {"legacy_pooled_row_equality": [], "macro_row_equality": [],
              "macro_speaker_cluster": []}
    repeated_labels = np.tile(labels, 3)
    before_pool = np.concatenate(before)
    after_pool = np.concatenate(after)
    for _ in range(repeats):
        count = np.bincount(rng.randint(len(names), size=len(names)), minlength=len(names))
        old_weight = np.where(left == right, count[sl], count[sl]*count[sr])
        # One copy of every genuine same-speaker trial per sampled cluster.
        weight = np.where(sl == sr, count[sl], count[sl]*count[sr])
        if min(weight[labels == 0].sum(), weight[labels == 1].sum()) <= 0:
            continue
        values["legacy_pooled_row_equality"].append(
            eer(repeated_labels, after_pool, np.tile(old_weight, 3)) -
            eer(repeated_labels, before_pool, np.tile(old_weight, 3)))
        for key, w in [("macro_row_equality",old_weight), ("macro_speaker_cluster",weight)]:
            values[key].append(float(np.mean([eer(labels,b,w)-eer(labels,a,w)
                                               for a,b in zip(before,after)])))
    return {k: {"mean_delta": float(np.mean(v)), "ci95": np.quantile(v,[.025,.975]).tolist(),
                "valid_repeats": len(v)} for k,v in values.items()}


def main():
    manifest = json.loads((CACHE / "DATA_MANIFEST.json").read_text())
    by_id = {row["id"]: row for row in manifest["rows"]}
    anchor_path = CACHE / "evaluations/dev/plus_online_anchor/embeddings.npz"
    teacher_path = CACHE / "evaluations/dev/lora_duration_s5201/embeddings.npz"
    anchor = dict(np.load(anchor_path, allow_pickle=False))
    teacher = dict(np.load(teacher_path, allow_pickle=False))
    assert list(anchor["ids"]) == list(teacher["ids"])
    rows = [by_id[str(i)] for i in anchor["ids"]]
    assert all(row["split"] == "dev" for row in rows)
    raw = np.load(ROOT / "model/native_teacher_w2vbert2_nap_adapter.npz",allow_pickle=False)["transform"]
    transform = np.eye(256) + .75*(raw-np.eye(256))
    teachers = {d: norm(teacher[d]) for d in DURATIONS}
    nap = {d: norm(teacher[d].astype(np.float64) @ transform.T) for d in DURATIONS}
    old_alpha = (.45,.32,.25)
    new_alpha = (.50,.325,.30)
    protocols = [("child_within", {"child"}, {"child"}),
                 ("youth_within", {"youth"}, {"youth"}),
                 ("child_enroll_child_youth_tests", {"child"}, {"child","youth"})]
    report = {"schema": "protocol_protocol_audit_v1", "fits_new_model": False,
              "independent_confirmation": False, "hidden_score_guarantee": False,
              "comparison": "fixed calibration source1 NAP policy versus fixed calibrated_fusion source1",
              "limitations": ["Reuses previously inspected dev speakers and center crops",
                              "Children and adults originate in different corpora",
                              "Missing source2 child/adult and source1 nearfar matched coverage",
                              "No new-model or submission authorization"], "protocols": {}}
    for name, ed, td in protocols:
        trial = pairs(rows, ed, td)
        left,right,labels = trial.T
        weights = np.ones(len(trial))
        if name.startswith("child_enroll"):
            # Partial official 7:6:1 composition: senior absent, explicitly 7:6 only.
            test_domain = np.asarray([rows[t]["domain"] for t in right])
            for domain, mass in [("child",7.),("youth",6.)]:
                mask = test_domain == domain
                weights[mask] = mass / int(mask.sum())
        cells, before, after = [], [], []
        for k, d in enumerate(DURATIONS):
            a = np.sum(anchor["long3"][left].astype(np.float64)*anchor[d][right],axis=1)
            t = np.sum(teachers["long3"][left]*teachers[d][right],axis=1)
            n = np.sum(nap["long3"][left]*nap[d][right],axis=1)
            old = np.sqrt((1-old_alpha[0])*(1-old_alpha[k]))*a + np.sqrt(old_alpha[0]*old_alpha[k])*t
            new = np.sqrt((1-new_alpha[0])*(1-new_alpha[k]))*a + np.sqrt(new_alpha[0]*new_alpha[k])*n
            # Validate the vectorized implementation against the established metric.
            for scores in [old,new]:
                assert abs(eer(labels,scores,weights)-compute_eer_percent(labels,scores,weights)) < 1e-8
            cells.append({"duration": d, "calibrated_fusion_eer": eer(labels,old,weights),
                          "calibration_eer": eer(labels,new,weights),
                          "delta": eer(labels,new,weights)-eer(labels,old,weights)})
            before.append(old); after.append(new)
        report["protocols"][name] = {"trials": len(trial),"positive":int(labels.sum()),
                                     "cells":cells,"macro_delta":float(np.mean([c["delta"] for c in cells]))}
        if name == "child_within":
            used = sorted(set(left.tolist()+right.tolist()))
            local_rows = [rows[i] for i in used]
            mapping = {j:i for i,j in enumerate(used)}
            local = np.asarray([(mapping[e],mapping[t],y) for e,t,y in trial])
            report["bootstrap_audit"] = bootstrap(local_rows,local,before,after)
    rng = np.random.RandomState(7)
    for _ in range(30):
        y = np.r_[np.zeros(40,dtype=int),np.ones(40,dtype=int)]
        s = rng.randint(0,9,80).astype(float)
        w = rng.randint(0,5,80).astype(float)
        assert abs(eer(y,s,w)-compute_eer_percent(y,s,w)) < 1e-8
    counts = collections.defaultdict(lambda: collections.defaultdict(set))
    for row in rows:
        gender = {"f":"female","m":"male"}.get(row.get("gender"), row.get("gender","unknown"))
        counts[row["domain"]][gender].add(row["speaker"])
    report["dev_speaker_coverage"] = {d:{g:len(s) for g,s in v.items()} for d,v in counts.items()}
    report["tests"] = {"metric_matches_existing":True,"tied_weighted_random_cases":30,
                       "speaker_cluster_rule_example": "same-speaker count 2 -> trial weight 2, not 4"}
    report["source_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [anchor_path,teacher_path,CACHE/"DATA_MANIFEST.json",Path(__file__)]}
    RECORD.mkdir(parents=True,exist_ok=True)
    (RECORD/"PROTOCOL_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_sha256",)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
