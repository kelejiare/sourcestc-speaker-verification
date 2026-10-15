#!/usr/bin/env python3
"""Train a ReDimNet2 acoustic expert with globally supervised speaker labels.

This is an experiment, not a submission builder.  It addresses the main flaw
in mobile_expert: batch-local pair classification never learned a stable global speaker
decision boundary.  The new objective combines speaker-balanced AAM-Softmax,
paired supervised contrastive learning, frozen-teacher feature/relation
distillation and optional adversarial domain removal.

Only audited speaker-disjoint *training* partitions are read.  HI-MIA paths are
reconstructed by joining its split manifest to the immutable file inventory;
the six-speaker H1 partition remains unseen for final confirmation.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_features import load_audio
from multidomain_kd_inference import _load_redim, sha256_file


MANIFEST_ROOT = ROOT / "records/2026-09-02/013_distillation_domain_kd/data_manifests"
MANIFESTS = {
    "child": MANIFEST_ROOT / "childmandarin.csv",
    "distance": MANIFEST_ROOT / "3dspeaker_test.csv",
    "meeting": MANIFEST_ROOT / "aishell4_meeting.jsonl",
    "youth": MANIFEST_ROOT / "commonvoice_beijing_youth.jsonl",
    "senior": MANIFEST_ROOT / "seniortalk.csv",
}
HI_SPLIT = ROOT / "data/distillation_sources/hi_mia_slr85/splits/cam_bn_aam_clean_v1/HI_MIA_DEV_UTTERANCE_SPLIT_MANIFEST_2026-09-03.jsonl"
HI_INVENTORY = ROOT / "data/distillation_sources/hi_mia_slr85/inventory/HI_MIA_DEV_EXTRACTED_FILE_INVENTORY_2026-09-03.jsonl"
HI_AUDIO_ROOT = ROOT / "data/distillation_sources/hi_mia_slr85/extracted/dev_2026-09-03_prepared_v1"
SOURCE_CHECKPOINT = ROOT / "model/redimnet2-b6-vb2-vox2-cnc2-lm.pt"
SOURCE_SHA256 = "287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868"
OUTPUT_ROOT = ROOT / "records/2026-09-11/089_domain_expert_redimnet2_aam_domain_expert"


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_rows(path):
    path = Path(path)
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    return read_jsonl(path)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def canonical_item(row):
    path = Path(str(row["path"]))
    # The full ChildMandarin training archive is intentionally kept in the
    # repository as a tarball.  Experiments may unpack it under /private/tmp
    # to avoid duplicating several gigabytes inside the release workspace.
    if not path.is_file() and "/childmandarin/extracted/train/" in str(path):
        suffix = str(path).split("/childmandarin/extracted/train/", 1)[1]
        candidate = Path("/private/tmp/game823-child-train/train") / suffix
        if candidate.is_file():
            path = candidate
    return {
        "path": str(path),
        "speaker": str(row["speaker_id"]),
        "condition": "|".join((
            str(row.get("domain", "")), str(row.get("device", "")),
            str(row.get("distance", "")), str(row.get("session", "")),
        )),
    }


def load_himia_rows():
    inventory = {
        str(row["normalized_path_sha256"]): row
        for row in read_jsonl(HI_INVENTORY)
        if row.get("object_type") == "regular_file"
        and str(row.get("relative_path", "")).lower().endswith(".wav")
    }
    rows = []
    missing = []
    for row in read_jsonl(HI_SPLIT):
        if str(row.get("split")) != "train":
            continue
        key = str(row["normalized_path_sha256"])
        source = inventory.get(key)
        if source is None:
            missing.append(key)
            continue
        path = HI_AUDIO_ROOT / str(source["relative_path"])
        if not path.is_file():
            missing.append(key)
            continue
        rows.append({
            "path": str(path),
            "speaker_id": "himia:" + str(row["speaker_stable_id"]),
            "split": "train",
            "domain": "himia",
            "device": str(row["channel_stable_id"]),
            "distance": str(row["distance_stable_id"]),
            "session": str(row["session_stable_id"]),
        })
    if missing:
        raise RuntimeError("HI-MIA join failed for {} train rows".format(len(missing)))
    return rows


def enabled_domains(mode):
    if mode == "child":
        return ("child",)
    if mode == "meeting_child":
        return ("child", "meeting")
    if mode == "mobile":
        return ("child", "distance", "himia", "senior", "youth")
    if mode == "meeting":
        return ("distance", "himia", "meeting")
    if mode == "joint":
        return ("child", "distance", "himia", "meeting", "senior", "youth")
    raise ValueError("unknown mode: {}".format(mode))


def load_groups(mode, max_files_per_speaker, seed):
    groups = {}
    audit = {}
    for domain in enabled_domains(mode):
        source_rows = load_himia_rows() if domain == "himia" else read_rows(MANIFESTS[domain])
        by_speaker = defaultdict(list)
        for row in source_rows:
            if str(row.get("split")) != "train":
                continue
            item = canonical_item(row)
            if Path(item["path"]).suffix.lower() != ".wav" or not Path(item["path"]).is_file():
                continue
            by_speaker[item["speaker"]].append(item)
        selected = {}
        for speaker, items in sorted(by_speaker.items()):
            if len(items) < 2:
                continue
            ranked = sorted(items, key=lambda item: hashlib.sha256(
                (str(seed) + "\0" + domain + "\0" + speaker + "\0" + item["path"]).encode("utf-8")
            ).hexdigest())
            selected[speaker] = ranked[:max_files_per_speaker]
        if len(selected) < 2:
            raise RuntimeError("too few speakers for domain {}".format(domain))
        groups[domain] = selected
        audit[domain] = {
            "speaker_count": len(selected),
            "utterance_count": sum(len(items) for items in selected.values()),
            "cross_condition_speaker_count": sum(
                len({item["condition"] for item in items}) >= 2
                for items in selected.values()
            ),
        }
    return groups, audit


def crop_or_wrap(waveform, samples, rng):
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size >= samples:
        maximum = waveform.size - samples
        start = 0 if maximum == 0 else rng.randrange(maximum + 1)
        return np.ascontiguousarray(waveform[start:start + samples])
    repeats = int(math.ceil(float(samples) / float(max(1, waveform.size))))
    return np.ascontiguousarray(np.tile(waveform, repeats)[:samples], dtype=np.float32)


def augment_channel(waveform, rng, reverb_probability):
    value = np.asarray(waveform, dtype=np.float32).copy()
    value *= float(10.0 ** (rng.uniform(-6.0, 3.0) / 20.0))
    if value.size > 2 and rng.random() < 0.65:
        coefficient = rng.uniform(-0.22, 0.22)
        value[1:] += coefficient * value[:-1]
    if value.size > 1024 and rng.random() < reverb_probability:
        source = value.copy()
        for _ in range(rng.randint(1, 3)):
            delay = rng.randint(128, 768)
            gain = rng.uniform(0.04, 0.18)
            value[delay:] += gain * source[:-delay]
    if rng.random() < 0.65:
        power = float(np.mean(value * value) + 1.0e-10)
        snr_db = rng.uniform(10.0, 30.0)
        sigma = math.sqrt(power / (10.0 ** (snr_db / 10.0)))
        noise = np.random.RandomState(rng.randrange(2 ** 32)).normal(
            0.0, sigma, size=value.shape
        ).astype(np.float32)
        value += noise
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if peak > 0.98:
        value *= 0.98 / peak
    return np.ascontiguousarray(value, dtype=np.float32)


def choose_pair(items, rng):
    first = rng.choice(items)
    cross = [item for item in items if item["condition"] != first["condition"]]
    second = rng.choice(cross if cross else [item for item in items if item is not first])
    return first, second


def sample_batch(groups, labels, speakers_per_domain, anchor_seconds,
                 short_seconds, reverb_probability, rng):
    anchors = []
    shorts = []
    targets = []
    domains = []
    for domain_index, domain in enumerate(sorted(groups)):
        speakers = sorted(groups[domain])
        chosen = rng.sample(speakers, min(speakers_per_domain, len(speakers)))
        for speaker in chosen:
            first, second = choose_pair(groups[domain][speaker], rng)
            anchor = load_audio(Path(first["path"]), 16000, remove_dc=False)
            short = load_audio(Path(second["path"]), 16000, remove_dc=False)
            anchor = crop_or_wrap(anchor, int(round(anchor_seconds * 16000)), rng)
            short = crop_or_wrap(short, int(round(short_seconds * 16000)), rng)
            short = augment_channel(short, rng, reverb_probability)
            anchors.append(anchor)
            shorts.append(short)
            targets.append(labels[(domain, speaker)])
            domains.append(domain_index)
    return (
        torch.from_numpy(np.stack(anchors)).float(),
        torch.from_numpy(np.stack(shorts)).float(),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(domains, dtype=torch.long),
    )


def configure_student(model, unfreeze_stage4):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [
        model.backbone.stage5, model.backbone.fin_wghtd if hasattr(model.backbone, "fin_wghtd") else model.backbone.fin_wght1d,
        model.backbone.head, model.pool, model.bn, model.linear,
    ]
    if unfreeze_stage4:
        modules.append(model.backbone.stage4)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def aam_logits(embeddings, weight, targets, margin, scale):
    cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(weight, dim=1))
    cosine = cosine.clamp(-1.0 + 1.0e-5, 1.0 - 1.0e-5)
    sine = torch.sqrt((1.0 - cosine * cosine).clamp_min(1.0e-8))
    phi = cosine * math.cos(margin) - sine * math.sin(margin)
    one_hot = F.one_hot(targets, num_classes=cosine.shape[1]).to(cosine.dtype)
    return (one_hot * phi + (1.0 - one_hot) * cosine) * scale


def paired_info_nce(anchor, short, temperature):
    left = F.normalize(anchor, dim=1)
    right = F.normalize(short, dim=1)
    logits = torch.mm(left, right.t()) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def run(args):
    started = time.time()
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type != "cpu" and device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if sha256_file(SOURCE_CHECKPOINT) != SOURCE_SHA256:
        raise RuntimeError("source ReDimNet2 SHA-256 mismatch")
    student, metadata = _load_redim(SOURCE_CHECKPOINT, "lm", device)
    trainable = configure_student(student, args.unfreeze_stage4)
    trainable_anchor = [parameter.detach().clone() for parameter in trainable]
    teacher = None
    if args.distill_mode == "feature":
        teacher, _ = _load_redim(SOURCE_CHECKPOINT, "lm", device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    groups, data_audit = load_groups(args.mode, args.max_files_per_speaker, seed)
    label_keys = [
        (domain, speaker)
        for domain in sorted(groups)
        for speaker in sorted(groups[domain])
    ]
    label_lookup = {key: index for index, key in enumerate(label_keys)}
    classifier = nn.Parameter(torch.randn(len(label_keys), 192, device=device) * 0.02)
    domain_head = nn.Linear(192, len(groups)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": trainable, "lr": args.learning_rate},
        {"params": [classifier], "lr": args.classifier_learning_rate},
        {"params": domain_head.parameters(), "lr": args.classifier_learning_rate},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.steps), eta_min=args.learning_rate * 0.1
    )
    rng = random.Random(seed * 1000003 + 823)
    trace = []
    student.train()
    for module in student.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()

    for step in range(1, args.steps + 1):
        anchor_seconds = rng.choices((2.0, 3.0, 4.0), (0.25, 0.45, 0.30), k=1)[0]
        short_seconds = rng.choices((0.50, 1.00, 2.00), (0.40, 0.35, 0.25), k=1)[0]
        anchors, shorts, targets, domains = sample_batch(
            groups, label_lookup, args.speakers_per_domain, anchor_seconds,
            short_seconds, args.reverb_probability, rng
        )
        anchors = anchors.to(device)
        shorts = shorts.to(device)
        targets = targets.to(device)
        domains = domains.to(device)
        student_anchor = student(anchors)
        student_short = student(shorts)
        progress = float(step) / float(max(1, args.steps))
        margin = args.margin * min(1.0, progress / max(args.margin_warmup, 1.0e-6))
        classification = 0.5 * (
            F.cross_entropy(aam_logits(student_anchor, classifier, targets, margin, args.scale), targets)
            + F.cross_entropy(aam_logits(student_short, classifier, targets, margin, args.scale), targets)
        )
        contrastive = paired_info_nce(student_anchor, student_short, args.temperature)
        if teacher is not None:
            with torch.no_grad():
                teacher_anchor = teacher(anchors)
                teacher_short = teacher(shorts)
            feature_anchor = 0.5 * (
                (1.0 - F.cosine_similarity(student_anchor, teacher_anchor, dim=1)).mean()
                + (1.0 - F.cosine_similarity(student_short, teacher_short, dim=1)).mean()
            )
            student_all = F.normalize(torch.cat((student_anchor, student_short), dim=0), dim=1)
            teacher_all = F.normalize(torch.cat((teacher_anchor, teacher_short), dim=0), dim=1)
            relation = F.mse_loss(
                torch.mm(student_all, student_all.t()),
                torch.mm(teacher_all, teacher_all.t()),
            )
        else:
            relative_drift = []
            for parameter, reference in zip(trainable, trainable_anchor):
                denominator = reference.square().mean().detach().clamp_min(1.0e-8)
                relative_drift.append((parameter - reference).square().mean() / denominator)
            feature_anchor = torch.stack(relative_drift).mean()
            relation = feature_anchor.new_zeros(())
        domain_strength = args.domain_weight * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
        domain_features = GradientReverse.apply(
            F.normalize(torch.cat((student_anchor, student_short), dim=0), dim=1),
            domain_strength,
        )
        domain_targets = torch.cat((domains, domains), dim=0)
        domain_loss = F.cross_entropy(domain_head(domain_features), domain_targets)
        loss = (
            classification
            + args.contrastive_weight * contrastive
            + args.teacher_weight * feature_anchor
            + args.relation_weight * relation
            + domain_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss at step {}".format(step))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            row = {
                "step": step,
                "anchor_seconds": anchor_seconds,
                "short_seconds": short_seconds,
                "margin": margin,
                "model_lr": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                "classification": float(classification.detach().cpu()),
                "contrastive": float(contrastive.detach().cpu()),
                "teacher_anchor": float(feature_anchor.detach().cpu()),
                "relation": float(relation.detach().cpu()),
                "domain_loss": float(domain_loss.detach().cpu()),
                "domain_grl_strength": domain_strength,
                "elapsed_seconds": time.time() - started,
            }
            trace.append(row)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

    output_dir = OUTPUT_ROOT / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "redimnet2_aam_domain_expert.pt"
    source_payload = torch.load(str(SOURCE_CHECKPOINT), map_location="cpu")
    payload = {
        "model_config": dict(source_payload["model_config"]),
        "state_dict": {key: value.detach().cpu() for key, value in student.state_dict().items()},
    }
    torch.save(payload, str(checkpoint))
    receipt = {
        "schema_version": 1,
        "artifact_kind": "domain_expert_redimnet2_aam_domain_expert_training",
        "name": args.name,
        "seed": seed,
        "mode": args.mode,
        "source_checkpoint": {
            "path": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
            "sha256": SOURCE_SHA256,
        },
        "output_checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "source_parameter_count": int(metadata["parameter_count"]),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "classification_speaker_count": len(label_keys),
        "data_audit": data_audit,
        "arguments": vars(args),
        "trace": trace,
        "elapsed_seconds": time.time() - started,
        "distillation_contract": (
            "frozen_teacher_cosine_and_relation" if teacher is not None
            else "L2-SP_relative_parameter_anchor"
        ),
        "hidden_score_guarantee": False,
    }
    atomic_json(output_dir / "TRAINING_RECEIPT.json", receipt)
    print(json.dumps({
        "status": "TRAINING_COMPLETE",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "sha256": receipt["output_checkpoint"]["sha256"],
        "elapsed_seconds": receipt["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=("child", "meeting_child", "mobile", "meeting", "joint"), default="joint")
    parser.add_argument("--seed", type=int, default=20260931)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--speakers-per-domain", type=int, default=1)
    parser.add_argument("--max-files-per-speaker", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--classifier-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--margin-warmup", type=float, default=0.35)
    parser.add_argument("--scale", type=float, default=24.0)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--contrastive-weight", type=float, default=0.40)
    parser.add_argument("--teacher-weight", type=float, default=3.0)
    parser.add_argument("--relation-weight", type=float, default=1.0)
    parser.add_argument("--distill-mode", choices=("parameter", "feature"), default="parameter")
    parser.add_argument("--domain-weight", type=float, default=0.0)
    parser.add_argument("--reverb-probability", type=float, default=0.30)
    parser.add_argument("--gradient-clip", type=float, default=3.0)
    parser.add_argument("--unfreeze-stage4", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
