#!/usr/bin/env python3
"""Build a true-duration cache for the multilingual w2v-BERT 2.0 teacher."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from audio_features import load_audio
from tools.build_embedding_cache import deterministic_crop_start
from w2vbert2_multidomain_model import (
    build_feature_extractor,
    extract_waveforms,
    load_teacher,
    sha256_file,
)


SUITES = {
    "new_holdouts": ROOT / "records/2026-09-04/027_distillation_new_holdout_cache/FROZEN_SOURCES.jsonl",
    "alimeeting_test": ROOT / "records/2026-09-04/030_distillation_alimeeting_test_cache/FROZEN_SOURCES.jsonl",
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def waveforms_for_batch(batch, duration, max_full_samples):
    output = []
    for _index, source in batch:
        waveform = load_audio(ROOT / source["source_path"], 16000, remove_dc=False)
        if duration == "full":
            if waveform.size <= max_full_samples:
                output.append(np.ascontiguousarray(waveform, dtype=np.float32))
            else:
                maximum = waveform.size - max_full_samples
                for start in sorted({0, maximum // 2, maximum}):
                    output.append(np.ascontiguousarray(
                        waveform[start:start + max_full_samples], dtype=np.float32
                    ))
            continue
        target = 8000 if duration == "0p5" else 16000
        if waveform.size < target:
            raise RuntimeError("source is shorter than requested crop: {}".format(source["source_path"]))
        for view in (0, 1):
            start = deterministic_crop_start(waveform.size, target, view, 2)
            output.append(
                np.ascontiguousarray(waveform[start:start + target], dtype=np.float32)
            )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES))
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--suite-name")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--dataset-prefix", default="")
    parser.add_argument("--batch-full", type=int, default=1)
    parser.add_argument("--batch-short", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-full-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if bool(args.source_manifest) == bool(args.suite):
        raise ValueError("choose exactly one of --suite or --source-manifest")
    if args.source_manifest and not args.suite_name:
        raise ValueError("--suite-name is required with --source-manifest")
    if (
        args.batch_full <= 0 or args.batch_short <= 0 or args.threads <= 0
        or args.max_full_seconds < 1.0
    ):
        raise ValueError("batch sizes and threads must be positive")
    # The public speaker head pools every output frame and exposes no padding
    # mask.  Full utterances therefore have to be evaluated one at a time to
    # match the author's inference path.  Fixed-duration short crops are equal
    # length and may still be batched safely.
    if args.batch_full != 1:
        raise ValueError("batch-full must be 1 for padding-safe full-utterance inference")

    suite_name = args.suite if args.suite else str(args.suite_name)
    source_manifest = SUITES[args.suite] if args.suite else args.source_manifest
    sources = read_jsonl(source_manifest)
    if args.dataset_prefix:
        sources = [
            row for row in sources
            if str(row.get("dataset", "")).startswith(args.dataset_prefix)
        ]
    if args.max_sources > 0:
        sources = sources[:args.max_sources]
    if not sources:
        raise RuntimeError("empty source selection")
    identities = [str(row["utterance_stable_id"]) for row in sources]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate source identities")

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    model, model_metadata = load_teacher(
        args.checkpoint, args.config_dir / "config.json", args.device
    )
    feature_extractor = build_feature_extractor(args.config_dir)
    max_full_samples = int(round(args.max_full_seconds * 16000.0))
    arrays = {
        "full": np.empty((len(sources), 256), dtype=np.float32),
        "0p5": np.empty((len(sources), 2, 256), dtype=np.float32),
        "1p0": np.empty((len(sources), 2, 256), dtype=np.float32),
    }
    indexed = list(enumerate(sources))
    for duration in ("full", "0p5", "1p0"):
        ordered = sorted(indexed, key=lambda item: int(item[1].get("source_bytes", 0)))
        batch_size = args.batch_full if duration == "full" else args.batch_short
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset:offset + batch_size]
            waveforms = waveforms_for_batch(batch, duration, max_full_samples)
            embeddings = extract_waveforms(
                model, feature_extractor, waveforms, args.device
            )
            if duration == "full":
                if len(batch) != 1:
                    raise RuntimeError("full aggregation requires one source")
                value = np.mean(
                    np.asarray(embeddings, dtype=np.float64), axis=0
                ).astype(np.float32)
                norm = float(np.linalg.norm(value))
                if not np.isfinite(norm) or norm <= 1.0e-12:
                    raise RuntimeError("invalid full-window aggregate")
                arrays[duration][batch[0][0]] = value / np.float32(norm)
            else:
                for local, (index, _source) in enumerate(batch):
                    arrays[duration][index] = embeddings[local * 2:(local + 1) * 2]
            print(
                "{} {} {}/{} elapsed={:.1f}s".format(
                    suite_name, duration, min(offset + len(batch), len(ordered)),
                    len(ordered), time.perf_counter() - started
                ),
                flush=True,
            )

    for key, value in arrays.items():
        norms = np.linalg.norm(value.reshape(-1, 256), axis=1)
        if not np.isfinite(value).all() or float(np.max(np.abs(norms - 1.0))) > 2.0e-4:
            raise RuntimeError("invalid cached embeddings: {}".format(key))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(args.output),
        schema_version=np.asarray(1, dtype=np.int64),
        suite=np.asarray(suite_name),
        utterance_stable_ids=np.asarray(identities),
        source_sha256=np.asarray([str(row["source_sha256"]) for row in sources]),
        full=arrays["full"],
        short05=arrays["0p5"],
        short10=arrays["1p0"],
    )
    cache_sha256 = sha256_file(args.output)
    receipt = {
        "schema_version": 1,
        "artifact_kind": "teacher_true_duration_teacher_cache",
        "suite": suite_name,
        "source_manifest": str(Path(source_manifest).resolve()),
        "source_count": len(sources),
        "max_sources": args.max_sources,
        "dataset_prefix": args.dataset_prefix,
        "crop_views": 2,
        "crop_policy": "deterministic_two_distinct_views_scored_independently",
        "batch_full": args.batch_full,
        "batch_short": args.batch_short,
        "threads": args.threads,
        "device": args.device,
        "max_full_seconds": args.max_full_seconds,
        "long_full_policy": "start_center_end_mean_then_l2",
        "model": model_metadata,
        "cache": str(args.output.resolve().relative_to(ROOT)),
        "cache_bytes": args.output.stat().st_size,
        "cache_sha256": cache_sha256,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
