#!/usr/bin/env python3
"""Build the frozen ReDimNet2+ cache without running an oracle score search."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio_features import load_audio
from tools.build_embedding_cache import deterministic_crop_start
from tools.nap_external_diagnostics import SUITES, load_suite
from tools.nap_lodo import atomic_json, sha256_file
from tools.w24_external_support import atomic_npz, shard_path, validate_shard
from tools.evaluate_redimnet2_mobile_acoustic_expert import (
    embed_waveforms,
    load_candidate,
)


CHECKPOINT = ROOT / "model/redimnet2plus-b6-lmft-robust.pt"
OUTPUT_ROOT = (
    ROOT
    / "records/2026-09-11/086_mobile_expert_redimnet2_mobile_acoustic_expert"
    / "plus_redimnet2plus_lmft"
)


def build_batch(model, batch, cache_dir):
    loaded = []
    for source in batch:
        waveform = load_audio(
            ROOT / source["source_path"], 16000, remove_dc=False
        )
        if waveform.size < 16000:
            raise RuntimeError(
                "source shorter than 1 second: {}".format(source["source_path"])
            )
        crops05 = []
        crops10 = []
        for view in (0, 1):
            start05 = deterministic_crop_start(waveform.size, 8000, view, 2)
            start10 = deterministic_crop_start(waveform.size, 16000, view, 2)
            crops05.append(
                np.ascontiguousarray(
                    waveform[start05:start05 + 8000], dtype=np.float32
                )
            )
            crops10.append(
                np.ascontiguousarray(
                    waveform[start10:start10 + 16000], dtype=np.float32
                )
            )
        loaded.append((source, waveform, crops05, crops10))

    short05 = embed_waveforms(
        model, [value for item in loaded for value in item[2]], None
    )
    # The frozen plus policy assigns zero auxiliary weight to 1.0-second
    # queries.  Preserve the cache schema with a normalized placeholder copied
    # from the 0.5-second views instead of spending a third model pass on a
    # score term that is mathematically multiplied by zero.
    short10 = short05.copy()
    for index, (source, waveform, _crops05, _crops10) in enumerate(loaded):
        identifier = str(source["utterance_stable_id"])
        arrays = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "utterance_id": np.asarray(identifier),
            "source_sha256": np.asarray(str(source["source_sha256"])),
            "full": embed_waveforms(model, [waveform], None)[0],
            "0p5": short05[index * 2:(index + 1) * 2],
            "1p0": short10[index * 2:(index + 1) * 2],
        }
        path = shard_path(cache_dir, identifier)
        atomic_npz(path, arrays)
        validate_shard(path, source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("new_holdouts", "alimeeting_test"), required=True
    )
    parser.add_argument("--batch-sources", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_sources <= 0:
        raise ValueError("batch-sources must be positive")

    suite = SUITES[0] if args.suite == "new_holdouts" else SUITES[1]
    sources, _trials, _joint_cache = load_suite(suite)
    cache_dir = OUTPUT_ROOT / (suite["name"] + "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.threads)))
    model, tensor_count, parameter_count = load_candidate(CHECKPOINT)
    pending = [
        source
        for source in sources
        if not shard_path(
            cache_dir, str(source["utterance_stable_id"])
        ).is_file()
    ]
    started = time.perf_counter()
    for offset in range(0, len(pending), args.batch_sources):
        batch = pending[offset:offset + args.batch_sources]
        build_batch(model, batch, cache_dir)
        completed = min(offset + len(batch), len(pending))
        print(
            "{} cache {}/{} new; reused={}; elapsed={:.1f}s".format(
                suite["name"], completed, len(pending),
                len(sources) - len(pending), time.perf_counter() - started,
            ),
            flush=True,
        )

    receipt = {
        "schema_version": 1,
        "artifact_kind": "plus_redimnet2plus_cache_receipt",
        "suite": suite["name"],
        "source_count": len(sources),
        "new_count": len(pending),
        "cache_dir": str(cache_dir.relative_to(ROOT)),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "batch_sources": args.batch_sources,
        "threads": args.threads,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = OUTPUT_ROOT / (suite["name"] + "_CACHE_RECEIPT.json")
    atomic_json(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
