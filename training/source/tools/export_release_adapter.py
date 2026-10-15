#!/usr/bin/env python3
"""Freeze the gate-passed short_metric short05 research adapter for deployment."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "records/2026-09-10/078_short_metric_joint_anchored_hierarchical_cml"
SOURCE = RECORD / "SHORT_METRIC_SHORT05_ONLY_ADAPTER.npz"
RESULT = RECORD / "SHORT05_ONLY_SEARCH_RESULT.json"
OUTPUT = ROOT / "model/short_duration_metric_adapter.npz"
RECEIPT = RECORD / "SHORT_METRIC_RELEASE_ADAPTER_RECEIPT.json"
SOURCE_SHA256 = "91cc8c72926ae21443b4b84a22fe98bb49bbeb5dc5f5178c060ee180bf100283"
DOMAIN = bytes.fromhex("67616d653832332d7631372d7631322d616e63686f7265642d73686f727430352d736565642d636d6c2d763100")
RELEASE_STATUS = "frozen_release"
EXPECTED_CONFIG = "short05_r8_i0p2"
EXPECTED_SEEDS = (826101, 826102, 826103)
EXPECTED_RANK = 8
EXPECTED_ALPHA = 0.16


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha(arrays):
    digest = hashlib.sha256()
    digest.update(DOMAIN)
    for key in (
        "schema_version", "release_status", "config_id", "seed_ids",
        "rank", "alpha",
    ):
        value = np.asarray(arrays[key])
        digest.update(key.encode("ascii") + b"\x00")
        if value.dtype.kind == "U":
            digest.update(str(value.item()).encode("utf-8") + b"\x00")
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
    for seed_index in range(3):
        for suffix in ("a", "b"):
            key = "seed{}_short05_{}".format(seed_index, suffix)
            digest.update(key.encode("ascii") + b"\x00")
            digest.update(np.ascontiguousarray(arrays[key], dtype="<f4").tobytes())
    return digest.hexdigest()


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main():
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("short_metric selected research adapter SHA-256 mismatch")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if result.get("status") != "RELEASE_GATE_PASS_RUNTIME_PENDING":
        raise RuntimeError("short_metric search release gate did not pass")
    with np.load(str(SOURCE), allow_pickle=False) as source:
        config = str(np.asarray(source["config_id"]).item())
        seeds = tuple(int(item) for item in np.asarray(source["seed_ids"]))
        rank = int(np.asarray(source["rank"]))
        alpha = float(np.asarray(source["alpha"]))
        if (config, seeds, rank, alpha) != (
            EXPECTED_CONFIG, EXPECTED_SEEDS, EXPECTED_RANK, EXPECTED_ALPHA
        ):
            raise RuntimeError("short_metric selected adapter semantic contract mismatch")
        arrays = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "release_status": np.asarray(RELEASE_STATUS),
            "config_id": np.asarray(config),
            "seed_ids": np.asarray(seeds, dtype=np.int64),
            "rank": np.asarray(rank, dtype=np.int64),
            "alpha": np.asarray(alpha, dtype=np.float64),
        }
        for seed_index in range(3):
            a_key = "seed{}_short05_a".format(seed_index)
            b_key = "seed{}_short05_b".format(seed_index)
            a = np.asarray(source[a_key])
            b = np.asarray(source[b_key])
            if a.shape != (rank, 832) or b.shape != (832, rank):
                raise RuntimeError("short_metric factor shape mismatch")
            if a.dtype != np.float32 or b.dtype != np.float32:
                raise RuntimeError("short_metric factor dtype mismatch")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise RuntimeError("short_metric factor contains NaN/Inf")
            arrays[a_key] = np.ascontiguousarray(a).copy()
            arrays[b_key] = np.ascontiguousarray(b).copy()
    semantic = payload_sha(arrays)
    arrays["payload_sha256"] = np.asarray(semantic)
    atomic_npz(OUTPUT, arrays)
    receipt = {
        "schema_version": 1,
        "status": "FROZEN",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_bytes": OUTPUT.stat().st_size,
        "output_sha256": sha256_file(OUTPUT),
        "payload_sha256": semantic,
        "config_id": config,
        "seed_ids": list(seeds),
        "rank": rank,
        "alpha": alpha,
    }
    RECEIPT.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
