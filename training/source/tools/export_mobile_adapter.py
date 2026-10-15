#!/usr/bin/env python3
"""Export the frozen mobile mobile adapter with a semantic digest."""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mobile_weight_refinement import load_mobile


SOURCE = ROOT / "records/2026-09-06/061_mobile_mobile_adapter_mix/MOBILE_ADAPTER.npz"
DESTINATION = ROOT / "model/mobile_kd_adapter.npz"
DOMAIN = bytes.fromhex("67616d653832332d7631332d6d6f62696c652d726564696d2d6f75747075742d6b642d763100")
RELEASE_STATUS = "frozen_proxy_gate_pass_runtime_pending"
CONFIG_ID = "mobile_d42_c42_short05_only"
SEEDS = np.asarray((825001, 825002, 825003), dtype=np.int64)
MATRIX_KEYS = ("lm_short05", "lm_short10", "ptn_short05", "ptn_short10")


def semantic_digest(matrices):
    digest = hashlib.sha256()
    digest.update(DOMAIN)
    digest.update(np.asarray(1, dtype="<i8").tobytes())
    digest.update(RELEASE_STATUS.encode("utf-8") + b"\x00")
    digest.update(CONFIG_ID.encode("ascii") + b"\x00")
    digest.update(SEEDS.astype("<i8", copy=False).tobytes())
    digest.update(np.asarray(1.0, dtype="<f8").tobytes())
    digest.update(np.asarray(0.0, dtype="<f8").tobytes())
    for key in MATRIX_KEYS:
        digest.update(key.encode("ascii") + b"\x00")
        digest.update(np.ascontiguousarray(matrices[key], dtype="<f4").tobytes())
    return digest.hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    nested = load_mobile()
    matrices = {
        model + "_" + duration: nested[model][duration]
        for model in ("lm", "ptn") for duration in ("short05", "short10")
    }
    payload = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "release_status": np.asarray(RELEASE_STATUS),
        "adapter_config_id": np.asarray(CONFIG_ID),
        "seed_ids": SEEDS,
        "mix05": np.asarray(1.0, dtype=np.float64),
        "mix10": np.asarray(0.0, dtype=np.float64),
        "payload_sha256": np.asarray(semantic_digest(matrices)),
    }
    payload.update(matrices)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + DESTINATION.name + ".", suffix=".tmp",
        dir=str(DESTINATION.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(DESTINATION))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    print("path={}".format(DESTINATION.relative_to(ROOT)))
    print("bytes={}".format(DESTINATION.stat().st_size))
    print("sha256={}".format(sha256(DESTINATION)))
    print("payload_sha256={}".format(str(payload["payload_sha256"].item())))


if __name__ == "__main__":
    main()
