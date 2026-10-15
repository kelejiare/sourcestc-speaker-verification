#!/usr/bin/env python3
"""Export the audited joint joint adapter into a strict deployment schema."""

import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "records/2026-09-05/054_joint_joint_adapter_weight_search/JOINT_ADAPTER.npz"
RESULT = ROOT / "records/2026-09-05/054_joint_joint_adapter_weight_search/JOINT_RESULT.json"
OUTPUT = ROOT / "model/joint_kd_adapter.npz"
DOMAIN = bytes.fromhex("67616d653832332d7631322d6a6f696e742d726564696d2d6f75747075742d6b642d763100")
EXPECTED_CONFIG = "d26_h29_c18_a08"
EXPECTED_SEEDS = (825001, 825002, 825003)
EXPECTED_WEIGHTS = {
    "enroll": np.asarray((0.72, 0.252, 0.028), dtype=np.float64),
    "short05": np.asarray((0.32, 0.578, 0.102), dtype=np.float64),
    "short10": np.asarray((0.26, 0.703, 0.037), dtype=np.float64),
}
MATRIX_KEYS = ("lm_short05", "lm_short10", "ptn_short05", "ptn_short10")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha(schema_version, release_status, config_id, seeds, matrices):
    digest = hashlib.sha256()
    digest.update(DOMAIN)
    digest.update(np.asarray(schema_version, dtype="<i8").tobytes())
    digest.update(release_status.encode("utf-8") + b"\x00")
    digest.update(config_id.encode("ascii") + b"\x00")
    digest.update(np.ascontiguousarray(seeds, dtype="<i8").tobytes())
    for key in MATRIX_KEYS:
        digest.update(key.encode("ascii") + b"\x00")
        digest.update(np.ascontiguousarray(matrices[key], dtype="<f4").tobytes())
    return digest.hexdigest()


def main():
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    if report.get("status") != "JOINT_POSTHOC_ROBUST_CANDIDATE_REQUIRES_RUNTIME":
        raise RuntimeError("joint search did not expose the expected audited candidate")
    selected = report["posthoc_all_suite_audit"]["selected"]
    if selected["adapter_config"]["id"] != EXPECTED_CONFIG:
        raise RuntimeError("audited adapter configuration changed")
    for name, expected in EXPECTED_WEIGHTS.items():
        observed = np.asarray(selected["weights"][name], dtype=np.float64)
        if observed.shape != (3,) or not np.array_equal(observed, expected):
            raise RuntimeError("{} block weights changed".format(name))
    if not selected.get("eligible"):
        raise RuntimeError("audited candidate is not eligible")
    checks = selected.get("checks", {})
    if not checks or not all(bool(value) for value in checks.values()):
        raise RuntimeError("audited candidate checks did not all pass")

    with np.load(str(SOURCE), allow_pickle=False) as source:
        config_id = str(np.asarray(source["adapter_config_id"]).item())
        seeds = np.asarray(source["seed_ids"])
        status = str(np.asarray(source["status"]).item())
        if config_id != EXPECTED_CONFIG or tuple(int(item) for item in seeds) != EXPECTED_SEEDS:
            raise RuntimeError("joint adapter identity changed")
        if status != "JOINT_POSTHOC_ROBUST_CANDIDATE_REQUIRES_RUNTIME":
            raise RuntimeError("joint adapter status changed")
        for name, expected in EXPECTED_WEIGHTS.items():
            observed = np.asarray(source[name + "_weights"], dtype=np.float64)
            if not np.array_equal(observed, expected):
                raise RuntimeError("joint adapter {} weights changed".format(name))
        matrices = {}
        for key in MATRIX_KEYS:
            value = np.asarray(source[key])
            if value.shape != (192, 192) or value.dtype != np.float32:
                raise RuntimeError("{} matrix shape/dtype mismatch".format(key))
            if not np.isfinite(value).all():
                raise RuntimeError("{} matrix contains NaN/Inf".format(key))
            matrices[key] = np.ascontiguousarray(value).copy()

    schema_version = 1
    release_status = "audited_posthoc_robust_candidate_runtime_pending"
    digest = payload_sha(
        schema_version, release_status, config_id, seeds, matrices
    )
    arrays = {
        "schema_version": np.asarray(schema_version, dtype=np.int64),
        "release_status": np.asarray(release_status),
        "adapter_config_id": np.asarray(config_id),
        "seed_ids": np.ascontiguousarray(seeds, dtype=np.int64),
        **matrices,
        "payload_sha256": np.asarray(digest),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + OUTPUT.name + ".", suffix=".tmp", dir=str(OUTPUT.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(OUTPUT))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    # Read it back before declaring the binary artifact usable.
    with np.load(str(OUTPUT), allow_pickle=False) as check:
        declared = str(np.asarray(check["payload_sha256"]).item())
        if not hmac.compare_digest(declared, digest):
            raise RuntimeError("deployment adapter readback digest mismatch")
    print(json.dumps({
        "path": str(OUTPUT.relative_to(ROOT)),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256_file(OUTPUT),
        "payload_sha256": digest,
        "config_id": config_id,
        "seed_ids": list(EXPECTED_SEEDS),
        "weights": {key: value.tolist() for key, value in EXPECTED_WEIGHTS.items()},
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
