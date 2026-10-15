#!/usr/bin/env python3
"""Strict batch runtime with unsupervised teacher-block mean adaptation.

All WAV files are first embedded by the unchanged calibrated_fusion extractor.  Within each
official source/duration group, only the final normalized 256-D teacher block
is mean centered and re-normalized.  No labels, trials, filenames, scores or
speaker identities are used.  The ZIP is published atomically only after a
complete read-back validation.
"""
import math
import os
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from strict_submission_runtime import InternalFallbackError
from runtime_io import (
    _extractor_dimension,
    collect_wav_entries,
    embedding_npz_bytes,
    find_input_root,
    find_output_dir,
    validate_embedding,
    validate_written_zip,
)


TEACHER_DIM = 256
SOURCE_LAMBDA = {
    "data_scenery1_mobile": 1.05,
    "data_scenery2_joint": 1.00,
}
MINIMUM_GROUP_SIZE = 8


def _extract(extractor, wav_path, relative_output, expected_dim):
    result = extractor.extract_file(wav_path)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("batch_centering extractor must return (embedding, metadata)")
    embedding, metadata = result
    if not isinstance(metadata, dict):
        raise TypeError("batch_centering extractor metadata must be a dict")
    fallback = str(metadata.get("fallback_mode", "none"))
    if fallback not in ("", "none"):
        raise InternalFallbackError(
            "{} requested fallback {}".format(relative_output.as_posix(), fallback)
        )
    value = validate_embedding(embedding, expected_dim, relative_output.as_posix())
    return value, metadata


def _adapt(values, metadata):
    output = [value.copy() for value in values]
    groups = defaultdict(list)
    for index, item in enumerate(metadata):
        route = str(item.get("teacher_fusion_route", ""))
        policy = str(item.get("teacher_fusion_policy", ""))
        alpha = float(item.get("teacher_fusion_alpha", 0.0))
        active = bool(item.get("teacher_fusion_teacher_active", False))
        if active:
            if route not in SOURCE_LAMBDA or not (0.0 < alpha < 1.0):
                raise RuntimeError("batch_centering active teacher metadata contract failed")
            groups[(route, policy)].append((index, alpha))

    audits = []
    for (route, policy), rows in sorted(groups.items()):
        indices = [row[0] for row in rows]
        alphas = np.asarray([row[1] for row in rows], dtype=np.float64)
        if np.max(np.abs(alphas - alphas[0])) > 1.0e-12:
            raise RuntimeError("batch_centering group contains inconsistent alpha values")
        teacher = np.stack([
            values[index][-TEACHER_DIM:] / np.float32(math.sqrt(alphas[0]))
            for index in indices
        ]).astype(np.float32)
        norms = np.linalg.norm(teacher, axis=1)
        if not np.isfinite(teacher).all() or np.max(np.abs(norms - 1.0)) > 2.0e-4:
            raise RuntimeError("batch_centering teacher block recovery failed")
        applied = len(indices) >= MINIMUM_GROUP_SIZE
        mean = teacher.mean(axis=0, keepdims=True)
        if applied:
            centered = teacher - np.float32(SOURCE_LAMBDA[route]) * mean
            centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1.0e-8)
            scale = np.float32(math.sqrt(alphas[0]))
            for local, index in enumerate(indices):
                candidate = values[index].copy()
                candidate[-TEACHER_DIM:] = scale * centered[local]
                output[index] = validate_embedding(candidate, len(candidate), "batch_centering adapted embedding")
        audits.append({
            "route": route,
            "policy": policy,
            "files": len(indices),
            "alpha": float(alphas[0]),
            "lambda": float(SOURCE_LAMBDA[route]),
            "mean_norm": float(np.linalg.norm(mean)),
            "applied": applied,
        })
    return output, audits


def run_batch_centering_inference(extractor, input_root=None, output_dir=None, logger=print, progress_interval=100):
    expected_dim = _extractor_dimension(extractor, "batch_centering extractor")
    resolved_input = find_input_root() if input_root is None else Path(input_root).resolve()
    resolved_output = find_output_dir() if output_dir is None else Path(output_dir).resolve()
    entries = collect_wav_entries(resolved_input)
    values, metadata = [], []
    started = time.perf_counter()
    for index, (wav_path, relative_output) in enumerate(entries, 1):
        value, item = _extract(extractor, wav_path, relative_output, expected_dim)
        values.append(value); metadata.append(item)
        if index == 1 or index == len(entries) or index % max(1, int(progress_interval)) == 0:
            logger("batch_centering extraction: {}/{}".format(index, len(entries)))
    values, groups = _adapt(values, metadata)

    resolved_output.mkdir(parents=True, exist_ok=True)
    output_zip = resolved_output / "submit.zip"
    if output_zip.exists():
        output_zip.unlink()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".submit.zip.", suffix=".tmp", dir=str(resolved_output)
    )
    os.close(descriptor)
    temporary_zip = Path(temporary_name)
    try:
        with zipfile.ZipFile(str(temporary_zip), "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for (_, relative_output), value in zip(entries, values):
                archive.writestr(relative_output.as_posix(), embedding_npz_bytes(value))
        with temporary_zip.open("rb") as handle:
            os.fsync(handle.fileno())
        audit = validate_written_zip(temporary_zip, entries, expected_dim)
        os.replace(str(temporary_zip), str(output_zip))
    except Exception:
        temporary_zip.unlink(missing_ok=True)
        output_zip.unlink(missing_ok=True)
        raise
    report = dict(audit)
    report.update({
        "input_root": str(resolved_input), "output_zip": str(output_zip),
        "processed_files": len(entries), "fallback_files": 0,
        "fallback_policy": "disabled_fail_closed",
        "adaptation": "unlabeled_teacher_block_group_mean",
        "minimum_group_size": MINIMUM_GROUP_SIZE,
        "groups": groups,
        "elapsed_seconds": time.perf_counter() - started,
    })
    logger("batch_centering validation: files={} dim={} bytes={}".format(
        report["file_count"], report["embedding_dim"], report["zip_bytes"]
    ))
    return report
