#!/usr/bin/env python3
"""Strict, bounded-output runtime for joint/mobile competition releases.

Unlike the earlier safe runtime, this module never substitutes another model
or a synthetic vector.  A release either emits every requested embedding from
its declared extractor or exits without publishing ``submit.zip``.
"""

import os
import tempfile
import time
import zipfile
from pathlib import Path

from runtime_io import (
    _extractor_dimension,
    collect_wav_entries,
    embedding_npz_bytes,
    find_input_root,
    find_output_dir,
    validate_embedding,
    validate_written_zip,
)


class InternalFallbackError(RuntimeError):
    """Raised when a candidate silently returns a lower-version embedding."""


def _strict_extract(extractor, wav_path, relative_output, expected_dim):
    result = extractor.extract_file(wav_path)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("strict extractor must return (embedding, metadata)")
    embedding, metadata = result
    if not isinstance(metadata, dict):
        raise TypeError("strict extractor metadata must be a dict")
    fallback_mode = str(metadata.get("fallback_mode", "none"))
    if fallback_mode not in ("", "none"):
        raise InternalFallbackError(
            "{} requested internal fallback {}: {}".format(
                relative_output.as_posix(),
                fallback_mode,
                metadata.get("fallback_reason", ""),
            )
        )
    return validate_embedding(
        embedding, expected_dim, relative_output.as_posix()
    )


def run_strict_inference(
    extractor,
    input_root=None,
    output_dir=None,
    logger=print,
    progress_interval=100,
):
    """Build, read back, and atomically publish a complete result archive."""

    expected_dim = _extractor_dimension(extractor, "strict extractor")
    resolved_input = (
        find_input_root() if input_root is None else Path(input_root).resolve()
    )
    resolved_output = (
        find_output_dir() if output_dir is None else Path(output_dir).resolve()
    )
    entries = collect_wav_entries(resolved_input)
    resolved_output.mkdir(parents=True, exist_ok=True)
    output_zip = resolved_output / "submit.zip"
    if output_zip.exists():
        output_zip.unlink()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".submit.zip.", suffix=".tmp", dir=str(resolved_output)
    )
    os.close(descriptor)
    temporary_zip = Path(temporary_name)
    started = time.perf_counter()
    try:
        with zipfile.ZipFile(
            str(temporary_zip),
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for index, (wav_path, relative_output) in enumerate(entries, 1):
                embedding = _strict_extract(
                    extractor,
                    wav_path,
                    relative_output,
                    expected_dim,
                )
                archive.writestr(
                    relative_output.as_posix(), embedding_npz_bytes(embedding)
                )
                if (
                    index == 1
                    or index == len(entries)
                    or index % max(1, int(progress_interval)) == 0
                ):
                    logger("strict progress: {}/{}".format(index, len(entries)))
        with temporary_zip.open("rb") as handle:
            os.fsync(handle.fileno())
        audit = validate_written_zip(temporary_zip, entries, expected_dim)
        os.replace(str(temporary_zip), str(output_zip))
    except Exception:
        try:
            temporary_zip.unlink()
        except FileNotFoundError:
            pass
        try:
            output_zip.unlink()
        except FileNotFoundError:
            pass
        raise

    report = dict(audit)
    report.update(
        {
            "input_root": str(resolved_input),
            "output_zip": str(output_zip),
            "processed_files": len(entries),
            "fallback_files": 0,
            "fallback_policy": "disabled_fail_closed",
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    logger(
        "strict validation: files={} dim={} bytes={}".format(
            report["file_count"], report["embedding_dim"], report["zip_bytes"]
        )
    )
    return report
