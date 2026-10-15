#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed output runtime for the Game823 distillation extractor.

The model implementation is deliberately injected instead of imported here.
Both the primary distillation extractor and the online-proven acoustic extractor implement:

    output_dim: int
    extract_file(path: pathlib.Path) -> (numpy.ndarray, metadata_dict)

This keeps model/training work independent from the platform I/O contract.
For each WAV, an invalid or failed distillation result first falls back to the exact acoustic
extractor.  A deterministic unit vector is used only if the WAV is so malformed
that both extractors fail.  The complete temporary ZIP is reopened and audited
before an atomic ``os.replace`` publishes ``submit.zip``.
"""

import hashlib
import io
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCENERY_PATTERN = re.compile(r"^data_scenery\d+$")
INPUT_ENV_NAMES = ("GAME823_INPUT_DIR", "AIOJ_INPUT_DIR", "INPUT_DIR")
OUTPUT_ENV_NAMES = ("AIOJ_OUTPUT_DIR", "OUTPUT_DIR")
DEFAULT_INPUT_CANDIDATES = (
    Path("/work/data/sourcestc-test"),
    Path("/work/data"),
    Path("/work/dataset"),
    Path("/data/input"),
    Path("/data"),
    Path("/work/data/input"),
    Path("/input"),
    Path("/dataset"),
    Path("/datasets"),
    Path("/mnt/data"),
    Path("/workspace/data"),
    Path("/workspace/input"),
    Path("/opt/ml/input/data"),
)
MAX_EMBEDDING_DIM = 4096
MAX_NPZ_BYTES = 16 * 1024 * 1024
UNIT_NORM_TOLERANCE = 1.0e-4


class InputContractError(ValueError):
    """Raised when the input tree cannot be mapped to the official contract."""


class OutputContractError(ValueError):
    """Raised before publishing a malformed result archive."""


def _is_wav(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".wav"


def _canonical_parts(relative_parts: Sequence[str]) -> bool:
    return (
        len(relative_parts) >= 3
        and SCENERY_PATTERN.fullmatch(relative_parts[0]) is not None
        and relative_parts[1] == "audio"
    )


def canonical_output_path(wav_path: Path, input_root: Path) -> Path:
    """Map exactly ``data_sceneryN/audio/**/*.wav`` to an NPZ member."""

    root = input_root.expanduser().resolve()
    lexical_path = wav_path.expanduser()
    try:
        relative = lexical_path.relative_to(root)
    except ValueError:
        # The path may be relative or unresolved; resolve once and retry.  The
        # containment check below still rejects symlink escapes.
        relative = lexical_path.resolve().relative_to(root)
    if not _canonical_parts(relative.parts):
        raise InputContractError(
            "WAV path must match data_sceneryN/audio/**/*.wav: {}".format(
                wav_path
            )
        )
    resolved = lexical_path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise InputContractError("WAV escapes input root: {}".format(wav_path))
    return relative.with_suffix(".npz")


def _roots_below_candidate(candidate: Path) -> List[Path]:
    """Return logical roots that contain canonical scenery trees."""

    candidate = candidate.expanduser().resolve()
    if not candidate.is_dir():
        return []
    roots = set()
    try:
        paths = candidate.rglob("*")
        for path in paths:
            if not _is_wav(path):
                continue
            relative = path.relative_to(candidate)
            parts = relative.parts
            for index in range(max(0, len(parts) - 2)):
                if (
                    SCENERY_PATTERN.fullmatch(parts[index]) is not None
                    and index + 1 < len(parts)
                    and parts[index + 1] == "audio"
                    and index + 2 < len(parts)
                ):
                    roots.add((candidate / Path(*parts[:index])).resolve())
                    break
    except (OSError, PermissionError) as exc:
        raise InputContractError(
            "cannot inspect input candidate {}: {}".format(candidate, exc)
        )
    return sorted(roots, key=lambda item: str(item))


def find_input_root(
    environ: Optional[Dict[str, str]] = None,
    candidates: Optional[Sequence[Path]] = None,
) -> Path:
    """Discover one unambiguous official-format input root.

    Environment paths are authoritative: a configured but invalid or
    ambiguous path is an error, rather than silently reading another tree.
    Automatic candidates are tried in order and the first unambiguous logical
    root is returned.
    """

    environment = os.environ if environ is None else environ
    for name in INPUT_ENV_NAMES:
        value = environment.get(name)
        if not value:
            continue
        configured = Path(value)
        roots = _roots_below_candidate(configured)
        if len(roots) != 1:
            raise InputContractError(
                "{} must contain exactly one canonical input root; found {}: {}"
                .format(name, len(roots), [str(path) for path in roots])
            )
        # Do a full strict collection now, before loading any model.
        collect_wav_entries(roots[0])
        return roots[0]

    checked = []
    for raw_candidate in DEFAULT_INPUT_CANDIDATES if candidates is None else candidates:
        candidate = Path(raw_candidate).expanduser()
        checked.append(str(candidate))
        roots = _roots_below_candidate(candidate)
        if not roots:
            continue
        if len(roots) != 1:
            raise InputContractError(
                "ambiguous input candidate {} contains roots {}".format(
                    candidate, [str(path) for path in roots]
                )
            )
        collect_wav_entries(roots[0])
        return roots[0]
    raise FileNotFoundError(
        "no canonical data_sceneryN/audio WAV tree found; checked {}. "
        "Set GAME823_INPUT_DIR, AIOJ_INPUT_DIR, or INPUT_DIR.".format(checked)
    )


def find_output_dir(environ: Optional[Dict[str, str]] = None) -> Path:
    environment = os.environ if environ is None else environ
    for name in OUTPUT_ENV_NAMES:
        value = environment.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return Path("/work/output")


def _find_required_file(project_dir: Path, relative_names: Sequence[str]) -> Path:
    candidates = [project_dir / name for name in relative_names]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "required acoustic resource not found; checked {}".format(
            [str(path) for path in candidates]
        )
    )


def build_online_acoustic_fallback(
    project_dir: Path,
    device_name: str = "auto",
    max_chunk_seconds: float = 60.0,
    chunk_hop_seconds: float = 30.0,
) -> Any:
    """Instantiate the exact 88.41131 online acoustic numerical baseline.

    The assertions intentionally fail closed if the checked-in acoustic policy is
    edited.  A distillation package must include ``ensemble_inference.py`` and the same
    PT/CAM++ checkpoints used by acoustic.
    """

    from ensemble_inference import DurationAdaptiveEnsembleExtractor

    assert_online_acoustic_contract(DurationAdaptiveEnsembleExtractor)
    base = Path(project_dir).expanduser().resolve()
    pt_model = _find_required_file(
        base, ("model/model_pt.pt", "model_pt.pt")
    )
    campplus_model = _find_required_file(
        base,
        (
            "model/campplus_cn_en_common.pt",
            "campplus_cn_en_common.pt",
        ),
    )
    return DurationAdaptiveEnsembleExtractor(
        pt_model_path=pt_model,
        campplus_model_path=campplus_model,
        device_name=device_name,
        max_chunk_seconds=max_chunk_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
    )


def assert_online_acoustic_contract(extractor_or_class: Any) -> None:
    """Fail if an alleged acoustic fallback differs from the online-best policy."""

    expected_policy = (448, 0.78, 0.5376047904191618, 1.0)
    try:
        actual_policy = (
            int(extractor_or_class.output_dim),
            float(extractor_or_class.long_pt_weight),
            float(extractor_or_class.short_0p5_pt_weight),
            float(extractor_or_class.short_1p0_pt_weight),
        )
    except Exception as exc:
        raise RuntimeError("invalid acoustic fallback contract: {}".format(exc))
    if actual_policy != expected_policy:
        raise RuntimeError(
            "acoustic fallback policy drift: expected {}, got {}".format(
                expected_policy, actual_policy
            )
        )


def collect_wav_entries(input_root: Path) -> List[Tuple[Path, Path]]:
    """Collect every WAV and reject noncanonical or duplicate mappings."""

    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise InputContractError("input root is not a directory: {}".format(root))
    entries = []
    seen = set()
    invalid = []
    try:
        wav_paths = sorted(
            (path for path in root.rglob("*") if _is_wav(path)),
            key=lambda item: item.as_posix(),
        )
    except (OSError, PermissionError) as exc:
        raise InputContractError("cannot enumerate {}: {}".format(root, exc))
    for wav_path in wav_paths:
        try:
            relative_output = canonical_output_path(wav_path, root)
        except InputContractError:
            invalid.append(str(wav_path))
            continue
        output_name = relative_output.as_posix()
        if output_name in seen:
            raise InputContractError(
                "duplicate WAV-to-NPZ mapping: {}".format(output_name)
            )
        seen.add(output_name)
        entries.append((wav_path, relative_output))
    if invalid:
        raise InputContractError(
            "noncanonical WAV files found under input root: {}".format(invalid)
        )
    if not entries:
        raise InputContractError("input root contains no canonical WAV files")
    return entries


def validate_embedding(
    embedding: Any,
    expected_dim: int,
    label: str = "embedding",
) -> np.ndarray:
    """Enforce the exact platform vector contract without silent coercion."""

    if expected_dim <= 0 or expected_dim > MAX_EMBEDDING_DIM:
        raise OutputContractError(
            "invalid expected embedding dimension: {}".format(expected_dim)
        )
    vector = np.asarray(embedding)
    if vector.dtype != np.float32:
        raise OutputContractError(
            "{} dtype must be float32, got {}".format(label, vector.dtype)
        )
    if vector.ndim != 1 or vector.size != expected_dim:
        raise OutputContractError(
            "{} shape must be ({},), got {}".format(
                label, expected_dim, vector.shape
            )
        )
    if not np.isfinite(vector).all():
        raise OutputContractError("{} contains NaN or Inf".format(label))
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise OutputContractError("{} has zero norm".format(label))
    if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
        raise OutputContractError(
            "{} norm must be 1 within {}, got {}".format(
                label, UNIT_NORM_TOLERANCE, norm
            )
        )
    return np.ascontiguousarray(vector)


def embedding_npz_bytes(embedding: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, embedding=embedding)
    payload = buffer.getvalue()
    if len(payload) > MAX_NPZ_BYTES:
        raise OutputContractError("one NPZ exceeds 16 MiB")
    return payload


def emergency_embedding(relative_output: Path, expected_dim: int) -> np.ndarray:
    """Generate a deterministic valid vector only after distillation and acoustic fail."""

    if expected_dim <= 0 or expected_dim > MAX_EMBEDDING_DIM:
        raise OutputContractError(
            "invalid emergency embedding dimension: {}".format(expected_dim)
        )
    digest = hashlib.sha256(relative_output.as_posix().encode("utf-8")).digest()
    raw = np.resize(np.frombuffer(digest, dtype=np.uint8), expected_dim)
    vector = (raw.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm < 1.0e-12:
        raise OutputContractError("cannot construct emergency embedding")
    vector = (vector / np.float32(norm)).astype(np.float32)
    return validate_embedding(vector, expected_dim, "emergency embedding")


def _unpack_extractor_result(result: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        embedding, metadata = result
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise TypeError("extractor metadata must be a dict")
        return embedding, dict(metadata)
    return result, {}


def safe_extract_file(
    wav_path: Path,
    relative_output: Path,
    primary_extractor: Any,
    acoustic_extractor: Any,
    expected_dim: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Try distillation, then exact acoustic, then a format-only emergency vector."""

    failures = []
    for mode, extractor in (("distillation", primary_extractor), ("acoustic", acoustic_extractor)):
        try:
            raw_embedding, metadata = _unpack_extractor_result(
                extractor.extract_file(wav_path)
            )
            embedding = validate_embedding(
                raw_embedding, expected_dim, "{} output".format(mode)
            )
            internal_fallback = str(metadata.get("fallback_mode", "none"))
            if mode == "distillation" and internal_fallback not in ("", "none"):
                # The distillation adapter deliberately returns its exact acoustic base when
                # adapter loading or a per-file transform fails.
                metadata["primary_fallback_mode"] = internal_fallback
                metadata["fallback_mode"] = "acoustic"
            else:
                metadata["fallback_mode"] = "none" if mode == "distillation" else "acoustic"
            if failures:
                metadata["fallback_reason"] = failures[-1]
            return embedding, metadata
        except Exception as exc:  # Per-file containment is intentional here.
            failures.append("{}: {}: {}".format(mode, type(exc).__name__, exc))
    embedding = emergency_embedding(relative_output, expected_dim)
    return embedding, {
        "fallback_mode": "emergency",
        "fallback_reason": " | ".join(failures),
    }


def _expected_member_names(
    entries: Iterable[Tuple[Path, Path]],
) -> List[str]:
    names = [relative.as_posix() for _, relative in entries]
    if len(names) != len(set(names)):
        raise OutputContractError("expected output contains duplicate members")
    return names


def validate_written_zip(
    submit_zip: Path,
    entries: Sequence[Tuple[Path, Path]],
    expected_dim: int,
) -> Dict[str, Any]:
    """Reopen ZIP and audit CRC, members, NPZ schema, values, and norms."""

    expected_names = _expected_member_names(entries)
    expected = set(expected_names)
    norms = []
    with zipfile.ZipFile(str(submit_zip), "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise OutputContractError(
                "ZIP CRC validation failed: {}".format(bad_member)
            )
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise OutputContractError("ZIP contains duplicate member names")
        actual = set(names)
        if actual != expected:
            raise OutputContractError(
                "ZIP/WAV mismatch: missing={} extra={}".format(
                    sorted(expected - actual), sorted(actual - expected)
                )
            )
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise OutputContractError("unsafe ZIP member name: {}".format(name))
            info = archive.getinfo(name)
            if info.is_dir():
                raise OutputContractError("unexpected ZIP directory: {}".format(name))
            if info.file_size > MAX_NPZ_BYTES:
                raise OutputContractError("NPZ exceeds 16 MiB: {}".format(name))
            payload = archive.read(name)
            try:
                with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                    if data.files != ["embedding"]:
                        raise OutputContractError(
                            "NPZ key must be exactly embedding: {} -> {}".format(
                                name, data.files
                            )
                        )
                    embedding = data["embedding"]
            except OutputContractError:
                raise
            except Exception as exc:
                raise OutputContractError(
                    "cannot read NPZ {}: {}".format(name, exc)
                )
            embedding = validate_embedding(embedding, expected_dim, name)
            norms.append(float(np.linalg.norm(embedding)))
    if not norms:
        raise OutputContractError("ZIP contains no embeddings")
    return {
        "status": "PASS",
        "file_count": len(expected),
        "embedding_dim": expected_dim,
        "norm_min": min(norms),
        "norm_max": max(norms),
        "zip_bytes": submit_zip.stat().st_size,
    }


def _extractor_dimension(extractor: Any, label: str) -> int:
    try:
        value = int(extractor.output_dim)
    except Exception:
        raise OutputContractError("{} must expose integer output_dim".format(label))
    if value <= 0 or value > MAX_EMBEDDING_DIM:
        raise OutputContractError(
            "{} output_dim is invalid: {}".format(label, value)
        )
    return value


def run_safe_inference(
    primary_extractor: Any,
    acoustic_extractor: Any,
    input_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    logger: Any = print,
) -> Dict[str, Any]:
    """Create and atomically publish a fully validated ``submit.zip``."""

    primary_dim = _extractor_dimension(primary_extractor, "distillation extractor")
    acoustic_dim = _extractor_dimension(acoustic_extractor, "acoustic extractor")
    if primary_dim != acoustic_dim:
        raise OutputContractError(
            "distillation/acoustic dimensions must match for numerical fallback: {} != {}"
            .format(primary_dim, acoustic_dim)
        )
    resolved_input = find_input_root() if input_root is None else Path(input_root).resolve()
    entries = collect_wav_entries(resolved_input)
    resolved_output = find_output_dir() if output_dir is None else Path(output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    output_zip = resolved_output / "submit.zip"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".submit.zip.", suffix=".tmp", dir=str(resolved_output)
    )
    os.close(descriptor)
    temporary_zip = Path(temporary_name)
    # ZipFile expects either a new/empty archive in write mode.  Keeping the
    # mkstemp reservation in the same directory guarantees same-filesystem
    # atomic replace without a predictable temp filename race.
    counters = {"distillation": 0, "acoustic": 0, "emergency": 0}
    fallback_files = []
    started = time.perf_counter()
    try:
        with zipfile.ZipFile(
            str(temporary_zip),
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for index, (wav_path, relative_output) in enumerate(entries, 1):
                file_started = time.perf_counter()
                embedding, metadata = safe_extract_file(
                    wav_path,
                    relative_output,
                    primary_extractor,
                    acoustic_extractor,
                    primary_dim,
                )
                mode = metadata.get("fallback_mode", "none")
                counter_name = "distillation" if mode == "none" else mode
                counters[counter_name] += 1
                if mode != "none":
                    fallback_files.append(
                        {
                            "wav": wav_path.relative_to(resolved_input).as_posix(),
                            "mode": str(mode),
                            "reason": str(metadata.get("fallback_reason", "")),
                            "primary_mode": str(
                                metadata.get("primary_fallback_mode", "")
                            ),
                        }
                    )
                archive.writestr(
                    relative_output.as_posix(), embedding_npz_bytes(embedding)
                )
                logger(
                    "[{}/{}] {} output={} fallback={} elapsed={:.3f}s".format(
                        index,
                        len(entries),
                        wav_path.name,
                        relative_output.as_posix(),
                        mode,
                        time.perf_counter() - file_started,
                    )
                )
        # Flush archive bytes before readback.  Validation includes CRC and an
        # actual load of every compressed NPZ payload.
        with temporary_zip.open("rb") as handle:
            os.fsync(handle.fileno())
        audit = validate_written_zip(temporary_zip, entries, primary_dim)
        os.replace(str(temporary_zip), str(output_zip))
        # Best-effort directory fsync makes the rename durable on Linux/macOS.
        try:
            directory_fd = os.open(str(resolved_output), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        try:
            temporary_zip.unlink()
        except FileNotFoundError:
            pass
        raise
    report = dict(audit)
    report.update(
        {
            "input_root": str(resolved_input),
            "output_zip": str(output_zip),
            "fallback_counts": counters,
            "fallback_files": fallback_files,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    logger("validated output: {}".format(report))
    return report
