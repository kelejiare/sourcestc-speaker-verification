#!/usr/bin/env python3
"""Build a resumable schema-v2 cache of exact acoustic duration views.

The builder consumes one or more canonical distillation manifests.  It freezes the
online-validated acoustic PT+CAM++ extractor, computes one long-policy teacher per
source utterance, and computes deterministic 0.5/1.0 second crop views.  Each
view is stored atomically as a small shard before the final strict NPZ is
assembled, so an interrupted CPU run resumes without recomputing completed
audio.

This is an offline data tool.  It is never included in the submission tar.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from audio_features import load_audio
from ensemble_inference import DurationAdaptiveEnsembleExtractor
from speaker_inference import l2_normalize
from distillation_training.cache_schema import (
    CACHE_SCHEMA_VERSION,
    EMBEDDING_DIM,
    write_embedding_cache,
)
from distillation_build_manifest import read_manifest


SHARD_SCHEMA_VERSION = 1
DEFAULT_SELECTION_SEED = 82311
DEFAULT_CROP_VIEWS = 2
DEFAULT_MAX_UTTERANCES_PER_SPEAKER = 6
DEFAULT_MINIMUM_UTTERANCES_PER_SPEAKER = 2
SHARD_KEYS = (
    "schema_version",
    "embedding_id",
    "utterance_id",
    "speaker_id",
    "dataset",
    "split",
    "source_path",
    "source_size",
    "crop_view",
    "crop05_start",
    "crop10_start",
    "full",
    "short05",
    "short10",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _stable_rank(record: Mapping[str, str], seed: int) -> str:
    text = "\0".join(
        (
            str(seed),
            record["dataset"],
            record["speaker_id"],
            record["split"],
            record["path"],
        )
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_records(
    records: Sequence[Mapping[str, str]],
    included_splits: Sequence[str],
    maximum_per_speaker: int,
    minimum_per_speaker: int,
    selection_seed: int,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    """Select a deterministic, speaker-stratified subset."""

    included = set(str(value) for value in included_splits)
    if not included or not included.issubset({"train", "dev", "test"}):
        raise ValueError("included_splits 必须是 train/dev/test 的非空子集")
    if maximum_per_speaker < 1 or minimum_per_speaker < 1:
        raise ValueError("每 speaker 的最大/最小 utterance 数必须为正数")
    if minimum_per_speaker > maximum_per_speaker:
        raise ValueError("minimum_per_speaker 不能大于 maximum_per_speaker")

    groups = defaultdict(list)  # type: Dict[Tuple[str, str, str], List[Dict[str, str]]]
    rejected_non_wav = []
    for original in records:
        record = dict(original)
        if record["split"] not in included:
            continue
        if Path(record["path"]).suffix.lower() != ".wav":
            rejected_non_wav.append(record["path"])
            continue
        groups[(record["dataset"], record["speaker_id"], record["split"])].append(
            record
        )

    selected = []
    excluded_small_groups = []
    group_counts = {}
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda row: _stable_rank(row, selection_seed))
        if len(ranked) < minimum_per_speaker:
            excluded_small_groups.append(
                {"dataset": key[0], "speaker_id": key[1], "split": key[2], "count": len(ranked)}
            )
            continue
        chosen = ranked[:maximum_per_speaker]
        selected.extend(chosen)
        group_counts["{}|{}|{}".format(*key)] = {
            "available": len(ranked),
            "selected": len(chosen),
        }

    selected.sort(
        key=lambda row: (
            row["dataset"], row["split"], row["speaker_id"], row["path"]
        )
    )
    if not selected:
        raise ValueError("筛选后没有 WAV 记录")

    utterance_ids = [
        "{}:{}".format(record["dataset"], Path(record["path"]).stem)
        for record in selected
    ]
    duplicates = sorted(
        value for value in set(utterance_ids) if utterance_ids.count(value) > 1
    )
    if duplicates:
        raise ValueError("utterance_id 不唯一：{}".format(duplicates[:5]))

    audit = {
        "input_record_count": len(records),
        "selected_utterance_count": len(selected),
        "selected_speaker_count": len({row["speaker_id"] for row in selected}),
        "selected_dataset_count": len({row["dataset"] for row in selected}),
        "selected_by_dataset_split": {},
        "excluded_small_groups": excluded_small_groups,
        "rejected_non_wav_count": len(rejected_non_wav),
        "rejected_non_wav_examples": rejected_non_wav[:10],
        "group_counts": group_counts,
    }
    counts = defaultdict(int)
    for row in selected:
        counts[(row["dataset"], row["split"])] += 1
    audit["selected_by_dataset_split"] = {
        "{}|{}".format(*key): value for key, value in sorted(counts.items())
    }
    return selected, audit


def deterministic_crop_start(
    sample_count: int, crop_count: int, view_index: int, view_count: int
) -> int:
    if crop_count <= 0 or sample_count < crop_count:
        raise ValueError(
            "音频不足以裁剪：samples={} crop={}".format(sample_count, crop_count)
        )
    if view_count <= 0 or not 0 <= view_index < view_count:
        raise ValueError("crop view 索引错误")
    maximum = sample_count - crop_count
    if maximum == 0 or view_count == 1:
        return maximum // 2
    fraction = float(view_index) / float(view_count - 1)
    return int(round(fraction * maximum))


def extract_acoustic_waveform(
    extractor: DurationAdaptiveEnsembleExtractor,
    waveform: np.ndarray,
    pt_weight: float,
) -> np.ndarray:
    """Run the exact acoustic acoustic path with an explicit duration policy."""

    waveform = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform 必须是一维非空 float32")
    pt_embedding, _ = extractor.pt_extractor._embed_segments(
        extractor.pt_extractor._segments(waveform), "VoxBlink2 PT 分支"
    )
    if pt_weight >= 1.0:
        embedding = np.concatenate(
            (
                pt_embedding,
                np.zeros(extractor.campplus_dim, dtype=np.float32),
            )
        ).astype(np.float32)
    else:
        campplus_embedding, _ = extractor._campplus_embedding(waveform)
        embedding = extractor._fuse(pt_embedding, campplus_embedding, pt_weight)
    embedding = l2_normalize(embedding, "distillation cache acoustic embedding")
    if embedding.shape != (EMBEDDING_DIM,) or embedding.dtype != np.float32:
        raise RuntimeError(
            "acoustic cache embedding 契约错误：shape={} dtype={}".format(
                embedding.shape, embedding.dtype
            )
        )
    return embedding


def _strict_scalar(payload: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(payload[name])
    if value.shape != ():
        raise ValueError("shard {} 不是标量".format(name))
    return str(value.item())


def validate_shard(
    path: Path,
    expected: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    if tuple(sorted(payload)) != tuple(sorted(SHARD_KEYS)):
        raise ValueError("shard schema keys 错误：{}".format(path))
    schema = np.asarray(payload["schema_version"])
    if schema.shape != () or schema.dtype != np.int64 or int(schema.item()) != SHARD_SCHEMA_VERSION:
        raise ValueError("shard schema_version 错误：{}".format(path))
    for key in ("embedding_id", "utterance_id", "speaker_id", "dataset", "split", "source_path"):
        if _strict_scalar(payload, key) != str(expected[key]):
            raise ValueError("shard {} 元数据不匹配：{}".format(key, path))
    for key in ("source_size", "crop_view", "crop05_start", "crop10_start"):
        value = np.asarray(payload[key])
        if value.shape != () or value.dtype != np.int64 or int(value.item()) != int(expected[key]):
            raise ValueError("shard {} 整数元数据不匹配：{}".format(key, path))
    for key in ("full", "short05", "short10"):
        value = np.asarray(payload[key])
        if value.shape != (EMBEDDING_DIM,) or value.dtype != np.float32:
            raise ValueError("shard {} embedding 契约错误：{}".format(key, path))
        if not np.isfinite(value).all() or abs(float(np.linalg.norm(value)) - 1.0) > 2.0e-4:
            raise ValueError("shard {} embedding 数值错误：{}".format(key, path))
    return payload


def _shard_path(work_dir: Path, embedding_id: str) -> Path:
    digest = hashlib.sha256(embedding_id.encode("utf-8")).hexdigest()
    return work_dir / "items" / (digest + ".npz")


def build_expected_rows(
    records: Sequence[Mapping[str, str]], crop_views: int
) -> List[Dict[str, object]]:
    rows = []
    for record in records:
        source = Path(record["path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError("manifest WAV 不存在：{}".format(source))
        utterance_id = "{}:{}".format(record["dataset"], source.stem)
        source_size = source.stat().st_size
        for view in range(crop_views):
            rows.append(
                {
                    "embedding_id": "{}@view{:02d}".format(utterance_id, view),
                    "utterance_id": utterance_id,
                    "speaker_id": record["speaker_id"],
                    "dataset": record["dataset"],
                    "split": record["split"],
                    "source_path": str(source),
                    "source_size": source_size,
                    "crop_view": view,
                }
            )
    return rows


def _plan_payload(
    manifests: Sequence[Path],
    selection_audit: Mapping[str, object],
    expected_rows: Sequence[Mapping[str, object]],
    arguments: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "distillation_embedding_cache_plan",
        "manifests": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifests
        ],
        "pt_model_sha256": sha256_file(arguments.pt_model),
        "campplus_model_sha256": sha256_file(arguments.campplus_model),
        "selection_seed": arguments.selection_seed,
        "included_splits": sorted(arguments.include_split),
        "maximum_utterances_per_speaker": arguments.max_utterances_per_speaker,
        "minimum_utterances_per_speaker": arguments.min_utterances_per_speaker,
        "crop_views": arguments.crop_views,
        "remove_dc": False,
        "device": arguments.device,
        "torch_threads": arguments.torch_threads,
        "expected_cache_row_count": len(expected_rows),
        "selection_audit": dict(selection_audit),
    }


def _same_json(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(
        right, ensure_ascii=False, sort_keys=True
    )


def run(arguments: argparse.Namespace) -> Dict[str, object]:
    manifests = [Path(path).resolve() for path in arguments.manifest]
    all_records = []  # type: List[Dict[str, str]]
    for manifest in manifests:
        all_records.extend(read_manifest(manifest, namespace_speakers=True))
    selected, selection_audit = select_records(
        all_records,
        arguments.include_split,
        arguments.max_utterances_per_speaker,
        arguments.min_utterances_per_speaker,
        arguments.selection_seed,
    )
    expected_rows = build_expected_rows(selected, arguments.crop_views)
    plan = _plan_payload(
        manifests, selection_audit, expected_rows, arguments
    )
    work_dir = Path(arguments.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    plan_path = work_dir / "plan.json"
    if plan_path.is_file():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if not _same_json(previous, plan):
            raise RuntimeError(
                "work-dir 已绑定不同 cache plan；请使用新目录，禁止混用 shards"
            )
    else:
        _atomic_json(plan_path, plan)

    torch.set_num_threads(arguments.torch_threads)
    extractor = DurationAdaptiveEnsembleExtractor(
        pt_model_path=arguments.pt_model,
        campplus_model_path=arguments.campplus_model,
        device_name=arguments.device,
    )
    sample_rate = int(extractor.fbank_config.sample_rate)
    crop05_count = int(round(0.5 * sample_rate))
    crop10_count = int(round(1.0 * sample_rate))
    completed = 0
    resumed = 0
    started = time.time()

    rows_by_utterance = defaultdict(list)  # type: Dict[str, List[Dict[str, object]]]
    for row in expected_rows:
        rows_by_utterance[str(row["utterance_id"])].append(row)

    for utterance_id, utterance_rows in rows_by_utterance.items():
        source_paths = {str(row["source_path"]) for row in utterance_rows}
        if len(source_paths) != 1:
            raise RuntimeError(
                "同一 utterance_id 指向多个源文件：{}".format(utterance_id)
            )
        source = Path(next(iter(source_paths)))
        waveform = load_audio(source, sample_rate, remove_dc=False)
        full_embedding = None  # type: Optional[np.ndarray]
        for row in utterance_rows:
            start05 = deterministic_crop_start(
                waveform.size,
                crop05_count,
                int(row["crop_view"]),
                arguments.crop_views,
            )
            start10 = deterministic_crop_start(
                waveform.size,
                crop10_count,
                int(row["crop_view"]),
                arguments.crop_views,
            )
            expected = dict(row)
            expected["crop05_start"] = start05
            expected["crop10_start"] = start10
            shard_path = _shard_path(work_dir, str(row["embedding_id"]))
            if shard_path.is_file():
                validate_shard(shard_path, expected)
                resumed += 1
                completed += 1
                continue

            if full_embedding is None:
                full_embedding = extract_acoustic_waveform(
                    extractor, waveform, extractor.long_pt_weight
                )
            short05 = extract_acoustic_waveform(
                extractor,
                waveform[start05 : start05 + crop05_count],
                extractor.short_0p5_pt_weight,
            )
            short10 = extract_acoustic_waveform(
                extractor,
                waveform[start10 : start10 + crop10_count],
                extractor.short_1p0_pt_weight,
            )
            arrays = {
                "schema_version": np.asarray(SHARD_SCHEMA_VERSION, dtype=np.int64),
                "embedding_id": np.asarray(row["embedding_id"]),
                "utterance_id": np.asarray(utterance_id),
                "speaker_id": np.asarray(row["speaker_id"]),
                "dataset": np.asarray(row["dataset"]),
                "split": np.asarray(row["split"]),
                "source_path": np.asarray(str(source)),
                "source_size": np.asarray(row["source_size"], dtype=np.int64),
                "crop_view": np.asarray(row["crop_view"], dtype=np.int64),
                "crop05_start": np.asarray(start05, dtype=np.int64),
                "crop10_start": np.asarray(start10, dtype=np.int64),
                "full": full_embedding,
                "short05": short05,
                "short10": short10,
            }
            _atomic_npz(shard_path, arrays)
            validate_shard(shard_path, expected)
            completed += 1
            if completed == 1 or completed % arguments.log_every == 0:
                elapsed = time.time() - started
                print(
                    "cache progress {}/{} resumed={} elapsed={:.1f}s".format(
                        completed, len(expected_rows), resumed, elapsed
                    ),
                    flush=True,
                )

    consolidated = defaultdict(list)  # type: Dict[str, List[np.ndarray]]
    for row in expected_rows:
        # Crop starts are authenticated by reading the completed shard; there
        # is no need to decode the audio again during consolidation.
        shard_path = _shard_path(work_dir, str(row["embedding_id"]))
        with np.load(str(shard_path), allow_pickle=False) as loaded:
            payload = {key: loaded[key] for key in loaded.files}
        expected = dict(row)
        expected["crop05_start"] = int(payload["crop05_start"].item())
        expected["crop10_start"] = int(payload["crop10_start"].item())
        payload = validate_shard(shard_path, expected)
        for key in ("embedding_id", "utterance_id", "speaker_id", "dataset", "split", "full", "short05", "short10"):
            consolidated[key].append(np.asarray(payload[key]))

    cache_payload = {
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
        "embedding_id": np.asarray(
            [str(value.item()) for value in consolidated["embedding_id"]]
        ),
        "utterance_id": np.asarray(
            [str(value.item()) for value in consolidated["utterance_id"]]
        ),
        "speaker_id": np.asarray(
            [str(value.item()) for value in consolidated["speaker_id"]]
        ),
        "dataset": np.asarray(
            [str(value.item()) for value in consolidated["dataset"]]
        ),
        "split": np.asarray(
            [str(value.item()) for value in consolidated["split"]]
        ),
        "full": np.ascontiguousarray(np.stack(consolidated["full"]), dtype=np.float32),
        "short05": np.ascontiguousarray(np.stack(consolidated["short05"]), dtype=np.float32),
        "short10": np.ascontiguousarray(np.stack(consolidated["short10"]), dtype=np.float32),
    }
    output = Path(arguments.output).resolve()
    write_embedding_cache(output, cache_payload)
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "artifact_kind": "distillation_embedding_cache_receipt",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "row_count": len(expected_rows),
        "utterance_count": len(selected),
        "crop_views": arguments.crop_views,
        "remove_dc": False,
        "device": arguments.device,
        "torch_threads": arguments.torch_threads,
        "resumed_shard_count": resumed,
        "computed_shard_count": completed - resumed,
        "elapsed_seconds": time.time() - started,
        "selection_audit": selection_audit,
    }
    receipt_path = Path(arguments.receipt).resolve()
    _atomic_json(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--pt-model", type=Path, default=PROJECT_ROOT / "model_pt.pt")
    parser.add_argument(
        "--campplus-model",
        type=Path,
        default=PROJECT_ROOT / "model/campplus_cn_en_common.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--include-split",
        action="append",
        choices=("train", "dev", "test"),
        default=None,
    )
    parser.add_argument(
        "--max-utterances-per-speaker",
        type=int,
        default=DEFAULT_MAX_UTTERANCES_PER_SPEAKER,
    )
    parser.add_argument(
        "--min-utterances-per-speaker",
        type=int,
        default=DEFAULT_MINIMUM_UTTERANCES_PER_SPEAKER,
    )
    parser.add_argument("--crop-views", type=int, default=DEFAULT_CROP_VIEWS)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.include_split is None:
        arguments.include_split = ["train", "dev", "test"]
    try:
        receipt = run(arguments)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(
            "distillation cache build failed: {}: {}".format(type(error).__name__, error),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
