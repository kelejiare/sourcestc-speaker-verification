#!/usr/bin/env python3
"""Create a classifier-free, fp16-storage copy of the public speaker teacher."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "560b2652224be4fdd4be56172c6cf5114513e8c9858361f44c5929a8100449a6"
EXPECTED_SOURCE_BYTES = 2508339095
EXPECTED_SPK_TENSORS = 933
EXPECTED_SPK_PARAMETERS = 586652993
EXPECTED_CLASSIFIER_SHAPE = (157548, 256)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    receipt_path = args.receipt.resolve()

    observed_bytes = source_path.stat().st_size
    observed_sha256 = sha256_file(source_path)
    if observed_bytes != EXPECTED_SOURCE_BYTES or observed_sha256 != SOURCE_SHA256:
        raise RuntimeError("source checkpoint identity mismatch")

    payload = torch.load(str(source_path), map_location="cpu", weights_only=True)
    if set(payload) != {"modules", "epoch_idx"} or set(payload["modules"]) != {
        "spk_model", "classifier"
    }:
        raise RuntimeError("unexpected source checkpoint schema")
    classifier = payload["modules"]["classifier"]
    if set(classifier) != {"weight"} or tuple(classifier["weight"].shape) != EXPECTED_CLASSIFIER_SHAPE:
        raise RuntimeError("unexpected classifier schema")

    source_state = payload["modules"]["spk_model"]
    parameter_count = sum(int(value.numel()) for value in source_state.values())
    if len(source_state) != EXPECTED_SPK_TENSORS or parameter_count != EXPECTED_SPK_PARAMETERS:
        raise RuntimeError("speaker state identity mismatch")

    state_dict = {}
    for key, value in source_state.items():
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float16)
        state_dict[str(key)] = tensor

    reduced = {
        "schema_version": 1,
        "artifact_kind": "w2vbert2_multidomain_speaker_teacher_fp16_storage",
        "source_sha256": observed_sha256,
        "source_epoch_idx": int(payload["epoch_idx"]),
        "embedding_dimension": 256,
        "state_dict": state_dict,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reduced, str(output_path))
    output_sha256 = sha256_file(output_path)
    receipt = {
        "schema_version": 1,
        "source": str(source_path.relative_to(ROOT)),
        "source_bytes": observed_bytes,
        "source_sha256": observed_sha256,
        "source_epoch_idx": int(payload["epoch_idx"]),
        "excluded_classifier_shape": list(EXPECTED_CLASSIFIER_SHAPE),
        "speaker_tensor_count": len(state_dict),
        "speaker_parameter_count": parameter_count,
        "output": str(output_path.relative_to(ROOT)),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": output_sha256,
        "floating_storage_dtype": "float16",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
