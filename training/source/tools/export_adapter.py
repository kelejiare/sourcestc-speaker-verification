#!/usr/bin/env python3
"""Export the frozen speaker_features three-seed ensemble under the short_pt runtime name."""

import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "records/2026-09-11/100_speaker_features_duration_conditional_pt_adapter/frozen_three_seed/adapter_ensemble_3seed.npz"
OUTPUT = ROOT / "model/short_duration_pt_adapter.npz"
EXPECTED = "b6376ee48daf7b41204e020bf2a2113dc952d02bf3dafa40d82fafa74d5496ae"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if sha256(SOURCE) != EXPECTED:
        raise RuntimeError("frozen source adapter digest mismatch")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(SOURCE), str(OUTPUT))
    if sha256(OUTPUT) != EXPECTED:
        raise RuntimeError("exported adapter digest mismatch")
    print("{}  {}".format(EXPECTED, OUTPUT))


if __name__ == "__main__":
    main()

