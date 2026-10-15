#!/usr/bin/env python3
"""Verify the submitted archive and expose weights under descriptive names."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

PACKAGE_BYTES = 1511236734
PACKAGE_SHA256 = "5bc58b2136e70d2b1756309e1e1a08a2a4054aeb84eb741c7a568c5b3e00ece2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(archive, target, manifest, expected_bytes, expected_sha256):
    archive, target = Path(archive), Path(target)
    if archive.stat().st_size != expected_bytes or sha256(archive) != expected_sha256:
        raise ValueError("model archive size or SHA-256 mismatch")
    expected = {}
    for item in manifest:
        path = PurePosixPath(item["path"])
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "model":
            raise ValueError("invalid model manifest path")
        if item["sha256"] in expected:
            raise ValueError("duplicate model manifest digest")
        expected[item["sha256"]] = item
    if target.exists():
        for item in manifest:
            path = target / item["path"]
            if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError("prepared model file is missing or modified: " + item["path"])
        return target / "model"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-prepare-", dir=str(target.parent)) as temp:
        staging = Path(temp) / "prepared"
        staging.mkdir()
        found = set()
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                    raise ValueError("unsafe model archive member")
                if not member.isfile() or len(path.parts) < 3 or path.parts[1] != "model":
                    continue
                temporary = Path(temp) / "member"
                with source.extractfile(member) as incoming, temporary.open("wb") as output:
                    shutil.copyfileobj(incoming, output, length=4 * 1024 * 1024)
                digest = sha256(temporary)
                if digest not in expected or digest in found:
                    raise ValueError("unrecognized or duplicate model weight")
                item = expected[digest]
                if temporary.stat().st_size != item["bytes"]:
                    raise ValueError("model weight size mismatch")
                destination = staging / item["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
                found.add(digest)
        if found != set(expected):
            raise ValueError("model archive is missing required weights")
        staging.replace(target)
    return target / "model"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=root / "weights/prepared")
    args = parser.parse_args()
    manifest = json.loads((root / "scripts/model_assets.json").read_text(encoding="utf-8"))
    print(prepare(args.archive, args.output, manifest, PACKAGE_BYTES, PACKAGE_SHA256))


if __name__ == "__main__":
    main()
