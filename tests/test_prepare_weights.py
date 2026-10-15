"""Weight aliases must preserve bytes and reject an untrusted archive."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_weights.py"


class PrepareWeights(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "weight alias preparation is not implemented")
        spec = importlib.util.spec_from_file_location("prepare_weights", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def fixture(self, root, name="submitted/model/internal_adapter.npz"):
        archive = root / "models.tar.gz"
        data = b"frozen model bytes"
        with tarfile.open(archive, "w:gz") as output:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
        manifest = [{"path":"model/joint_kd_adapter.npz", "bytes":len(data),
                     "sha256":hashlib.sha256(data).hexdigest()}]
        return archive, data, manifest

    def test_alias_retains_exact_weight_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, data, manifest = self.fixture(root)
            self.module.prepare(archive, root/"prepared", manifest,
                                archive.stat().st_size, self.module.sha256(archive))
            self.assertEqual((root/"prepared/model/joint_kd_adapter.npz").read_bytes(),data)

    def test_wrong_archive_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, _, manifest = self.fixture(root)
            with self.assertRaises(ValueError):
                self.module.prepare(archive, root/"prepared", manifest,
                                    archive.stat().st_size, "0"*64)
            self.assertFalse((root/"prepared").exists())

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            archive, _, manifest=self.fixture(root,"../model/internal_adapter.npz")
            with self.assertRaises(ValueError):
                self.module.prepare(archive,root/"prepared",manifest,
                                    archive.stat().st_size,self.module.sha256(archive))
            self.assertFalse((root/"prepared").exists())

    def test_modified_prepared_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            archive, _, manifest=self.fixture(root)
            args=(archive,root/"prepared",manifest,archive.stat().st_size,self.module.sha256(archive))
            self.module.prepare(*args)
            (root/"prepared/model/joint_kd_adapter.npz").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                self.module.prepare(*args)


if __name__ == "__main__":
    unittest.main()
