"""Synthetic vector/I-O checks; these do not evaluate speaker accuracy."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from batch_submission_runtime import _adapt, run_batch_centering_inference
from strict_submission_runtime import InternalFallbackError


def fixtures(count, alpha=0.32, route="data_scenery1_mobile"):
    rng = np.random.RandomState(71)
    acoustic = rng.randn(count, 3520).astype(np.float32)
    acoustic /= np.linalg.norm(acoustic, axis=1, keepdims=True)
    teacher = rng.randn(count, 256).astype(np.float32)
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True)
    values = np.concatenate((np.sqrt(1-alpha)*acoustic,
                             np.sqrt(alpha)*teacher), axis=1).astype(np.float32)
    metadata = [dict(teacher_fusion_route=route, teacher_fusion_policy="short05", teacher_fusion_alpha=alpha,
                     teacher_fusion_teacher_active=True, fallback_mode="none") for _ in values]
    return values, metadata, teacher


class RuntimeContract(unittest.TestCase):
    def test_small_group_is_unchanged(self):
        values, metadata, _ = fixtures(7)
        actual, audit = _adapt(values, metadata)
        np.testing.assert_array_equal(actual, values)
        self.assertFalse(audit[0]["applied"])

    def test_only_teacher_block_changes_at_eight_files(self):
        values, metadata, teacher = fixtures(8)
        actual, audit = _adapt(values, metadata)
        actual = np.stack(actual)
        centered = teacher - 1.05 * teacher.mean(axis=0, keepdims=True)
        centered /= np.linalg.norm(centered, axis=1, keepdims=True)
        np.testing.assert_allclose(actual[:, -256:], np.sqrt(.32)*centered, atol=1e-6)
        np.testing.assert_array_equal(actual[:, :3520], values[:, :3520])
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
        self.assertTrue(audit[0]["applied"])

    def test_sources_are_not_pooled(self):
        values, metadata, _ = fixtures(8)
        for item in metadata[4:]:
            item["teacher_fusion_route"] = "data_scenery2_joint"
        actual, audit = _adapt(values, metadata)
        np.testing.assert_array_equal(actual, values)
        self.assertEqual([a["files"] for a in audit], [4, 4])

    def test_inconsistent_alpha_is_rejected(self):
        values, metadata, _ = fixtures(8)
        metadata[-1]["teacher_fusion_alpha"] = .4
        with self.assertRaisesRegex(RuntimeError, "inconsistent alpha"):
            _adapt(values, metadata)

    def test_zip_mapping_and_payload(self):
        values, metadata, _ = fixtures(8)
        class Extractor:
            output_dim = 3776
            def extract_file(self, path):
                index = int(path.stem)
                return values[index], metadata[index]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "input/data_scenery1/audio"
            audio.mkdir(parents=True)
            for i in range(8):
                (audio / (str(i) + ".wav")).write_bytes(b"synthetic-path-fixture")
            report = run_batch_centering_inference(Extractor(), root / "input", root / "out",
                                       logger=lambda _: None)
            self.assertEqual(report["file_count"], 8)
            with zipfile.ZipFile(root / "out/submit.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()),
                                 {"data_scenery1/audio/%d.npz" % i for i in range(8)})
                for name in archive.namelist():
                    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as item:
                        self.assertEqual(item.files, ["embedding"])
                        self.assertEqual(item["embedding"].shape, (3776,))
                        self.assertEqual(item["embedding"].dtype, np.float32)

    def test_internal_fallback_never_publishes_zip(self):
        values, metadata, _ = fixtures(1)
        metadata[0]["fallback_mode"] = "substitute"
        class Extractor:
            output_dim = 3776
            def extract_file(self, path):
                return values[0], metadata[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "input/data_scenery1/audio"
            audio.mkdir(parents=True)
            (audio / "0.wav").write_bytes(b"synthetic-path-fixture")
            with self.assertRaises(InternalFallbackError):
                run_batch_centering_inference(Extractor(), root / "input", root / "out",
                                  logger=lambda _: None)
            self.assertFalse((root / "out/submit.zip").exists())


if __name__ == "__main__":
    unittest.main()
