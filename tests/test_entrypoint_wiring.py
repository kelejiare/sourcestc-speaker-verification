"""Exercise actual entrypoint wiring without loading external neural weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class EntrypointWiring(unittest.TestCase):
    def test_joint_device_survives_source_adaptation(self):
        source = Path(__file__).resolve().parents[1] / "inference/run.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        graph = []
        def component(kind, device=False):
            def construct(*args, **kwargs):
                node = SimpleNamespace(kind=kind, parents=args, output_dim=3776,
                                       audit_summary=lambda: {})
                if device:
                    node.device = "cpu"
                graph.append(node)
                return node
            return construct
        ns = {
            "find_input_root":lambda:Path("input"), "find_output_dir":lambda:Path("outputs"),
            "required":lambda env,names:Path(names[0]), "required_dir":lambda env,names:Path(names[0]),
            "torch":SimpleNamespace(manual_seed=lambda _:None, set_num_threads=lambda _:None,
                                    set_num_interop_threads=lambda _:None),
            "os":SimpleNamespace(environ={}),
            "assert_online_acoustic_contract":lambda _:None,
            "DurationAdaptiveEnsembleExtractor":component("acoustic",True),
            "JointKDReweightExtractor":component("joint",True),
            "SourceAdaptedExtractor":component("source"),
            "DurationMetricExtractor":component("duration"),
            "PlusFeatureFusionExtractor":component("plus"),
            "CalibratedTeacherFusionExtractor":component("calibrated"),
            "run_batch_centering_inference":lambda *a,**k:{"groups":[],"output_zip":"submit.zip"},
            "write_report":lambda _:None, "print":lambda *a,**k:None,
            "json":SimpleNamespace(dumps=lambda *a,**k:"{}"),
        }
        exec(compile(ast.Module(body=[main],type_ignores=[]),str(source),"exec"),ns)
        ns["main"]()
        self.assertEqual([node.kind for node in graph],
                         ["acoustic","joint","source","duration","plus","calibrated"])
        self.assertIs(graph[2].parents[0],graph[1])
        self.assertIs(graph[3].parents[0],graph[2])
        self.assertIs(graph[4].parents[0],graph[3])
        self.assertIs(graph[5].parents[0],graph[4])


if __name__ == "__main__":
    unittest.main()
