"""Host-only benchmark tests; no MLX import, model loading or GPU work."""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_mlx as benchmark


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.model = Mock()
        self.model.generate.side_effect = lambda *args, **kwargs: (
            self.events.append("generate")
            or SimpleNamespace(
                text="hello",
                generation_tokens=2,
                prompt_tokens=10,
                total_time=0.5,
                language=["English"],
            )
        )
        self.mx = ModuleType("mlx.core")
        self.mx.gpu = "gpu"
        self.mx.set_default_device = Mock()
        self.mx.random = SimpleNamespace(seed=Mock())
        self.mx.synchronize = lambda stream=None: self.events.append(
            "generation_synchronize" if stream else "synchronize"
        )
        self.stt = ModuleType("mlx_audio.stt")
        self.stt.load = Mock(return_value=self.model)
        self.generate_module = ModuleType("mlx_audio.lm.generate")
        self.generate_module.generation_stream = "generation_stream"
        self.modules = {
            "mlx": ModuleType("mlx"),
            "mlx.core": self.mx,
            "mlx_audio": ModuleType("mlx_audio"),
            "mlx_audio.stt": self.stt,
            "mlx_audio.lm": ModuleType("mlx_audio.lm"),
            "mlx_audio.lm.generate": self.generate_module,
        }

    def test_explicit_gpu_strict_loading_and_synchronized_sample(self):
        with patch.dict(sys.modules, self.modules):
            transcribe, load_seconds = benchmark.load_backend(Path("local-model"), 42)
            result = transcribe(np.zeros(16000), "en", 256)
        self.assertGreaterEqual(load_seconds, 0)
        self.assertEqual(
            self.events[-5:],
            [
                "generation_synchronize",
                "synchronize",
                "generate",
                "generation_synchronize",
                "synchronize",
            ],
        )
        self.mx.set_default_device.assert_called_once_with("gpu")
        self.stt.load.assert_called_once_with(
            Path("local-model"), strict=True, lazy=False
        )
        self.assertEqual(result["hypothesis"], "hello")
        self.assertEqual(result["detected_language"], "en")
        self.assertEqual(self.model.generate.call_args.kwargs["language"], "English")
        self.assertEqual(self.model.generate.call_args.kwargs["temperature"], 0.0)

    def test_token_budget_and_error_paths_still_synchronize(self):
        with patch.dict(sys.modules, self.modules):
            transcribe, _ = benchmark.load_backend(Path("local-model"), 42)
            with self.assertRaisesRegex(RuntimeError, "token budget"):
                transcribe(np.zeros(16000), None, 2)
            self.model.generate.side_effect = RuntimeError("native failure")
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                transcribe(np.zeros(16000), None, 256)
        self.assertEqual(self.events[-1], "synchronize")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sf.write(self.root / "audio.wav", np.zeros(16000), 16000)
        (self.root / "manifest.jsonl").write_text(
            json.dumps(
                {
                    "id": "one",
                    "audio_path": "audio.wav",
                    "reference": "hello",
                    "language": "en",
                }
            )
            + "\n"
        )
        self.args = argparse.Namespace(
            model_dir=self.root,
            model_repo="Qwen/Qwen3-ASR-1.7B",
            model_revision=benchmark.OFFICIAL_REVISION,
            output=self.root / "run.jsonl",
            inspect_only=False,
            manifest=self.root / "manifest.jsonl",
            warmups=1,
            repeats=2,
            max_new_tokens=256,
            language_mode="manifest",
            seed=42,
        )

    def test_jsonl_matches_shared_evaluator_and_tracks_warmups(self):
        transcribe = Mock(
            return_value={
                "hypothesis": "hello",
                "detected_language": "en",
                "seconds": 0.25,
            }
        )
        with (
            patch.object(benchmark, "provenance", return_value={"environment": {}}),
            patch.object(benchmark, "load_backend", return_value=(transcribe, 1.0)),
        ):
            self.assertEqual(benchmark.run(self.args), 0)
        rows = [json.loads(line) for line in self.args.output.read_text().splitlines()]
        self.assertEqual(
            [row["phase"] for row in rows], ["warmup", "measured", "measured"]
        )
        self.assertTrue(all(row["compute_units"] == "mlx_gpu" for row in rows))
        self.assertTrue(all(row["weights_dtype"] == "bf16" for row in rows))
        summary = json.loads(Path(str(self.args.output) + ".summary.json").read_text())
        self.assertEqual(summary["quality"]["all"]["wer"]["rate"], 0.0)
        self.assertEqual(summary["quality"]["all"]["samples"], 1)
        self.assertEqual(summary["latency"]["successful_attempts"], 2)
        self.assertFalse(summary["energy"]["measured"])

    def test_inspection_never_loads_backend(self):
        self.args.inspect_only = True
        with (
            patch.object(benchmark, "provenance", return_value={}),
            patch.object(benchmark, "load_backend") as loader,
        ):
            self.assertEqual(benchmark.run(self.args), 0)
            loader.assert_not_called()
        self.assertFalse(json.loads(self.args.output.read_text())["model_loaded"])


if __name__ == "__main__":
    unittest.main()
