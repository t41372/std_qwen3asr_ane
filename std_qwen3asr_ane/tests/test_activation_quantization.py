"""A8 insertion changes only projection inputs, preserving other FP16 consumers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))


def test_projection_quantization_does_not_rewrite_other_consumers(tmp_path):
    ct = pytest.importorskip("coremltools")
    from build_w8a8_head import quantize_projection_inputs
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    from std_qwen3asr_ane.runtime import PersistentInputModel

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 8, 1, 1), dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def program(x):
        raw = mb.identity(x=x, name="raw")
        y = mb.conv(x=x, weight=np.eye(8, dtype=np.float16).reshape(8, 8, 1, 1), name="y")
        return y, raw

    assert quantize_projection_inputs(program, 0.125) == 1
    converted = ct.convert(
        program, minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_ONLY
    )
    model = PersistentInputModel(converted)
    converted = None
    x = np.array([-0.26, 0.04, 0.14, 0.26, 0.49, -0.04, 0.34, -0.34], np.float16).reshape(
        1, 8, 1, 1
    )
    try:
        result = model.predict({"x": x})
        np.testing.assert_array_equal(result["raw"], x)
        np.testing.assert_array_equal(
            result["y"], (np.round(x / np.float16(0.125)) * np.float16(0.125))
        )
    finally:
        model.close()
