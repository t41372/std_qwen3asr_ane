"""A compiler optimization must not silently undo the measured ANE workaround."""

import pytest
import torch
from torch import nn

from std_qwen3asr_ane.conversion.encoder import exact_gelu
from std_qwen3asr_ane.conversion.passes import ane_pass_pipeline, verify_activation_operators


class Activation(nn.Module):
    def forward(self, x):
        return exact_gelu(x)


def test_pipeline_preserves_exact_gelu_in_serialized_graph():
    ct = pytest.importorskip("coremltools")
    sample = torch.randn(1, 8, 1, 4)
    traced = torch.jit.trace(Activation().eval(), sample)

    def convert(pipeline):
        return ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=sample.shape)],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            skip_model_load=True,
            pass_pipeline=pipeline,
        )

    native = convert(ct.PassPipeline.DEFAULT)
    with pytest.raises(RuntimeError, match="native gelu"):
        verify_activation_operators(native)
    precise = convert(ane_pass_pipeline())
    verify_activation_operators(precise)
    operations = [
        operation.type
        for function in precise.get_spec().mlProgram.functions.values()
        for block in function.block_specializations.values()
        for operation in block.operations
    ]
    assert "erf" in operations


def test_weight_compression_preserves_exact_gelu(tmp_path):
    ct = pytest.importorskip("coremltools")
    from std_qwen3asr_ane.conversion.compress import compress_model

    module = nn.Sequential(nn.Conv2d(128, 64, 1), Activation()).eval()
    sample = torch.randn(1, 128, 1, 4)
    source, output = tmp_path / "dense.mlpackage", tmp_path / "lut8.mlpackage"
    model = ct.convert(
        torch.jit.trace(module, sample),
        inputs=[ct.TensorType(name="input", shape=sample.shape)],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        pass_pipeline=ane_pass_pipeline(),
    )
    model.save(str(source))
    counts = compress_model(source, output, "palette", 8, 32, preserve_activation_expressions=True)
    compressed = ct.models.MLModel(str(output), skip_model_load=True)
    verify_activation_operators(compressed)
    assert counts["compressed"] == 1
    for field in ("input", "output", "state"):
        assert getattr(compressed.get_spec().description, field) == getattr(
            model.get_spec().description, field
        )
    operations = [
        operation.type
        for function in compressed.get_spec().mlProgram.functions.values()
        for block in function.block_specializations.values()
        for operation in block.operations
    ]
    assert "erf" in operations and "constexpr_lut_to_dense" in operations
