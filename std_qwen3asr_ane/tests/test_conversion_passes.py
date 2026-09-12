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
