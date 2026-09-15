"""Public KV transfer preserves the prompt and clears unconsumed poisoned state."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from std_qwen3asr_ane.runtime import PreparedPrompt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from benchmark_speculative import transfer_prompt


def fixture():
    value = np.full((1, 2, 1, 8), np.nan, np.float16)
    value[..., :4] = np.arange(4)
    written = {}
    target = SimpleNamespace(write_state=lambda name, data: written.update({name: data.copy()}))
    destination = SimpleNamespace(
        decoders=[SimpleNamespace(make_state=lambda: target)],
        _transfer_layout=[{"key_0": (value.shape, value.dtype)}],
    )
    prepared = PreparedPrompt(
        hidden=np.zeros((1, 2, 1, 1)),
        states=(SimpleNamespace(read_state=lambda name: value),),
        token_ids=(1, 2, 3, 4),
        audio_tokens=1,
        timings={},
    )
    return value, written, destination, prepared


def test_copy_keeps_valid_prefix_and_clears_poisoned_future():
    value, written, destination, prepared = fixture()
    result = transfer_prompt(prepared, destination)
    assert result.states != prepared.states
    np.testing.assert_array_equal(written["key_0"][..., :4], value[..., :4])
    assert np.all(written["key_0"][..., 4:] == 0)
    assert np.isnan(value[..., 4:]).all(), "The source state must remain untouched"


def test_nonfinite_prompt_is_rejected():
    value, _, destination, prepared = fixture()
    value[..., 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        transfer_prompt(prepared, destination)


def test_transfer_requires_the_validated_state_layout():
    _, _, destination, prepared = fixture()
    destination._transfer_layout[0]["key_0"] = ((1, 2, 1, 9), np.dtype("float16"))
    with pytest.raises(ValueError, match="contract"):
        transfer_prompt(prepared, destination)
