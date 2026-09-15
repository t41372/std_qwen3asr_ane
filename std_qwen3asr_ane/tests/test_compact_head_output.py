"""Compact serial selection must preserve indices and full-vocabulary tie order."""

import numpy as np
import pytest

from std_qwen3asr_ane.bundle import language_head_output
from std_qwen3asr_ane.runtime import compact_token


def outputs(values, indices):
    return {
        "max_values": np.array(values, np.float16)[None, :, None],
        "max_indices": np.array(indices, np.int32)[None, :, None],
    }


def test_ties_keep_the_first_chunk_and_local_indices_remain_exact():
    assert compact_token(outputs([0, 2, 2], [1, 0, 0]), vocabulary_size=10, chunk_size=4) == 4
    assert compact_token(outputs([0, 0, 0], [0, 0, 0]), vocabulary_size=10, chunk_size=4) == 0
    assert compact_token(outputs([2, 1], [4097, 0]), vocabulary_size=9000, chunk_size=8192) == 4097


@pytest.mark.parametrize("indices", [[-1, 0, 0], [0, 4, 0], [0, 0, 2]])
def test_chunk_indices_are_checked_including_the_short_final_chunk(indices):
    with pytest.raises(RuntimeError, match="chunk"):
        compact_token(outputs([0, 1, 2], indices), vocabulary_size=10, chunk_size=4)


def test_nonfinite_values_and_float_indices_are_rejected():
    invalid = outputs([0, np.nan, 2], [0, 0, 0])
    with pytest.raises(RuntimeError, match="invalid"):
        compact_token(invalid, vocabulary_size=10, chunk_size=4)
    invalid = outputs([0, 1, 2], [0, 0, 0])
    invalid["max_indices"] = invalid["max_indices"].astype(np.float16)
    with pytest.raises(RuntimeError, match="dtype"):
        compact_token(invalid, vocabulary_size=10, chunk_size=4)


def test_compact_head_contract_requires_an_explicit_schema_version():
    descriptor = {"kind": "chunk_max", "token_batch_size": 1, "vocabulary_chunk": 8192}
    assert language_head_output({"schema_version": 1})["kind"] == "logits"
    assert language_head_output({"schema_version": 2, "head_output": descriptor}) == descriptor
    with pytest.raises(ValueError, match="schema 2"):
        language_head_output({"schema_version": 1, "head_output": descriptor})
    with pytest.raises(ValueError, match="head_output"):
        language_head_output({"schema_version": 2})
    with pytest.raises(ValueError, match="width 1"):
        language_head_output(
            {"schema_version": 2, "head_output": {**descriptor, "token_batch_size": 16}}
        )
