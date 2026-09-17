"""Output ownership and greedy selection at the synchronous Core ML boundary."""

import gc
import weakref

import numpy as np
import pytest

from std_qwen3asr_ane.runtime import PersistentInputModel, logits_token


def test_consumption_keeps_native_array_alive_only_during_callback():
    references = []

    class Native:
        def predict(self, data):
            output = np.arange(6, dtype=np.float16)
            references.append(weakref.ref(output))
            return {"logits_0": output}

    model = PersistentInputModel(Native())

    def consume(outputs):
        assert references[-1]() is outputs["logits_0"]
        return logits_token(outputs, vocabulary_size=6)

    for _ in range(100):
        assert model.predict_consumed({"hidden": np.ones(4, np.float16)}, consume) == 5
    gc.collect()
    assert all(reference() is None for reference in references)
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.predict_consumed({"hidden": np.ones(4)}, consume)


def test_consumer_exception_preserves_borrowed_input_owner():
    retained = []

    class Native:
        def predict(self, data, *, state):
            assert state == "state"
            retained.append(data["hidden"])
            return {"logits_0": np.ones(1)}

    model = PersistentInputModel(Native())

    def fail(outputs):
        raise ValueError("consumer failed")

    with pytest.raises(ValueError, match="consumer failed"):
        model.predict_consumed({"hidden": np.ones(4)}, fail, state="state")
    reference = weakref.ref(retained[0])
    with pytest.raises(RuntimeError, match="still borrows"):
        model.close(timeout=0)
    assert reference() is not None
    retained.clear()
    model.close()
    assert reference() is None


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_chunk_ties_use_first_vocabulary_index_with_strided_outputs(dtype):
    first = np.array([1, 99, 7, 99, 7, 99], dtype=dtype)[::2]
    assert not first.flags.c_contiguous
    # Dictionary insertion order must not control the vocabulary offset.
    outputs = {"logits_1": np.array([7, 2], dtype=dtype), "logits_0": first}
    assert logits_token(outputs, vocabulary_size=5) == 1


@pytest.mark.parametrize(
    "outputs,size,error",
    [
        ({"logits_0": np.array([])}, 0, "empty"),
        ({"logits_0": np.array([np.nan])}, 1, "non-finite"),
        ({"logits_0": np.array([np.inf])}, 1, "non-finite"),
        ({"logits_0": np.array([-np.inf])}, 1, "non-finite"),
        ({"logits_0": np.array([1]), "logits_2": np.array([2])}, 2, "contiguous"),
        ({"logits_0": np.array([1])}, 2, "vocabulary"),
        ({}, 1, "vocabulary"),
    ],
)
def test_invalid_head_output_fails_instead_of_selecting_token(outputs, size, error):
    with pytest.raises(RuntimeError, match=error):
        logits_token(outputs, vocabulary_size=size)
