"""Exceptions shared by the runtime and the protocol adapter without heavy imports."""


class ModelLimitError(ValueError):
    """A request exceeds a fixed capacity of the loaded bundle or its generation budget.

    The audio duration, the decoder KV cache and ``max_new_tokens`` are all
    fixed once a bundle is loaded. These failures are request-shaped, not
    inference faults, so the adapter reports them with their exact limit.
    """
