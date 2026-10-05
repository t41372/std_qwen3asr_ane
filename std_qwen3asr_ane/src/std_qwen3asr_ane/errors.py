"""Exceptions and cancellation helpers shared by the native runtime and adapter."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event

type CancellationToken = Event | Callable[[], bool]


class ModelLimitError(ValueError):
    """A request exceeds a fixed capacity of the loaded bundle or its generation budget.

    The audio duration, the decoder KV cache and ``max_new_tokens`` are all
    fixed once a bundle is loaded. These failures are request-shaped, not
    inference faults, so the adapter reports them with their exact limit.
    """


class InferenceCancelled(RuntimeError):
    """A caller cancelled between complete native predictions.

    A Core ML or MLX prediction already running when cancellation arrives is
    allowed to finish. Callers must discard its output, invalidate any affected
    per-request state, and release their serialized execution lane before
    starting another request.
    """


def raise_if_cancelled(cancel: CancellationToken | None) -> None:
    """Raise :class:`InferenceCancelled` when a cooperative token is set.

    ``Event`` is the cross-thread path used by streaming. A callback permits
    other runtimes to provide the same cooperative contract without importing
    threading. This helper intentionally makes no attempt to interrupt native
    execution; it is called only at safe boundaries around complete predictions.
    """
    if cancel is None:
        return
    requested = cancel.is_set() if isinstance(cancel, Event) else cancel()
    if requested:
        raise InferenceCancelled("Inference cancelled between native predictions")
