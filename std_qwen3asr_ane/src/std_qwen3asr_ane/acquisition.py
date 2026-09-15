"""Run conversion in a managed environment when the host cannot do it.

Inference and discovery never enter this module's subprocess path. The worker
uses the installed plugin's source and its own pinned conversion dependencies,
so a GPU draft's Transformers 5 cannot replace the converter's Transformers 4.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field
from standard_asr.contract.exceptions import ArtifactAcquisitionError
from standard_asr.engine import ArtifactAction, ArtifactProgress, allow_downloads

if TYPE_CHECKING:
    from collections.abc import Callable

    from .plugin import Qwen3ASRConfig


class AcquisitionFailure(BaseModel):
    """Structured error frame shared by the parent and conversion worker."""

    model_config = ConfigDict(extra="forbid")
    message: str
    reason: Literal["downloads_disabled", "action_required", "unsupported", "busy", "failed"]
    hint: str | None = None
    required_actions: tuple[ArtifactAction, ...] = ()
    retriable_after: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @classmethod
    def from_exception(cls, error: ArtifactAcquisitionError) -> AcquisitionFailure:
        return cls(
            message=str(error),
            reason=error.reason,
            hint=error.hint,
            required_actions=error.required_actions,
            retriable_after=error.retriable_after,
        )

    def to_exception(self) -> ArtifactAcquisitionError:
        return ArtifactAcquisitionError(
            self.message,
            reason=self.reason,
            hint=self.hint,
            required_actions=self.required_actions,
            retriable_after=self.retriable_after,
        )


@contextmanager
def acquisition_lock(config: Qwen3ASRConfig, *, include_draft: bool = True):
    """Protect shared checkpoints and targets across engines and processes.

    Lock files stay in place: unlinking a locked file lets another caller lock
    a new inode while an older caller still owns the original one.
    """
    paths = {config.model_dir, config.source_dir}
    if include_draft and config.draft_dir is not None:
        paths.update((config.draft_dir, config.draft_source_dir))
    with ExitStack() as stack:
        for path in sorted({path.expanduser().resolve() for path in paths}):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.with_name(f".{path.name}.lock").open("a"))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ArtifactAcquisitionError(
                    "Another process is acquiring this model or its source checkpoint.",
                    reason="busy",
                    retriable_after=1.0,
                ) from exc
        yield


def acquire_in_worker(
    config: Qwen3ASRConfig,
    targets: set[str],
    progress: Callable[[ArtifactProgress], None] | None,
) -> None:
    """Forward only structured progress; native conversion logs stay on stderr."""
    from uv import find_uv_bin

    worker = Path(__file__).with_name("conversion_worker.py")
    command = [
        find_uv_bin(),
        "run",
        "--no-project",
        "--no-config",
        "--python",
        sys.executable,
    ]
    if not allow_downloads():
        command.append("--offline")
    command.extend(["--script", str(worker)])
    request = {"config": config.model_dump(mode="json"), "targets": sorted(targets)}
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.close()
            for line in process.stdout:
                frame = json.loads(line)
                if isinstance(frame, dict) and set(frame) == {"error"}:
                    raise AcquisitionFailure.model_validate(frame["error"]).to_exception()
                event = ArtifactProgress.model_validate(frame)
                if progress is not None:
                    progress(event)
            if process.wait() != 0:
                raise RuntimeError("The conversion worker failed; see its stderr output.")
        finally:
            # Do not leave an orphan converter if the caller interrupts pull.
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    # uv may exit before a child which ignored SIGTERM. Keep
                    # our artifact locks until the entire worker group is stopped.
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
