# Core ML input ownership investigation

## Failure and hypothesis

The first real incremental streaming verification terminated with exit 139. The
macOS crash report `python3.12-2026-09-12-112646.ips` shows `_PyObject_Free`, followed
by `libcoremlpython.so`, Objective-C C++ instance destruction, `MLFeatureValue`
deallocation, and `MLE5InputPortBinder` / `resetAfterLingering` cleanup on a native
worker thread. This was a native process crash, not session cancellation.

The observed stack is consistent with a borrowed NumPy object's final Python
reference being released by delayed Core ML cleanup. It does not establish that
all models or all idle intervals reproduce the failure.

## Investigation in order

1. Used an existing one-layer decoder to avoid repeatedly loading the complete
   1.7B bundle. Models were loaded on the main thread and predictions submitted to
   a single worker, matching the original application's loading pattern.
2. Ran 256 transient-input predictions followed by idle periods and model release.
   That control did not crash.
3. Tested individual predictions separated by two-second idle gaps, both on the
   main thread and with retained inputs on a worker. Those controls also completed.
4. Tested the existing stateless audio frontend with two-second gaps, then with
   ten-second gaps and ten-second post-prediction/post-release observation periods.
   The small controls still did not reproduce the complete-runtime failure.
5. Inspected coremltools 9's Python bridge. `MLModel.predict()` unconditionally
   promotes each float16 input array to a newly allocated float32 array, replacing
   the dictionary value. Merely keeping the caller's original float16 arrays alive
   would therefore not protect the actual arrays passed into Core ML.
6. Inspected the public native implementation. `PybindCompatibleArray` retains a
   `py::array` member; its source does not install a GIL-aware custom destructor.
   This supports the ownership hypothesis, without proving every path in the
   shipped binary or operating system's cleanup behavior.
7. Added `PersistentInputModel` around every runtime model. It owns exactly one
   fixed-shape float32 buffer per input, copies new values into those buffers and
   copies every result into host-owned NumPy storage. Decoder math still rounds its
   values to float16 before the bridge copy, matching the original effective input
   conversion. Input identities do not grow with the number of predictions.
8. Added explicit close ordering: release the MLModel first, retain input owners
   until external Python/native references drain, then release the buffers. Close
   has a timeout that raises while retaining resources. A nonblocking finalizer
   retires the single fixed buffer set on a Python cleanup thread if another live
   frame or native borrower still holds it; it does not save every call's inputs.
   The initial synchronous-finalizer experiment was rejected because a pytest
   failure traceback could keep buffers alive and stall object collection.
9. Ran the actual persistent wrapper with the small decoder on a worker, two-second
   inter-call idle periods, five-second lingering observation and explicit close.
   It completed successfully.
10. Loaded the full T16 bundle once and ran the real 4.2039375-second Chinese audio
    twice as 250 ms realtime PCM frames. Each session produced valid partials,
    `closed`, and `done`; both final texts were
    `甚至出现交易几乎停滞的情况。`. Both recorded sequences passed Standard ASR
    compliance. Five-second idle periods after each session, explicit close, and
    a final five-second idle period completed with exit 0.

## Evidence

The incremental record is
`artifacts/evaluation/smoke/streaming-lifetime-persistent.jsonl`; stdout/stderr are
in the matching `.log`. It records every event before any later native crash could
erase the record. The model loaded in about 42.86 seconds; explicit close completed
in approximately 41 ms. The full process reached its final `completed` record at
75.19 seconds and exited normally.

Small controls are recorded under `artifacts/probes/`:

- `thread-lifetime-transient.jsonl`
- `thread-lifetime-main-idle-transient.jsonl`
- `thread-lifetime-worker-idle-retain.jsonl`
- `frontend-lifetime-main-idle-transient.jsonl`
- `frontend-lifetime-long-idle-transient.jsonl`
- `thread-lifetime-worker-persistent-wrapper.jsonl`

`experiments/probe_coreml_thread_lifetime.py` reproduces the bounded controls;
`experiments/probe_streaming_lifetime.py` records the complete runtime with realtime
input, idle gaps and explicit model close. Unit tests verify stable buffer identity
over repeated predictions, copied output ownership, shape rejection, model-before-
buffer release ordering, and a close timeout while a borrower retains an input.

## Interpretation and limits

The complete streaming path that previously crashed now completed two real sessions,
idle cleanup and close with persistent inputs. This is a tested ownership mitigation,
not a proof that the native Core ML bridge is universally crash-free. The small
transient controls did not reproduce the original failure, so they do not establish
a clean small-model A/B reproduction. This change does not provide process-level
fault isolation against unrelated future native failures.

Callers must continue to serialize predictions and close. The Standard ASR engine
already serializes predictions with its lock; direct runtime callers must do the
same. Explicit `runtime.close()` is the observable shutdown path: timeout means
close failed, not that buffers were released safely. A future native failure should
be retained as an engine error and can motivate a separate worker process with
replayable request state; it must not be relabeled as successful cancellation.

The plugin now exposes `engine.close(timeout=5.0)`, which takes the same lock as
inference, waits for any active worker, and clears its runtime reference only after
close succeeds. A failed close keeps the owner for an explicit retry. A later
`prepare()` can load a fresh runtime. Prefer `with create_engine(...) as engine:`
or a `try/finally: engine.close()` scope around application use. Async applications
should finish/cancel their sessions, then call blocking close from a worker thread.

Retirement polling backs off from 10 ms to five seconds when native or external
references remain. During interpreter finalization it does not attempt to launch a
new Python cleanup thread; late resources remain retained instead of generating an
unraisable thread-start exception. This is a last-resort ownership precaution,
**not evidence of safe native interpreter shutdown**. The real experiment above
demonstrated explicit close before exit; implicit interpreter teardown remains a
different, unproven path.

Primary implementation references:

- [coremltools 9 Python model bridge](https://github.com/apple/coremltools/blob/9.0/coremltools/models/model.py)
- [coremltools 9 NumPy/Core ML wrapper](https://github.com/apple/coremltools/blob/9.0/coremlpython/CoreMLPythonArray.mm)
