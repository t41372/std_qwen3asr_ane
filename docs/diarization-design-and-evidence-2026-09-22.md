# Optional CPU speaker diarization: design and evidence

## Decision

The optional diarization path uses sherpa-onnx's maintained offline speaker
diarization API on CPU. It combines a learned pyannote segmentation model, a
learned 3D-Speaker voice embedding model, and clustering. It does not create
speaker turns from fixed windows, signal energy, ASR token position, or input
duration.

The runtime and both selected models are public and ungated:

- sherpa-onnx documents `OfflineSpeakerDiarization` and requires
  `sherpa-onnx>=1.10.28` for its Python diarization example. Version 1.13.8 was
  used for local validation. The project and Python package are Apache-2.0.
  Its official example constructs `OfflineSpeakerDiarizationConfig` from a
  segmentation model, embedding extractor, and `FastClusteringConfig`, then
  calls `process(samples).sort_by_start_time()`.
- `pyannote/segmentation-3.0` converted by the sherpa-onnx maintainer is MIT.
  The chosen FP32 ONNX file is 5,992,913 bytes.
- 3D-Speaker ERes2Net-Base is published under Apache-2.0. The chosen ONNX file
  is 39,593,761 bytes. The 3D-Speaker project identifies this exact model family
  as a released pretrained model trained on the 3D-Speaker data.

Primary references:

- <https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html>
- <https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/python.html>
- <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py>
- <https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speaker-identification-with-vad-dynamic.py>
- <https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/blob/main/LICENSE>
- <https://github.com/modelscope/3D-Speaker>

The alternative `sherpa-onnx-reverb-diarization-v1` segmentation model was not
selected because sherpa-onnx's official model page marks it non-commercial.

## Reproducible artifacts

Inference never downloads a model. `diarization.py` exports these immutable
artifact records and verifies byte size plus SHA-256 before loading sherpa-onnx:

| Purpose | Revision | Bytes | SHA-256 | License |
|---|---:|---:|---|---|
| pyannote segmentation 3.0 ONNX | `9403a6902bb58e3d5ae8c7e77c3422de279db2e0` | 5,992,913 | `220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079` | MIT |
| 3D-Speaker ERes2Net-Base ONNX | `8be2a75c9ed7a590538b268e46fbb65e1aa9d208` | 39,593,761 | `1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b` | Apache-2.0 |

The URLs include those full Hugging Face revisions. The repository's artifact
acquisition layer should copy each file to the exported local filename and use
`diarization_artifact_status()` as its read-only readiness check. The optional
Python dependency is `sherpa-onnx==1.13.8`; its macOS wheel requires the matching
`sherpa-onnx-core==1.13.8` native runtime wheel. Neither dependency belongs in
the ordinary ASR inference closure when diarization is disabled.

## Runtime semantics

`OfflineSpeakerDiarizer.diarize()` validates finite mono PCM, resamples to the
backend's required sample rate when needed, and returns sorted `SpeakerTurn`
values with model-measured starts, ends, and anonymous labels. The module-level
`diarize(samples, sample_rate=16000)` convenience API requires
`STD_QWEN3ASR_ANE_DIARIZATION_MODEL_DIR` to point at an already populated model
directory. Absence or corruption fails before runtime construction.

The pinned segmentation model records a 16,000 Hz sample rate, 991-sample
receptive field, and 270-sample output-frame shift in its ONNX metadata.
sherpa-onnx 1.13.8 (wheel git revision `11afbd00`) represents a turn boundary as
`frame_index * 270 / 16000 + 0.5 * 991 / 16000`; its padded final frame can
therefore end 30.96875 ms beyond the actual input. The adapter permits exactly
the integer support radius `ceil(991 / 2) = 496` model samples, or 31.000 ms at
16 kHz. Within that bound it intersects the raw turn with the actual input.
Anything farther outside fails as a backend error. This is a fixed model-grid
translation, not a duration-relative quality tolerance.

`OfflineSpeakerDiarizer.measure()` exposes immutable raw turns and structured
`DiarizationBoundaryAdjustment` records alongside adjusted turns. This lets the
Standard ASR adapter disclose a clamp in diagnostics/result `extra` without
replacing the original model evidence. A turn wholly inside padded support is
retained only as raw evidence, because it has no measured intersection with the
submitted audio. Existing speaker labels are copied unchanged through every
adjustment.

Primary implementation evidence for the boundary rule:

- <https://github.com/k2-fsa/sherpa-onnx/blob/11afbd00/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L630-L640>
- <https://github.com/k2-fsa/sherpa-onnx/blob/11afbd00/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L666-L711>
- <https://github.com/k2-fsa/sherpa-onnx/blob/11afbd00/sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.cc#L85-L112>

Overlapping turns are returned as overlapping intervals. The adapter does not
trim them, merge them, choose a dominant speaker, or duplicate one speaker's
identity onto another. Parent transcription/alignment code can map words or
audio spans against these intervals, but it must define how a word that overlaps
multiple simultaneous turns is represented.

Whole-recording diarization has one clustering space and therefore uses its
labels directly. Independently diarized streaming windows have unrelated local
labels. `SpeakerTracker` handles that case with a separate learned embedding
step:

1. Group measured turns by window-local speaker.
2. Exclude simultaneous-speaker intersections from identity audio while
   preserving those intersections in returned turns.
3. Embed the remaining speech with sherpa-onnx's 3D-Speaker extractor.
4. Match normalized embeddings to bounded, duration-weighted global centroids
   by cosine threshold, with one-to-one assignments inside each window.
5. Mint a new monotonic global label for an unmatched embedding. Raise when the
   configured maximum number of stable identities would be exceeded.

A local speaker with less than the configured minimum exclusive speech receives
a unique `unresolved_<window>_<local>` label. Such a label is valid only for that
window and makes no cross-window identity claim. This is preferable to guessing
from overlap-contaminated or too-short audio. Tracker state is bounded by
`max_speakers`; it stores one centroid and scalar speech weight per stable
identity, not prior PCM.

## Local native evidence

Validation ran on the current Apple Silicon machine with Python 3.12,
`sherpa-onnx==1.13.8`, `sherpa-onnx-core==1.13.8`, four CPU threads, and both
verified FP32 model files under `artifacts/auxiliary/diarization`. Audio came
from the existing CC-BY-4.0 LibriSpeech test-clean evaluation set at pinned
dataset revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`.

For one 46.505-second synthetic meeting, four known utterances were concatenated
with one second of silence between each:

| Expected source speaker | Expected interval | Measured principal interval | Measured cluster |
|---|---:|---:|---|
| 2300 | 0.000–9.125 | 0.166–8.030 | `speaker_00` |
| 260 | 10.125–22.030 | 10.679–21.580 | `speaker_01` |
| 2300 | 23.030–40.625 | 23.318–40.447 | `speaker_00` |
| 260 | 41.625–46.505 | 42.168–46.184 | `speaker_01` |

The model also emitted 8.030–9.042 as `speaker_01`, incorrectly assigning about
the final second of source speaker 2300. This validation proves that model-backed
boundaries and clustering execute locally and that the main two-speaker pattern
is recovered on this constructed case. It does not establish a general DER,
overlap-detection accuracy, language-independent quality, or production
threshold.

The same four clips were then diarized independently as four windows. Native
window-local labels were `speaker_01`, `speaker_00`, `speaker_00`, and
`speaker_01`, so direct label reuse would have switched identities. With the
learned tracker at cosine threshold 0.6, the global labels were respectively
`speaker_00`, `speaker_01`, `speaker_00`, and `speaker_01`; the final tracker
contained exactly two centroids. This is evidence for identity continuity on
two known voices, not a universal threshold calibration.

An additional 14.905-second mixture placed source speaker 2300 at 0 seconds and
source speaker 260 at 3 seconds. The learned backend returned `speaker_00` at
0.031–8.890 and `speaker_01` at 3.524–4.030 plus 4.705–14.459. Thus the public
adapter returned two real cross-speaker overlaps, 3.524–4.030 and
4.705–8.890, without trimming either track. This constructed mixture confirms
the overlap representation path; it is not a benchmark of overlap precision.

The release fixture's 46.505-second A/B/A/B meeting was also run as the exact
29.700-second low-energy first window plus its 16.805-second remainder. Before
boundary translation, sherpa returned 23.318470–29.730970 for the final turn in
the first window. The 30.970 ms excess matches the pinned model's centered-frame
support (including float32 rounding), so it was clamped to 23.318470–29.700000
and recorded as one `clamped` adjustment. The second window required no
adjustment. All adjusted local turns remained within their input windows, the
cross-window tracker retained two stable identities, and no actual speech-side
start or speaker label was changed.

The focused model-free suite covers pinned metadata and hashes,
read-only status, CPU-only sherpa configuration, resampling, malformed backend
output, overlap preservation, window label resets, one-to-one matching,
overlap exclusion from embeddings, unresolved short speech, and bounded tracker
capacity. It also pins the 30.970 ms regression, rejection immediately beyond
the 496-sample support radius, and padded-only turn handling. Native validation
should remain an opt-in release check because it
needs two external model artifacts and the optional runtime wheels.
