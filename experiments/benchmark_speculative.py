"""Compare target-verified ANE speculation with serial target greedy decoding."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from evaluate import audio_samples, manifest_rows
from std_qwen3asr_ane.runtime import CoreMLRuntime
from std_qwen3asr_ane.speculative import greedy_speculative_decode
from std_qwen3asr_ane.transcript_draft import TranscriptDraft


class DecoderCursor:
    def __init__(self, runtime, prepared, batch_head=None):
        self.runtime = runtime
        self.states = prepared.states
        self.batch_head = batch_head
        self.head_width = (
            batch_head.get_spec().description.input[0].type.multiArrayType.shape[-1]
            if batch_head
            else None
        )
        if batch_head is not None:
            # The compact head returns one (max, index) pair per vocabulary
            # chunk; the chunk size must be the bundle head's, or the ids would
            # be wrong. Compiled models expose no spec, so probe both heads once
            # per runtime with a zero state (outside any measured repeat).
            cached = getattr(runtime, "_vocabulary_chunks", None)
            if cached is None:
                width_hidden = runtime.embeddings.shape[1]
                probe = runtime.lm_head.predict(
                    {"hidden_states": np.zeros((1, width_hidden, 1, 1), np.float16)}
                )
                keys = sorted(probe, key=lambda name: int(name.removeprefix("logits_")))
                sizes = {int(np.asarray(probe[key]).size) for key in keys[:-1]}
                if len(sizes) != 1:
                    raise ValueError("Bundle LM head chunks are not uniform")
                compact = batch_head.predict(
                    {
                        "hidden_states": np.zeros(
                            (1, width_hidden, 1, self.head_width), np.float32
                        )
                    }
                )
                if compact["max_values"].shape[1] != len(keys):
                    raise ValueError(
                        "Compact head chunk count differs from the bundle head"
                    )
                cached = runtime._vocabulary_chunks = (sizes.pop(), len(keys))
            self.chunk_size = cached[0]

    def step(self, tokens, position):
        embeddings = np.concatenate(
            [self.runtime._embedding(token) for token in tokens], axis=-1
        )
        hidden = self.runtime._decode_step(
            embeddings, position, self.states, all_rows=True
        )
        if self.batch_head is not None:
            width = self.head_width
            if len(tokens) > width:
                raise ValueError("Verifier block exceeds the vocabulary head width")
            padded = np.zeros((*hidden.shape[:-1], width), np.float32)
            padded[..., : len(tokens)] = hidden
            output = self.batch_head.predict({"hidden_states": padded})
            values, indices = output["max_values"][0], output["max_indices"][0]
            chunks = np.argmax(values, axis=0)
            return [
                int(chunks[index] * self.chunk_size + indices[chunks[index], index])
                for index in range(len(tokens))
            ]
        return [hidden[..., index : index + 1] for index in range(len(tokens))]

    def choose(self, hidden):
        if isinstance(hidden, int):
            return hidden
        return self.runtime._next_token(hidden)


def validate_state_transfer(source, destination):
    """Only transfer between equal weights, layout and uniformly partitioned models."""
    for key in (
        "model_id",
        "source_revision",
        "max_sequence_length",
        "residual_scale",
        "head_dim",
    ):
        if source.manifest[key] != destination.manifest[key]:
            raise ValueError(f"State transfer requires matching {key}")
    if len(source.decoders) != len(destination.decoders) or 28 % len(source.decoders):
        raise ValueError("This experiment requires matching uniform decoder partitions")
    for runtime in (source, destination):
        if "compiled_from" not in runtime.manifest:
            raise ValueError(
                "State transfer requires compiled source-weight provenance"
            )

    def weights(runtime):
        records = {
            row["compiled"]: row for row in runtime.manifest["compiled_from"]["models"]
        }
        return [
            sorted(
                value
                for name, value in records[path]["source_sha256"].items()
                if name.endswith("/weight.bin")
            )
            for path in runtime.manifest["decoder_partitions"]
        ]

    if weights(source) != weights(destination) or any(
        not row for row in weights(source)
    ):
        raise ValueError(
            "Prefill and generation must use byte-identical decoder weights"
        )


def transfer_prompt(prepared, destination):
    """Use public MLState read/write; never pass state handles between models."""
    states = tuple(model.make_state() for model in destination.decoders)
    layers = 28 // len(states)
    for source, target in zip(prepared.states, states, strict=True):
        for layer in range(layers):
            for kind in ("key", "value"):
                name = f"{kind}_{layer}"
                value = source.read_state(name)
                if (
                    value.shape != target.read_state(name).shape
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        "State shape differs or prefill cache contains non-finite values"
                    )
                target.write_state(name, value)
    return replace(prepared, states=states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=Path("artifacts/qwen3-asr-1.7b-compiled")
    )
    parser.add_argument(
        "--draft", type=Path, default=Path("artifacts/qwen3-asr-0.6b-draft-compiled")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookahead", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--verify-head", type=Path)
    parser.add_argument("--prefill-target", type=Path)
    parser.add_argument("--serial-target", type=Path)
    parser.add_argument("--draft-mode", choices=("ane", "oracle"), default="ane")
    parser.add_argument("--state-transfer", choices=("copy", "share"), default="copy")
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("Use a fresh output and positive repeats")
    started = perf_counter()
    target = CoreMLRuntime(args.target)
    draft = None
    batch_head = None
    prefill_target = None
    serial_target = None
    try:
        if args.draft_mode == "ane":
            draft = CoreMLRuntime(args.draft, model_id="Qwen/Qwen3-ASR-0.6B")
        if args.serial_target:
            serial_target = CoreMLRuntime(args.serial_target)
        if args.prefill_target:
            prefill_target = CoreMLRuntime(args.prefill_target)
            validate_state_transfer(prefill_target, target)
        if args.verify_head:
            import coremltools as ct
            from std_qwen3asr_ane.runtime import PersistentInputModel

            batch_head = PersistentInputModel(
                ct.models.MLModel(
                    str(args.verify_head), compute_units=ct.ComputeUnit.CPU_AND_NE
                )
            )
        load_seconds = perf_counter() - started
        if draft is not None and (
            target.tokenizer.to_str() != draft.tokenizer.to_str()
            or target.eos_token_ids != draft.eos_token_ids
        ):
            raise ValueError("Target and draft tokenizers or EOS definitions differ")
        if not 0 <= args.lookahead < target.token_batch_size:
            raise ValueError(
                "Lookahead must fit held token plus proposals in the target block"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as output:
            for row in manifest_rows(args.manifest):
                samples, digest = audio_samples(Path(row["audio_path"]))
                for repeat in range(-1, args.repeats):
                    serial = (serial_target or target).transcribe(
                        samples, language=None, max_new_tokens=256
                    )
                    begin = perf_counter()
                    target_prompt = (prefill_target or target).prepare_prompt(
                        samples, language=None, max_new_tokens=256
                    )
                    transfer_started = perf_counter()
                    if prefill_target is not None and args.state_transfer == "copy":
                        target_prompt = transfer_prompt(target_prompt, target)
                    target_prepared = perf_counter()
                    if args.draft_mode == "oracle":
                        from std_qwen3asr_ane.languages import LANGUAGE_NAMES

                        raw = f"language {LANGUAGE_NAMES[row['language']]}<asr_text>{row['reference']}"
                        draft_decoder = TranscriptDraft(
                            target.tokenizer.encode(raw, add_special_tokens=False).ids,
                            eos_token=target.tokenizer.token_to_id("<|im_end|>"),
                        )
                        draft_position = 0
                    else:
                        draft_prompt = draft.prepare_prompt(
                            samples, language=None, max_new_tokens=256
                        )
                        if target_prompt.token_ids != draft_prompt.token_ids:
                            raise ValueError("Draft and target prompt IDs differ")
                        draft_decoder = DecoderCursor(draft, draft_prompt)
                        draft_position = len(draft_prompt.token_ids)
                    draft_prepared = perf_counter()
                    result = greedy_speculative_decode(
                        DecoderCursor(target, target_prompt, batch_head),
                        draft_decoder,
                        target_prompt.hidden,
                        target_position=len(target_prompt.token_ids),
                        draft_position=draft_position,
                        eos_token_ids=frozenset(target.eos_token_ids),
                        max_new_tokens=256,
                        lookahead=args.lookahead,
                    )
                    end = perf_counter()
                    record = {
                        "id": row["id"],
                        "repeat": repeat,
                        "load_seconds": load_seconds,
                        "audio_sha256": digest,
                        "lookahead": args.lookahead,
                        "draft_mode": args.draft_mode,
                        "state_transfer": args.state_transfer,
                        "measurement_scope": "verification lower bound with ground-truth proposals; excludes obtaining a transcript"
                        if args.draft_mode == "oracle"
                        else "complete ANE draft and target inference",
                        "verify_head": str(args.verify_head)
                        if args.verify_head
                        else None,
                        "prefill_target": str(args.prefill_target)
                        if args.prefill_target
                        else None,
                        "state_transfer_seconds": target_prepared - transfer_started,
                        "serial_seconds": serial.timings["total_seconds"],
                        "speculative_seconds": end - begin,
                        "target_prepare_seconds": target_prepared - begin,
                        "draft_prepare_seconds": draft_prepared - target_prepared,
                        "generation_seconds": end - draft_prepared,
                        "exact_token_parity": result.token_ids == serial.token_ids,
                        "serial_tokens": list(serial.token_ids),
                        "speculative": asdict(result),
                    }
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(
                        json.dumps(
                            {
                                key: value
                                for key, value in record.items()
                                if key not in ("serial_tokens", "speculative")
                            }
                        ),
                        flush=True,
                    )
                    if not record["exact_token_parity"]:
                        raise AssertionError(
                            "Batched verifier changed serial target token IDs"
                        )
    finally:
        target_prompt = draft_prompt = None
        draft_decoder = None
        if batch_head is not None:
            batch_head.close()
        if prefill_target is not None:
            prefill_target.close()
        if serial_target is not None:
            serial_target.close()
        if draft is not None:
            draft.close()
        target.close()


if __name__ == "__main__":
    main()
