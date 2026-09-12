"""Record real ANE streaming events and run Standard ASR event compliance."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from evaluate import audio_samples
from standard_asr import discover_models
from standard_asr.compliance import check_event_sequence, check_transcription_result
from standard_asr.engine import AudioFormat, RuntimeParams


async def stream(engine, samples, *, realtime):
    events, arrival = [], []
    chunk_samples = 4000

    async def frames():
        for start in range(0, samples.size, chunk_samples):
            frame = samples[start : start + chunk_samples]
            if realtime:
                await asyncio.sleep(len(frame) / 16000)
            yield frame.astype("<f4").tobytes()

    session = engine.start_transcription(
        audio_format=AudioFormat(sample_rate=16000, encoding="pcm_f32le"),
        params=RuntimeParams(language="auto"),
    )
    started = perf_counter()
    async with session:
        session.feed(frames())
        async for event in session:
            events.append(event)
            arrival.append(perf_counter() - started)
            print(
                json.dumps(
                    {
                        "wall_seconds": arrival[-1],
                        "event": event.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    report = check_event_sequence(events, capabilities=engine.declared_capabilities)
    result = session.result()
    result_report = check_transcription_result(
        result, capabilities=engine.declared_capabilities
    )
    return {
        "events": [
            {"wall_seconds": stamp, "event": event.model_dump(mode="json")}
            for stamp, event in zip(arrival, events, strict=True)
        ],
        "event_compliance_passed": report.passed,
        "event_issues": [str(issue) for issue in report.issues],
        "result_compliance_passed": result_report.passed,
        "result": result.model_dump(mode="json"),
        "realtime_feed": realtime,
        "audio_seconds": len(samples) / 16000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--silence", action="store_true")
    args = parser.parse_args()
    engine = discover_models(strict=True).create(
        "std-qwen3asr-ane/1.7b", model_dir=args.model_dir, stream_chunk_seconds=1.0
    )
    report = {}
    try:
        started = perf_counter()
        engine.prepare()
        loaded = perf_counter() - started
        samples, digest = audio_samples(args.audio)
        report = asyncio.run(stream(engine, samples, realtime=args.realtime))
        report.update(model_load_seconds=loaded, audio_sha256=digest)
        batch = engine.transcribe((samples, 16000))
        report["batch_result"] = batch.model_dump(mode="json")
        report["stream_matches_batch"] = report["result"]["text"] == batch.text
        if args.silence:
            report["silence"] = []
            for seconds in (0.1, 0.5, 5.0, 29.99):
                result = engine.transcribe(
                    (np.zeros(round(seconds * 16000), np.float32), 16000)
                )
                report["silence"].append(
                    {"audio_seconds": seconds, "text": result.text}
                )
    finally:
        close_started = perf_counter()
        engine.close()
        report["explicit_close_seconds"] = perf_counter() - close_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return (
        0
        if report["event_compliance_passed"] and report["result_compliance_passed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
