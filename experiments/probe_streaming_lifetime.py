"""Write real streaming events incrementally and observe native cleanup while idle."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from evaluate import audio_samples
from standard_asr.compliance import check_event_sequence
from standard_asr.engine import AudioFormat, RuntimeParams
from std_qwen3asr_ane.plugin import create_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--idle", type=float, default=5)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.open("x")
    started = time.monotonic()

    def record(kind, **data):
        line = json.dumps(
            {"kind": kind, "seconds": time.monotonic() - started, **data},
            ensure_ascii=False,
        )
        log.write(line + "\n")
        log.flush()
        print(line, flush=True)

    engine = create_engine(model_dir=args.model_dir, stream_chunk_seconds=1.0)
    record("load_start")
    engine.prepare()
    record("load_done")
    samples, digest = audio_samples(args.audio)
    record("audio", sha256=digest, audio_seconds=len(samples) / 16000)

    async def run():
        for index in range(args.rounds):
            record("session_start", index=index)
            session = engine.start_transcription(
                audio_format=AudioFormat(sample_rate=16000, encoding="pcm_f32le"),
                params=RuntimeParams(language="auto"),
            )

            async def chunks():
                for offset in range(0, len(samples), 4000):
                    chunk = samples[offset : offset + 4000]
                    await asyncio.sleep(len(chunk) / 16000)
                    yield chunk.astype("<f4").tobytes()

            events = []
            async with session:
                session.feed(chunks())
                async for event in session:
                    events.append(event)
                    record("event", index=index, event=event.model_dump(mode="json"))
            report = check_event_sequence(
                events, capabilities=engine.declared_capabilities
            )
            record(
                "session_done",
                index=index,
                compliance=report.passed,
                issues=[str(issue) for issue in report.issues],
                result=session.result().model_dump(mode="json"),
            )
            if not report.passed or events[-1].type != "done":
                raise RuntimeError(
                    "Streaming failed or emitted an invalid event sequence"
                )
            await asyncio.sleep(args.idle)
            record("session_idle_done", index=index)

    asyncio.run(run())
    record("close_start")
    engine._runtime.close(timeout=5)
    record("close_done")
    time.sleep(args.idle)
    record("completed")
    log.close()


if __name__ == "__main__":
    main()
