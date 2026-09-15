"""Materialize natural <=6-second speech from pinned CV/FLEURS source shards.

Selection uses source order, duration and nonempty reference only. Speech is
never cropped or accelerated. Raw audio remains outside version control.
"""

import argparse
import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq
import soundfile as sf


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire(source):
    path = Path(
        source.get(
            "local_path",
            f"artifacts/evaluation/sources/{source['dataset']}-{source['language']}-{source['sha256'][:16]}.parquet",
        )
    )
    if not path.exists():
        temporary = path.with_suffix(".partial")
        received, started = 0, time.monotonic()
        with urlopen(source["url"], timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > source["bytes"] or time.monotonic() - started > 600:
                    raise ValueError("Download exceeded pinned size/time budget")
                output.write(chunk)
        if received != source["bytes"] or digest(temporary) != source["sha256"]:
            raise ValueError("Download differs from pinned source")
        temporary.replace(path)
    if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
        raise ValueError("Cached parquet differs from pinned source")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("Count must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    grouped = defaultdict(list)
    for source in json.loads(args.sources.read_text()):
        grouped[source["language"]].append(source)
    records = []
    report = {
        "complete": False,
        "sources_sha256": digest(args.sources),
        "selection": "first unique nonempty-reference clips <=6 seconds; no model output used",
        "languages": [],
    }
    try:
        for language, sources in grouped.items():
            selected, seen, speakers, scans = [], set(), set(), []
            count = sources[0].get("count", args.count)
            for source in sources:
                path = acquire(source)
                report["active_source"] = source
                ordinal = 0
                with pq.ParquetFile(path) as parquet:
                    for batch in parquet.iter_batches(batch_size=16):
                        for row in batch.to_pylist():
                            index, ordinal = ordinal, ordinal + 1
                            blob = row["audio"]["bytes"]
                            reference = row[source.get("reference_field", "sentence")]
                            info = sf.info(io.BytesIO(blob))
                            audio_hash = hashlib.sha256(blob).hexdigest()
                            if (
                                not 0 < info.duration <= 6
                                or not isinstance(reference, str)
                                or not reference.strip()
                                or audio_hash in seen
                            ):
                                continue
                            samples, rate = sf.read(
                                io.BytesIO(blob), dtype="float32", always_2d=True
                            )
                            if len(samples) / rate > 6:
                                continue
                            seen.add(audio_hash)
                            if row.get("client_id"):
                                speakers.add(row["client_id"])
                            identity = (
                                f"{source['dataset']}-{language}-{source['sha256'][:8]}-{index:05d}"
                            )
                            suffix = (
                                ".flac"
                                if info.format == "FLAC"
                                else ".wav"
                                if info.format == "WAV"
                                else ".mp3"
                            )
                            audio = args.output / f"{identity}{suffix}"
                            audio.write_bytes(blob)
                            selected.append(
                                {
                                    "id": identity,
                                    "audio_path": str(audio.resolve()),
                                    "audio_sha256": audio_hash,
                                    "reference": reference,
                                    "language": language,
                                    "duration_seconds": len(samples) / rate,
                                    "split": "test",
                                    "dataset_revision": source["revision"],
                                    "source_sha256": source["sha256"],
                                    "source_row": index,
                                }
                            )
                            if len(selected) == count:
                                break
                        if len(selected) == count:
                            break
                scans.append({**source, "scanned": ordinal})
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError(f"Only {len(selected)} eligible {language} clips")
            records.extend(selected)
            report["languages"].append(
                {
                    "language": language,
                    "selected": len(selected),
                    "speakers": len(speakers) if speakers else None,
                    "sources": scans,
                }
            )
            print(json.dumps({"language": language, "selected": len(selected)}), flush=True)
        manifest = args.output / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        report["manifest_sha256"] = digest(manifest)
        report["complete"] = True
        report.pop("active_source", None)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
