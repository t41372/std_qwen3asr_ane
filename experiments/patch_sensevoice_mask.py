"""Isolate an infinite-mask hypothesis in a separate, experimental compiled asset.

Only the MIL constant changes; model I/O metadata and weight bytes stay intact.
This is a host-runtime diagnostic, not a replacement for a reproducible source
conversion before distributing an optimized SenseVoice artifact.
The tested patch did not eliminate the NaNs; it is a recorded negative result.
"""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("artifacts/drafts/sensevoice-small/models")
    source = root / "SenseVoiceSmall_int8.mlmodelc"
    output = root / "SenseVoiceSmall_int8_finite.mlmodelc"
    if output.exists():
        raise FileExistsError(output)
    original = (source / "model.mil").read_text()
    needle = "val = tensor<fp16, []>(-inf)"
    if original.count(needle) != 1 or original.count("-inf") != 1:
        raise ValueError(
            "The pinned graph no longer has exactly one infinite mask constant"
        )
    subprocess.run(["/bin/cp", "-cR", str(source), str(output)], check=True)
    changed = original.replace(needle, "val = tensor<fp16, []>(-0x1.388p+13)")
    (output / "model.mil").write_text(changed)
    report = {
        "purpose": "infinite versus -10000 attention mask diagnostic",
        "source": str(source),
        "output": str(output),
        "original_mil_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "patched_mil_sha256": hashlib.sha256(changed.encode()).hexdigest(),
        "changed_constants": 1,
        "weights_modified": False,
    }
    (root / "finite-mask-patch.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
