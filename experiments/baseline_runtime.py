"""Load an immutable baseline package under an isolated Python namespace."""

import argparse
import importlib
import importlib.util
import runpy
import sys
from pathlib import Path

from std_qwen3asr_ane.bundle import digest


def load_baseline_runtime(package: Path):
    package = package.resolve()
    name = "_frozen_asr_" + digest(package / "runtime.py")[:16]
    specification = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        specification.loader.exec_module(module)
        return importlib.import_module(f"{name}.runtime").CoreMLRuntime
    finally:
        sys.dont_write_bytecode = previous


def prediction_models(runtime):
    method = getattr(runtime, "_prediction_models", None)
    if method is not None:
        return method()
    return (runtime.frontend, runtime.encoder, *runtime.decoders, runtime.lm_head)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide an experiment script after --")
    script = Path(__file__).parent / command[0]
    if not script.is_file() or script.resolve().parent != Path(__file__).resolve().parent:
        parser.error("Expected a script in experiments/")
    from std_qwen3asr_ane import runtime

    runtime.CoreMLRuntime = load_baseline_runtime(args.source)
    sys.argv = [str(script), *command[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
