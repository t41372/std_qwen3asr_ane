"""Build a greedy vocabulary head returning only each chunk's winner.

Local indices stay int32; casting an 8192-entry index to FP16 would silently
lose exact integers above 2048. Ties choose the first index within each chunk.
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from std_qwen3asr_ane.conversion.decoder import LanguageHead, SourceWeights


class CompactHead(LanguageHead):
    def forward(self, hidden_states):
        normalized = self.norm(hidden_states)
        values, indices = [], []
        for head in self.heads:
            value, index = torch.max(head(normalized).squeeze(2), dim=1)
            values.append(value)
            indices.append(index.to(torch.int32))
        return torch.stack(values, dim=1), torch.stack(indices, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    source = Path("artifacts/source/Qwen3-ASR-1.7B")
    config = json.loads((source / "config.json").read_text())["thinker_config"][
        "text_config"
    ]
    weights = SourceWeights(source)
    module = CompactHead(config).eval()
    module.norm.weight.data.copy_(weights.get("thinker.model.norm.weight"))
    embedding = weights.get("thinker.model.embed_tokens.weight")
    offset = 0
    for head in module.heads:
        count = head.out_channels
        head.weight.data.copy_(embedding[offset : offset + count, :, None, None])
        offset += count
    del embedding
    example = torch.zeros(1, config["hidden_size"], 1, args.token_batch_size)
    model = ct.convert(
        torch.jit.trace(module, example, check_trace=False),
        inputs=[
            ct.TensorType(name="hidden_states", shape=example.shape, dtype=np.float16)
        ],
        outputs=[
            ct.TensorType(name="max_values", dtype=np.float16),
            ct.TensorType(name="max_indices", dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    model.save(str(args.output))


if __name__ == "__main__":
    main()
