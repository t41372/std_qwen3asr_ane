# Default native decoding regression check

The current runtime and commit `884c22e` were imported in separate Python processes and run against the same existing compiled Qwen3-ASR bundle. The archived baseline source lives at `.cache/native-baseline-884c22e`; it was created with `git archive 884c22e std_qwen3asr_ane/src` and did not alter the working tree.

The comparison ran the two official smoke fixtures and the first two rows, in manifest order, from each cached English and Chinese held-out manifest. All six default greedy runs had identical text, raw text, token IDs, and EOS ID. The command, source and bundle hashes, input checksums, and each equality result are recorded in [`native-baseline-884c22e.json`](./native-baseline-884c22e.json).

This verifies that the new native cancellation, metadata, language-control, and guidance code preserves the default path when those optional features are absent. It is not a new quality or performance evaluation.
