# Benchmark protocol

Use the actual target GPU and representative feature files.

1. Record code revision, command, environment, GPU, precision, batch, workers, and seed.
2. Keep the compared work identical. Exclude process startup, first file-cache fill, and
   compilation warm-up from the stable-step window, but report those costs separately.
3. Time data wait, transfer, forward/backward/update, validation, and checkpoints. CUDA
   timings require synchronization or CUDA events at measurement boundaries.
4. Repeat short windows when the observed difference is close to normal variance. Report
   samples/second and peak allocated/reserved memory, plus median and range where possible.
5. Compare outputs, loss, gradients, or a one-step update according to the change's risk.
6. Retain the candidate only if the real train-step or epoch result improves and the
   correctness gate passes. A faster standalone mask or collator is only supporting
   evidence until it changes end-to-end timing.

Do not benchmark while another process materially loads the same GPU. Do not extrapolate
CPU-only results to CUDA throughput.
