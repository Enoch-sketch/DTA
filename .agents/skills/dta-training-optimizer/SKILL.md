---
name: dta-training-optimizer
description: >-
  Audit and optimize PyTorch training for PMHGT, PairSel-DTA, and MMCLKin-style
  drug-target affinity models. Use for training throughput, GPU utilization,
  data loading, mixed precision, synchronization, or memory work; do not use it
  to claim a scientific model contribution without an explicit method change.
---

# DTA training optimizer

Optimize the user's real training entrypoint, not a synthetic image pipeline. These
models combine molecular graphs, protein sequences, and 3D structure tensors; image
specific advice is relevant only when an actual image decoder or augmentation path is
present.

## Workflow

1. Preserve the experiment contract: split, sampler order, seed handling, batch size,
   optimizer, scheduler stepping, losses, validation cadence, and checkpoint selection.
2. Run `scripts/audit_dta_training.py` against the selected repository. Read
   [model playbook](references/model-playbook.md) only for the detected family.
3. Establish a short stable baseline after warm-up. Separate input/collation time,
   host-to-device transfer, forward/backward/update, validation, and checkpoint time.
4. Apply one low-risk candidate at a time, then measure the current combination.
   Prefer removing accidental synchronization and quadratic collation before changing
   precision or compilation.
5. Gate every retained change on representative outputs/losses and, for implementation
   rewrites, gradients or one-step parameter updates. Use the original batch and seed.
6. Report only the measured scope: local function, stable train step, epoch, or full run.
   Never convert a microbenchmark multiplier into an end-to-end training claim.

Run `scripts/benchmark_hotspots.py` to verify the two recurring implementation rewrites
(mask construction and tensor collation) on the current machine. Its results are
mechanism tests, not model-training speedups.

For MMCLKin, `scripts/optimized_mmclkin_collate.py` contains a drop-in, one-concatenation
collator and a CLI that checks every collated field against the original implementation
on real feature files before timing it. Do not integrate it unless that equivalence check
passes on the selected dataset.

## Decision boundaries

- Safe first candidates: `zero_grad(set_to_none=True)`, removal of per-epoch
  `empty_cache()`/`gc.collect()` after confirming no leak, batched metric transfer,
  list accumulation followed by one `torch.cat`, and device-native mask construction.
- Conditional candidates: pinned memory plus non-blocking transfer, persistent workers,
  bounded feature caching, BF16/FP16 autocast, fused optimizers, and `torch.compile`.
  Benchmark them on the actual GPU and data because PyG/dynamic shapes can reverse the
  expected benefit.
- Recipe changes such as batch size, gradient accumulation, sampler changes, reduced
  validation, approximate loss, or altered scheduler cadence are not silent engineering
  optimizations. Treat them as new experiments.
- Do not edit upstream model repositories unless the user authorizes that target. This
  skill may live in a separate repository and produce patches for review.

Before describing publication impact, read
[paper and reproducibility boundary](references/paper-boundary.md). For GPU measurements,
read [benchmark protocol](references/benchmark-protocol.md).
