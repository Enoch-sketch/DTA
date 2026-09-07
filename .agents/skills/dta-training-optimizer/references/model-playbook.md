# Model-specific playbook

Read only the section matching the current training entrypoint.

## PMHGT

Observed structure in the current project:

- `generate_masks` creates a CPU tensor, fills it in a Python loop, then copies it to the
  GPU every forward pass. Construct the boolean/float mask directly on `adj.device` with
  `arange < lengths`; verify masks and predictions exactly or within the project tolerance.
- DataLoaders use workers but do not request pinned memory or persistent workers, and
  batches use blocking `.to(device)`. Benchmark pinned memory + non-blocking transfer as
  one candidate; retain it only on the actual CUDA host.
- `torch.cuda.empty_cache()` and `gc.collect()` run every epoch. They can add allocator
  and host stalls and do not free live tensors. Remove them after a multi-epoch memory
  check shows no monotonic leak.
- Use `optimizer.zero_grad(set_to_none=True)`. Convert validation to
  `torch.inference_mode()` if it does not rely on autograd-side behavior.
- BF16 autocast is preferable to FP16 on capable recent hardware, but compare validation
  loss and gradients before keeping it. LSTM, graph scatter, and dynamic PyG paths make
  `torch.compile` a measured experiment rather than a default.

## MMCLKin

The highest-confidence CPU-side candidate is collation:

- The collator repeatedly concatenates growing tensors while loading every `.pt` feature.
  This copies accumulated data again on each sample and scales poorly with batch size.
  Collect tensors in lists, apply index offsets, and concatenate each field once.
- Validate every resulting field, shape, dtype, edge offset, batch vector, label, and
  model output against the original collator on the same feature paths.
- Use `scripts/optimized_mmclkin_collate.py` for the field-level equivalence and local
  timing check. Import its `collate_once` function only after the check passes.
- Feature files are loaded per sampled occurrence. Pair sampling can revisit examples.
  Measure storage wait, then consider a size-bounded per-worker cache; never preload the
  whole set without a RAM budget.
- Avoid `.item()` or `.cpu()` for every loss component on every train step. Accumulate
  detached device scalars and transfer at the existing logging interval or epoch end.
- Test pinned memory/non-blocking transfer only after the collated PyG object is confirmed
  pinnable. Persistent workers require `num_workers > 0`.

Use the numerically stable symmetric contrastive expression based on `logsumexp` or
cross-entropy only as a separately validated rewrite. Compare loss and gradients; it may
change floating-point rounding even when mathematically equivalent.

## PairSel-DTA on MMCLKin

PairSel inherits the MMCLKin collation and feature-I/O costs and adds two constraints:

- The point and pair streams use deterministic generators and, in the dual-stream path,
  distinct forward seeds. Do not fuse forwards if doing so changes dropout masks, batch
  normalization behavior, sampling order, or the sequential backward-accumulation contract.
- Current smoke scripts use `num_workers=0` and synchronize multiple loss scalars to CPU
  every step. First benchmark workers/caching and batched metric transfer.
- Gradient-norm diagnostics are intentional for the first batch. Keep them outside the
  timed stable window instead of deleting them.
- Preserve pair orientation, drug-balanced sampling weights, exact-pair membership,
  floor-valued point observations, and the one-optimizer-step dual-stream update.

## Order of experiments

1. Vectorized collation/masks with equivalence tests.
2. Remove avoidable synchronization and allocator cleanup.
3. DataLoader workers, persistent workers, pinning, non-blocking copies, and bounded cache.
4. BF16 autocast; then fused optimizer if supported.
5. Selective compilation of stable submodules, not an unconditional whole-model compile.
6. Batch-size or algorithmic changes only as separately named training recipes.
