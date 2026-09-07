# Paper and reproducibility boundary

The skill is an engineering tool, not a scientific model component, so it does not need
to be cited as a method contribution. The resulting execution choices may still belong
in reproducibility details.

Usually report:

- GPU model/count, framework and CUDA versions;
- precision mode (FP32, TF32, BF16, or FP16) and loss scaling when applicable;
- effective batch size, accumulation, distributed strategy, seed/determinism policy;
- compilation or custom kernels when they materially affect reproducibility;
- training time or hardware cost when the paper makes an efficiency claim.

Usually no model-method discussion is needed for pinned memory, worker count, asynchronous
copies, `set_to_none`, logging frequency, allocator cleanup, or mathematically equivalent
collation/mask rewrites, provided the experiment contract and accepted numerical behavior
are unchanged.

An optimization must be described as part of the method or an ablation when it changes
the loss, sampler, data split, batch semantics, validation/checkpoint selection, scheduler
cadence, model computation, approximation, or final scientific conclusions. Mixed
precision can be an engineering setting, but it must be disclosed because it can affect
numerics.
