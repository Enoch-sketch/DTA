#!/usr/bin/env python3
"""Microbenchmark recurring PMHGT/MMCLKin implementation patterns."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, repeats: int, device: torch.device) -> float:
    for _ in range(3):
        fn()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def legacy_mask(lengths: torch.Tensor, max_size: int, device: torch.device) -> torch.Tensor:
    out = torch.ones(len(lengths), max_size)
    for index, length in enumerate(lengths.tolist()):
        out[index, length:max_size] = 0
    return out.to(device)


def vectorized_mask(lengths: torch.Tensor, max_size: int, device: torch.device) -> torch.Tensor:
    lengths = lengths.to(device=device, dtype=torch.long)
    return (torch.arange(max_size, device=device).unsqueeze(0) < lengths.unsqueeze(1)).to(torch.float32)


def legacy_cat(parts: list[torch.Tensor]) -> torch.Tensor:
    out = torch.empty((0, parts[0].shape[1]), dtype=parts[0].dtype)
    for part in parts:
        out = torch.cat((out, part), dim=0)
    return out


def vectorized_cat(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(parts, dim=0)


def original_gcl(left: torch.Tensor, right: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    between = torch.exp((left @ right.T) / tau)
    diagonal = torch.diag(between)
    return (-torch.log(diagonal / between.sum(1)) - torch.log(diagonal / between.sum(0))).sum()


def stable_gcl(left: torch.Tensor, right: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    logits = (F.normalize(left, dim=1) @ F.normalize(right, dim=1).T) / tau
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels, reduction="sum") + F.cross_entropy(logits.T, labels, reduction="sum")


def result(name: str, before: float, after: float, equivalent: bool, max_abs_diff: float) -> dict:
    return {
        "name": name,
        "before_median_ms": before * 1000,
        "after_median_ms": after * 1000,
        "local_speedup": before / after,
        "equivalent": equivalent,
        "max_abs_diff": max_abs_diff,
        "scope": "microbenchmark_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    wants_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device("cuda" if wants_cuda else "cpu")
    generator = torch.Generator().manual_seed(args.seed)

    lengths = torch.randint(32, 1000, (128,), generator=generator)
    old_mask = legacy_mask(lengths, 1024, device)
    new_mask = vectorized_mask(lengths, 1024, device)
    mask_before = measure(lambda: legacy_mask(lengths, 1024, device), args.repeats, device)
    mask_after = measure(lambda: vectorized_mask(lengths, 1024, device), args.repeats, device)

    parts = [torch.randn((37 + index % 11, 64), generator=generator) for index in range(128)]
    old_cat = legacy_cat(parts)
    new_cat = vectorized_cat(parts)
    cat_before = measure(lambda: legacy_cat(parts), args.repeats, device)
    cat_after = measure(lambda: vectorized_cat(parts), args.repeats, device)

    left = torch.randn((128, 256), generator=generator, device=device)
    right = torch.randn((128, 256), generator=generator, device=device)
    old_loss = original_gcl(left, right)
    new_loss = stable_gcl(left, right)
    loss_before = measure(lambda: original_gcl(left, right), args.repeats, device)
    loss_after = measure(lambda: stable_gcl(left, right), args.repeats, device)

    payload = {
        "device": str(device),
        "torch": torch.__version__,
        "warning": "Local function timings are not end-to-end model-training speedups.",
        "results": [
            result("pmhgt_mask_construction", mask_before, mask_after, torch.equal(old_mask, new_mask), float((old_mask - new_mask).abs().max())),
            result("mmclkin_tensor_collation", cat_before, cat_after, torch.equal(old_cat, new_cat), float((old_cat - new_cat).abs().max())),
            result("stable_symmetric_gcl", loss_before, loss_after, torch.allclose(old_loss, new_loss, rtol=1e-5, atol=1e-5), float((old_loss - new_loss).abs())),
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
