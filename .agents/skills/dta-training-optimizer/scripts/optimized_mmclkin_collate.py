#!/usr/bin/env python3
"""One-concatenation MMCLKin collator with a real-feature equivalence benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data


CAT_FIELDS = (
    "po_dist", "po_theta", "po_phi", "po_tau",
    "mol_dist", "mol_theta", "mol_phi", "mol_tau",
    "pr_dist", "pr_theta", "pr_phi", "pr_tau",
    "mol_atoms_feats", "mol_edges_feats", "mol_coords_feats",
    "poc_coords_feats", "poc_atoms_feats_s", "poc_edges_feats_s",
    "poc_atoms_feats_v", "poc_edges_feats_v", "pro_atoms_feats_s",
    "pro_atoms_feats_v", "pro_coords_feats", "pro_edges_feats_s",
    "pro_edges_feats_v",
)


def collate_once(paths: list[str]) -> Data:
    """Preserve the official MMCLKin collate contract without growing torch.cat calls."""
    values = {name: [] for name in CAT_FIELDS}
    pro_edges: list[torch.Tensor] = []
    poc_edges: list[torch.Tensor] = []
    mol_edges: list[torch.Tensor] = []
    pocinpro_flat: list[torch.Tensor] = []
    atominmol_flat: list[torch.Tensor] = []
    subwinsmi_flat: list[torch.Tensor] = []
    atominmol_indexes: list[torch.Tensor] = []
    pocinpro_indexes: list[torch.Tensor] = []
    subwinsmi_indexes: list[torch.Tensor] = []
    smiles: list[torch.Tensor] = []
    pockets: list[torch.Tensor] = []
    proteins: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    mol_batch: list[int] = []
    poc_batch: list[int] = []
    pro_batch: list[int] = []
    subword_offset = protein_offset = pocket_offset = atom_offset = 0

    for batch_index, path in enumerate(paths):
        feature = torch.load(path, map_location="cpu", weights_only=False)
        for name in CAT_FIELDS:
            values[name].append(getattr(feature, name))

        atom_index = torch.as_tensor(feature.atoinmol_index, dtype=torch.long)
        pocket_index = torch.as_tensor(feature.pocinpro_index, dtype=torch.long)
        subword_index = torch.as_tensor(feature.subwinsmi_index, dtype=torch.long)
        atominmol_indexes.append(atom_index)
        pocinpro_indexes.append(pocket_index[pocket_index <= 1022])
        subwinsmi_indexes.append(subword_index)

        pro_edges.append(feature.pro_edge_index + protein_offset)
        poc_edges.append(feature.poc_edge_index + pocket_offset)
        mol_edges.append(feature.mol_edge_index + atom_offset)
        pocinpro_flat.append(pocket_index.unsqueeze(0) + protein_offset)
        atominmol_flat.append(atom_index.unsqueeze(0) + atom_offset)
        subwinsmi_flat.append(subword_index.unsqueeze(0) + subword_offset)

        protein_count = feature.pro_atoms_feats_s.shape[0]
        pocket_count = feature.poc_atoms_feats_s.shape[0]
        atom_count = feature.mol_atoms_feats.shape[0]
        pro_batch.extend([batch_index] * protein_count)
        poc_batch.extend([batch_index] * pocket_count)
        mol_batch.extend([batch_index] * atom_count)
        protein_offset += protein_count
        pocket_offset += pocket_count
        atom_offset += atom_count
        subword_offset += feature.mol_embedding.shape[1] - 2

        labels.append(feature.com_affinity.unsqueeze(0))
        smiles.append(feature.mol_embedding.squeeze(0))
        pockets.append(feature.poc_token_repre.squeeze(0))
        proteins.append(feature.pro_token_repre.squeeze(0))

    output = {name: torch.cat(parts, dim=0) for name, parts in values.items()}
    output.update(
        y=torch.cat(labels, dim=0),
        pro_edge_index=torch.cat(pro_edges, dim=1).long(),
        poc_edge_index=torch.cat(poc_edges, dim=1).long(),
        mol_edge_index=torch.cat(mol_edges, dim=1).long(),
        mol_embedding=pad_sequence(smiles).permute(1, 0, 2),
        poc_token_repre=pad_sequence(pockets).permute(1, 0, 2),
        pro_token_repre=pad_sequence(proteins).permute(1, 0, 2),
        mol_batch=torch.tensor(mol_batch, dtype=torch.int64),
        poc_batch=torch.tensor(poc_batch, dtype=torch.int64),
        pro_batch=torch.tensor(pro_batch, dtype=torch.int64),
        atominmol_indexes=atominmol_indexes,
        pocinpro_indexes=pocinpro_indexes,
        subwinsmi_indexes=subwinsmi_indexes,
        pocinpro_index=torch.cat(pocinpro_flat, dim=1).long(),
        atominmol_index=torch.cat(atominmol_flat, dim=1).long(),
        subwinsmi_index=torch.cat(subwinsmi_flat, dim=1).long(),
    )
    return Data(**output)


def load_original(path: Path):
    spec = importlib.util.spec_from_file_location("mmclkin_original_dataset_loader", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collate


def compare_value(left, right) -> tuple[bool, float]:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, float("inf")
        difference = 0.0 if left.numel() == 0 else float((left - right).abs().max())
        return torch.equal(left, right), difference
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        comparisons = [compare_value(a, b) for a, b in zip(left, right)]
        return all(item[0] for item in comparisons), max((item[1] for item in comparisons), default=0.0)
    return left == right, 0.0 if left == right else float("inf")


def compare_data(original: Data, optimized: Data) -> dict:
    original_fields = original.to_dict()
    optimized_fields = optimized.to_dict()
    keys_match = original_fields.keys() == optimized_fields.keys()
    fields = {}
    for key in sorted(original_fields.keys() | optimized_fields.keys()):
        if key not in original_fields or key not in optimized_fields:
            fields[key] = {"equal": False, "max_abs_diff": None}
            continue
        equal, difference = compare_value(original_fields[key], optimized_fields[key])
        fields[key] = {"equal": bool(equal), "max_abs_diff": difference}
    return {"equivalent": keys_match and all(item["equal"] for item in fields.values()), "fields": fields}


def timed(fn, paths: list[str], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn(paths)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-loader", required=True, type=Path)
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1:
        parser.error("batch size and repeats must be positive")
    paths = [str(path) for path in sorted(args.feature_dir.glob("*.pt"))[: args.batch_size]]
    if len(paths) != args.batch_size:
        parser.error("feature directory does not contain enough .pt files")

    original_collate = load_original(args.original_loader)
    original = original_collate(paths)
    optimized = collate_once(paths)
    comparison = compare_data(original, optimized)
    if not comparison["equivalent"]:
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 1

    original_seconds = timed(original_collate, paths, args.repeats)
    optimized_seconds = timed(collate_once, paths, args.repeats)
    payload = {
        "equivalent": True,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "original_median_seconds": original_seconds,
        "optimized_median_seconds": optimized_seconds,
        "local_speedup": original_seconds / optimized_seconds,
        "scope": "collate_plus_cached_local_feature_reads_only",
        "warning": "Run an end-to-end GPU training benchmark before claiming training speedup.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
