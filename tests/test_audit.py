from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / ".agents" / "skills" / "dta-training-optimizer" / "scripts" / "audit_dta_training.py"
SPEC = importlib.util.spec_from_file_location("audit_dta_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AuditTests(unittest.TestCase):
    def test_detects_loop_and_loader_candidates(self) -> None:
        source = """
import torch
from torch.utils.data import DataLoader

def collate(paths):
    out = torch.empty(0)
    for path in paths:
        value = torch.load(path)
        out = torch.cat((out, value))
        value.cpu()
    return out

loader = DataLoader([], num_workers=2)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.py").write_text(source, encoding="utf-8")
            findings, errors = MODULE.audit_project("fixture", root)
        self.assertFalse(errors)
        codes = {finding.code for finding in findings}
        self.assertTrue({"load-in-loop", "cat-in-loop", "device-sync-in-loop", "unpinned-loader", "nonpersistent-workers"}.issubset(codes))

    def test_vectorizable_mask_is_reported(self) -> None:
        source = """
def generate_masks(adj, sizes):
    for index, size in enumerate(sizes):
        adj[index, size:] = 0
    return adj
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text(source, encoding="utf-8")
            findings, errors = MODULE.audit_project("fixture", root)
        self.assertFalse(errors)
        self.assertIn("python-mask-loop", {finding.code for finding in findings})


if __name__ == "__main__":
    unittest.main()
