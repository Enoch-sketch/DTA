#!/usr/bin/env python3
"""Read-only static audit for common DTA training bottlenecks."""

import argparse
import ast
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Finding:
    project: str
    file: str
    line: int
    code: str
    severity: str
    message: str


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


class AuditVisitor(ast.NodeVisitor):
    def __init__(self, project: str, relative_file: str) -> None:
        self.project = project
        self.relative_file = relative_file
        self.loop_depth = 0
        self.function_stack: list[str] = []
        self.findings: list[Finding] = []

    def add(self, node: ast.AST, code: str, severity: str, message: str) -> None:
        self.findings.append(
            Finding(self.project, self.relative_file, getattr(node, "lineno", 0), code, severity, message)
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        if node.name == "generate_masks" and any(isinstance(item, (ast.For, ast.While)) for item in ast.walk(node)):
            self.add(node, "python-mask-loop", "high", "Mask generation contains a Python loop; build it on-device with broadcasting.")
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        keyword_names = {item.arg for item in node.keywords}

        if name == "torch.cuda.empty_cache":
            self.add(node, "empty-cache", "medium", "Allocator cache is explicitly emptied; avoid this in a recurring training path unless required by measured pressure.")
        elif name == "gc.collect":
            self.add(node, "gc-collect", "medium", "Full Python GC is invoked; recurring calls can stall epoch boundaries.")
        elif name.endswith("zero_grad") and "set_to_none" not in keyword_names:
            self.add(node, "zero-grad-fill", "low", "Consider zero_grad(set_to_none=True) and verify one-step behavior.")
        elif name.endswith("DataLoader"):
            workers = next((item.value for item in node.keywords if item.arg == "num_workers"), None)
            positive_workers = isinstance(workers, ast.Constant) and isinstance(workers.value, int) and workers.value > 0
            if "pin_memory" not in keyword_names:
                self.add(node, "unpinned-loader", "low", "CUDA input pipeline candidate: benchmark pinned memory with non-blocking transfer.")
            if positive_workers and "persistent_workers" not in keyword_names:
                self.add(node, "nonpersistent-workers", "low", "Workers restart at epoch boundaries; benchmark persistent_workers=True.")

        if self.loop_depth:
            if name == "torch.cat":
                self.add(node, "cat-in-loop", "high", "torch.cat occurs inside a loop; accumulating lists then concatenating once can avoid repeated copies.")
            elif name == "torch.load":
                self.add(node, "load-in-loop", "high", "torch.load occurs inside a loop; measure feature I/O and consider workers or a bounded cache.")
            elif name.endswith(".item") or name.endswith(".cpu"):
                self.add(node, "device-sync-in-loop", "medium", "A scalar/data transfer occurs inside a loop and may synchronize CUDA.")
            elif name.endswith(".to") and "non_blocking" not in keyword_names:
                self.add(node, "blocking-transfer", "low", "Device transfer in a loop does not request non_blocking; pair with pinned memory before testing.")

        self.generic_visit(node)


def iter_python_files(root: Path) -> Iterable[Path]:
    ignored = {".git", "__pycache__", ".venv", "venv", "build", "dist"}
    for path in root.rglob("*.py"):
        if not any(part in ignored for part in path.parts):
            yield path


def audit_project(label: str, root: Path) -> tuple[list[Finding], list[dict[str, str]]]:
    findings: list[Finding] = []
    errors: list[dict[str, str]] = []
    for path in iter_python_files(root):
        relative = str(path.relative_to(root))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            errors.append({"project": label, "file": relative, "error": str(exc)})
            continue
        visitor = AuditVisitor(label, relative)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return findings, errors


def deduplicate_findings(findings: list[Finding]) -> list[Finding]:
    """Keep the first example of each mechanism in a file."""
    unique: dict[tuple[str, str, str], Finding] = {}
    for finding in findings:
        unique.setdefault((finding.project, finding.file, finding.code), finding)
    return list(unique.values())


def parse_project(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("project must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    root = Path(raw_path).expanduser().resolve()
    if not label or not root.is_dir():
        raise argparse.ArgumentTypeError(f"invalid project: {value}")
    return label, root


def markdown_report(payload: dict) -> str:
    lines = ["# DTA training static audit", "", "Static candidates require runtime validation on the target GPU.", ""]
    lines.append(f"Python: {payload['environment']['python']} ({payload['environment']['implementation']})")
    lines.append("")
    lines.append(f"Findings: {payload['summary']['finding_count']}; parse errors: {payload['summary']['parse_error_count']}")
    lines.append("")
    lines.append("| Project | Severity | Code | Location | Candidate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for finding in payload["findings"]:
        message = finding["message"].replace("|", "\\|")
        lines.append(f"| {finding['project']} | {finding['severity']} | {finding['code']} | `{finding['file']}:{finding['line']}` | {message} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", required=True, type=parse_project, metavar="LABEL=PATH")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    findings: list[Finding] = []
    errors: list[dict[str, str]] = []
    for label, root in args.project:
        project_findings, project_errors = audit_project(label, root)
        findings.extend(project_findings)
        errors.extend(project_errors)
    findings = deduplicate_findings(findings)
    severity_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda item: (item.project, severity_order[item.severity], item.file, item.line, item.code))
    payload = {
        "environment": {"python": platform.python_version(), "implementation": platform.python_implementation()},
        "projects": [label for label, _ in args.project],
        "summary": {"finding_count": len(findings), "parse_error_count": len(errors)},
        "findings": [asdict(item) for item in findings],
        "parse_errors": errors,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_report(payload), encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
