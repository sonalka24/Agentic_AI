#!/usr/bin/env python3
"""Generate JavaDoc-like API documentation for Python modules as Markdown."""

from __future__ import annotations

import ast
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class FunctionDoc:
    name: str
    signature: str
    doc: str


@dataclass
class ClassDoc:
    name: str
    signature: str
    doc: str
    methods: list[FunctionDoc] = field(default_factory=list)


@dataclass
class ModuleDoc:
    module_path: Path
    doc: str
    functions: list[FunctionDoc] = field(default_factory=list)
    classes: list[ClassDoc] = field(default_factory=list)


def _format_annotation(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _format_arg(arg: ast.arg, default: ast.AST | None = None) -> str:
    text = arg.arg
    ann = _format_annotation(arg.annotation)
    if ann:
        text = f"{text}: {ann}"
    if default is not None:
        try:
            text = f"{text} = {ast.unparse(default)}"
        except Exception:
            text = f"{text} = ..."
    return text


def _signature_from_function(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    posonly = list(fn.args.posonlyargs)
    regular = list(fn.args.args)
    defaults = list(fn.args.defaults)
    default_offset = len(posonly) + len(regular) - len(defaults)

    for i, arg in enumerate(posonly + regular):
        default_idx = i - default_offset
        default = defaults[default_idx] if default_idx >= 0 else None
        args.append(_format_arg(arg, default))
        if i + 1 == len(posonly):
            args.append("/")

    if fn.args.vararg:
        args.append(f"*{_format_arg(fn.args.vararg)}")
    elif fn.args.kwonlyargs:
        args.append("*")

    for kwarg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        args.append(_format_arg(kwarg, default))

    if fn.args.kwarg:
        args.append(f"**{_format_arg(fn.args.kwarg)}")

    ret = _format_annotation(fn.returns)
    ret_text = f" -> {ret}" if ret else ""
    return f"{fn.name}({', '.join(a for a in args if a)}){ret_text}"


def _signature_from_class(cls: ast.ClassDef) -> str:
    if not cls.bases:
        return cls.name
    try:
        bases = ", ".join(ast.unparse(base) for base in cls.bases)
    except Exception:
        bases = ", ".join("Base" for _ in cls.bases)
    return f"{cls.name}({bases})"


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if any(part in {"__pycache__", ".venv", "venv"} for part in path.parts):
            continue
        if path.name == Path(__file__).name:
            continue
        yield path


def parse_module(path: Path, root: Path) -> ModuleDoc:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_doc = ast.get_docstring(tree) or "No module docstring."
    out = ModuleDoc(module_path=path.relative_to(root), doc=module_doc)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.functions.append(
                FunctionDoc(
                    name=node.name,
                    signature=_signature_from_function(node),
                    doc=ast.get_docstring(node) or "No documentation.",
                )
            )
        elif isinstance(node, ast.ClassDef):
            cls = ClassDoc(
                name=node.name,
                signature=_signature_from_class(node),
                doc=ast.get_docstring(node) or "No documentation.",
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cls.methods.append(
                        FunctionDoc(
                            name=item.name,
                            signature=_signature_from_function(item),
                            doc=ast.get_docstring(item) or "No documentation.",
                        )
                    )
            out.classes.append(cls)
    return out


def render_markdown(modules: list[ModuleDoc]) -> str:
    lines = [
        "# API Documentation",
        "",
        "Auto-generated JavaDoc-style reference for Python modules.",
        "",
    ]
    for module in modules:
        lines.extend(
            [
                f"## Module `{module.module_path.as_posix()}`",
                "",
                module.doc,
                "",
            ]
        )
        if module.functions:
            lines.extend(["### Functions", ""])
            for fn in module.functions:
                lines.extend(
                    [
                        f"#### `{fn.signature}`",
                        "",
                        fn.doc,
                        "",
                    ]
                )
        if module.classes:
            lines.extend(["### Classes", ""])
            for cls in module.classes:
                lines.extend(
                    [
                        f"#### `{cls.signature}`",
                        "",
                        cls.doc,
                        "",
                    ]
                )
                if cls.methods:
                    lines.extend(["Methods:", ""])
                    for method in cls.methods:
                        lines.extend(
                            [
                                f"- `{method.signature}`: {method.doc}",
                            ]
                        )
                    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown API docs from Python source.")
    parser.add_argument("--root", default="/app", help="Project root to scan.")
    parser.add_argument("--output", default="/app/docs/API_REFERENCE.md", help="Output Markdown file.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    modules = [parse_module(path, root) for path in _iter_python_files(root)]
    markdown = render_markdown(modules)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(modules)} module docs to {output}")


if __name__ == "__main__":
    main()
