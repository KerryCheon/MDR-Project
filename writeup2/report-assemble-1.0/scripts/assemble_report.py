"""Assemble the Paper 2 report and claims ledger from saved evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import report_data as rd


TOKEN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def render(template_path: Path, context: dict[str, str]) -> str:
    template = template_path.read_text(encoding="utf-8")
    missing = sorted(set(TOKEN.findall(template)) - set(context))
    if missing:
        raise ValueError(f"unresolved template keys in {template_path}: {missing}")
    rendered = TOKEN.sub(lambda match: context[match.group(1)], template)
    if TOKEN.search(rendered):
        raise ValueError(f"template tokens remain in {template_path}")
    return rendered.rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="Also export report.pdf")
    args = parser.parse_args()
    context = rd.build_context()
    for name in ("report", "claims-ledger"):
        target = rd.ASSEMBLY_ROOT / f"{name}.md"
        template = rd.ASSEMBLY_ROOT / "templates" / f"{name}.md"
        target.write_text(render(template, context), encoding="utf-8")
        print(f"assembled {target}")
    manifest_path = rd.ASSEMBLY_ROOT / "provenance" / "source_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rd.source_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    if args.pdf:
        from export_pdf import export_pdf

        export_pdf(rd.ASSEMBLY_ROOT / "report.md", rd.ASSEMBLY_ROOT / "report.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
