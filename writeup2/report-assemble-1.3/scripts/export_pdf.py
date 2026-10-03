"""Render the assembled Markdown handoff to a self-contained PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import mistune
from weasyprint import CSS, HTML


ROOT = Path(__file__).resolve().parents[1]


def export_pdf(input_path: Path, output_path: Path) -> None:
    markdown = input_path.read_text(encoding="utf-8")
    html = mistune.create_markdown(
        escape=True,
        plugins=["table", "strikethrough", "footnotes"],
    )(markdown)
    page = f'<!doctype html><html lang="en"><head><meta charset="utf-8"></head><body>{html}</body></html>'
    HTML(string=page, base_url=input_path.parent.as_uri() + "/").write_pdf(
        output_path,
        stylesheets=[CSS(filename=ROOT / "templates" / "report-pdf.css")],
    )
    print(f"exported {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "report.pdf")
    args = parser.parse_args()
    export_pdf(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
