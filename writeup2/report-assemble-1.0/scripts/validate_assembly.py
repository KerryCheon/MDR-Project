"""Validate the frozen evidence, assembled text, figures, links, and PDF."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path
from urllib.parse import unquote, urlsplit

import report_data as rd
from pypdf import PdfReader
from assemble_report import TOKEN, render


ROOT = rd.ASSEMBLY_ROOT
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FIGURES = (
    "main_temporal_and_loso.png",
    "ece_policies_and_stations.png",
    "ece_guarded_daily_overlay.png",
)


def validate_links(path: Path) -> int:
    content = path.read_text(encoding="utf-8")
    count = 0
    for raw in LINK.findall(content):
        target = raw.split(' "', 1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme in {"http", "https", "mailto"}:
            continue
        if parsed.scheme:
            raise ValueError(f"unsupported link scheme in {path}: {target}")
        if not parsed.path:
            continue
        resolved = (path.parent / unquote(parsed.path)).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"broken link in {path}: {target}")
        count += 1
    return count


def validate_figure(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"invalid PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width < 800 or height < 500:
        raise ValueError(f"undersized report figure: {path}: {width}×{height}")
    return width, height


def main() -> int:
    context = rd.build_context()
    for name in ("report", "claims-ledger"):
        path = ROOT / f"{name}.md"
        expected = render(ROOT / "templates" / f"{name}.md", context)
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            raise ValueError(f"{path} differs from current source artifacts/template")
        if TOKEN.search(actual):
            raise ValueError(f"unresolved template token in {path}")
        print(f"validated deterministic {name}.md: {len(actual.splitlines())} lines")

    manifest = json.loads((ROOT / "provenance" / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest != rd.source_manifest():
        raise ValueError("source manifest hashes or metadata differ from current inputs")
    if not manifest["inputs"] or not all(len(item["sha256"]) == 64 for item in manifest["inputs"].values()):
        raise ValueError("source manifest lacks SHA-256 hashes")
    print(f"validated {len(manifest['inputs'])} source hashes and protocol metadata")

    report = (ROOT / "report.md").read_text(encoding="utf-8")
    ledger = (ROOT / "claims-ledger.md").read_text(encoding="utf-8")
    for required in (
        "## 1. Executive synthesis", "## 5. Primary temporal results", "## 6. In-state spatial generalization",
        "## 8. ECE collaboration stations", "## 9. Robustness", "## 10. Paper contribution",
        "## 12. References", "## Appendix A.", "sensitivity-anchored", "non-deployable",
        "test period", "G_API class 0", "Washington validation", "seven held-out",
    ):
        if required not in report:
            raise ValueError(f"report is missing required content: {required}")
    if "C21" not in ledger or "Unresolved future work" not in ledger:
        raise ValueError("claims ledger does not record the paired global gap")
    if "paper2-final-evidence-1.0/ece_guarded" in report:
        raise ValueError("obsolete 1.0 ECE result leaked into report")

    for figure in FIGURES:
        path = ROOT / "figures" / figure
        dimensions = validate_figure(path)
        if f"figures/{figure}" not in report:
            raise ValueError(f"figure is unlinked: {figure}")
        print(f"validated {figure}: {dimensions[0]}×{dimensions[1]}")
    pdf = ROOT / "report.pdf"
    if pdf.stat().st_size < 40_000 or pdf.open("rb").read(5) != b"%PDF-":
        raise ValueError("missing or invalid report PDF")
    reader = PdfReader(pdf)
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if len(reader.pages) < 10 or "Appendix A. Exact shared" not in pdf_text or "ECE collaboration stations" not in pdf_text:
        raise ValueError("PDF text or page coverage is incomplete")
    image_count = sum(len(page.images) for page in reader.pages)
    if image_count < len(FIGURES):
        raise ValueError(f"PDF has {image_count} images; expected all {len(FIGURES)} figures")
    print(f"validated report.pdf: {len(reader.pages)} pages, {image_count} embedded images, {pdf.stat().st_size:,} bytes")

    for filename in ("report.md", "claims-ledger.md", "README.md", "provenance/literature-verification.md"):
        path = ROOT / filename
        print(f"validated {validate_links(path)} local links in {filename}")
    report_hash = hashlib.sha256((ROOT / "report.md").read_bytes()).hexdigest()
    print(f"report.md sha256 {report_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
