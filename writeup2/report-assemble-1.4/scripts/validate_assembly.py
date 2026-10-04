"""Validate the frozen evidence, assembled text, figures, links, and PDF."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
        "# Regional Models for Daily Soil-Moisture Estimation in Washington", "10.1109/AIIoT68874.2026.11569136",
        "## 1. Executive synthesis", "## 5. Primary temporal results", "## 6. In-state spatial generalization",
        "## 8. Regional model interpretation at ECE stations", "## 9. Robustness", "## 10. Paper contribution",
        "## 12. References", "## Appendix A.", "test period",
        "antecedent-precipitation index", "Washington validation", "seven held-out",
        "0.579737", "minimum station purity",
    ):
        if required not in report:
            raise ValueError(f"report is missing required content: {required}")
    expected_network_rows = (
        "| Spokane | NOAA USCRN |", "| Quinault | NOAA USCRN |", "| Darrington | NOAA USCRN |",
        "| Beaver Pass | SNOTEL |", "| Cayuse Pass | SNOTEL |", "| Paradise | SNOTEL |",
        "| Sourdough Gulch | SNOTEL |",
    )
    if any(row not in report for row in expected_network_rows):
        raise ValueError("station table does not show the expected NOAA USCRN/SNOTEL networks")
    if len(re.findall(r"SHA-256: `[0-9a-f]{64}`", ledger)) < 20:
        raise ValueError("claims ledger does not preserve source hashes alongside exact paths")
    external_text = "\n".join((report, ledger, (ROOT / "README.md").read_text(encoding="utf-8")))
    excluded_result = re.compile(r"test[ -]?selected|c0\s*=\s*0\s*,\s*c1\s*=\s*10|DELTA_BOOTSTRAP", re.IGNORECASE)
    if excluded_result.search(external_text):
        raise ValueError("excluded feature-addition result appears in reader-facing Markdown")
    if any(code in report for code in ("Guarded_Backbone54_k2", "Clustering_Backbone54_k2", "Clustering_V0_Full_k2")):
        raise ValueError("internal model identifiers appear in reader-facing report text")
    if re.search(r"SHA-256: `[0-9a-f]{64}`", report):
        raise ValueError("source hashes should appear in the claims ledger and manifest, not reader-facing report prose")
    if "V0" in report:
        raise ValueError("earlier-variant references are not part of the paired 1.4 report")
    if re.search(r"## 8\. Regional model interpretation at ECE stations\n\n###", report):
        raise ValueError("Section 8 must begin directly with the regional-model explanation")
    if "C22" not in ledger or "Unresolved future work" not in ledger:
        raise ValueError("claims ledger does not record station metadata and the open fallback work")
    if "C5" not in ledger or "paired" not in ledger:
        raise ValueError("claims ledger does not record the paired global comparison")
    if "paper2-final-evidence-1.0/ece_guarded" in report:
        raise ValueError("an obsolete ECE source path leaked into report text")
    # 1.4 paired-report checks: one name per model, every comparison
    # seed-paired, and every 1.2/early-1.3 alias (including the generic
    # "regional model" forms) stays retired.
    for required in (
        "station-majority rule",
        "station-majority shared-feature cluster-routed multi-regime model",
        "without station consistency guarantee",
        "shared-feature cluster-routed multi-regime model",
        "within a single regime",
        "bit-identical",
        "Existing single-regime global model",
        "Three-feature K-means grouping",
        "May–Oct",
        "SMAP_sm_pm_interp_lag1",
        "0.16",
    ):
        if required not in report:
            raise ValueError(f"report 1.4 is missing required explicit content: {required}")
    if "Comparison role" in report or "Historical reference:" in report or "| paired comparison |" in report:
        raise ValueError("report 1.4 revived the comparison-role taxonomy")
    retired = re.compile(
        r"primary regional model|unguarded|read this first|three main models"
        r"|station-majority regional model|two-regime regional model"
        r"|existing global model|contemporary global model"
        r"|changing-covariate grouping|dynamic-feature grouping"
        r"|single regional predictor|v0 regional model|regional model \(54",
        re.IGNORECASE,
    )
    for filename, text in (
        ("report.md", report),
        ("claims-ledger.md", ledger),
        ("README.md", (ROOT / "README.md").read_text(encoding="utf-8")),
    ):
        hit = retired.search(text)
        if hit:
            raise ValueError(f"retired 1.2 terminology in {filename}: {hit.group(0)!r}")

    for figure in FIGURES:
        path = ROOT / "figures" / figure
        dimensions = validate_figure(path)
        if f"figures/{figure}" not in report:
            raise ValueError(f"figure is unlinked: {figure}")
        print(f"validated {figure}: {dimensions[0]}×{dimensions[1]}")
    pdf = ROOT / "report.pdf"
    # report.pdf is an intentional build artifact (see .gitignore): it is
    # never committed. Validate it when present; otherwise skip with a notice
    # so a fresh clone still passes the tracked-content checks.
    if not pdf.is_file():
        print("skipping report.pdf checks: build artifact not present (run scripts/export_pdf.py to generate it)")
    else:
        if pdf.stat().st_size < 40_000 or pdf.open("rb").read(5) != b"%PDF-":
            raise ValueError("missing or invalid report PDF")
        reader = PdfReader(pdf)
        pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if (
            len(reader.pages) < 8
            or "Appendix A. Exact shared" not in pdf_text
            or "Regional model interpretation at ECE stations" not in pdf_text
        ):
            raise ValueError("PDF text or page coverage is incomplete")
        if excluded_result.search(pdf_text):
            raise ValueError("excluded feature-addition result appears in the PDF")
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
