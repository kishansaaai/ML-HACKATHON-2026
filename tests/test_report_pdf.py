"""Tests for scripts/build_report_pdf.py (assembles the final report PDF)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("reportlab")

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("build_report_pdf", REPO / "scripts" / "build_report_pdf.py")
mod = importlib.util.module_from_spec(SPEC)
sys.modules["build_report_pdf"] = mod
SPEC.loader.exec_module(mod)


def _page_count(path: Path) -> int:
    data = path.read_bytes()
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except ImportError:
        return data.count(b"/Type /Page") - data.count(b"/Type /Pages")


def _make_png(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot([0, 1, 2], [0, 1, 0])
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _write_sections(src: Path, png: Path) -> None:
    src.mkdir(parents=True, exist_ok=True)
    (src / "sec1_overview.md").write_text(
        "DRAFT prepared with AI assistance - rewrite me\n\n"
        "# Overview\n\nSome **bold**, *italic* and `code` text with unicode → ≥.\n\n"
        "## Goals\n\n- first item\n- second item that is long " + "word " * 40 + "\n\n1. one\n2. two\n",
        encoding="utf-8",
    )
    (src / "sec2_data_preprocessing.md").write_text(
        "# Data\n\n| Model | Macro F1 | A very long column heading that has to wrap | Notes |\n"
        "|---|---|---|---|\n"
        "| LightGBM | 0.91 | " + "wrapping text " * 20 + " | a_very_long_unbroken_identifier_" + "x" * 60 + " |\n"
        "| Baseline | N/A (no probabilities) | 1 | n |\n\n"
        f"![Real figure]({png.as_posix()})\n\n"
        "![Absent figure](results/plots/does_not_exist.png)\n",
        encoding="utf-8",
    )
    (src / "sec3_models.md").write_text(
        "# Models\n\n```python\n" + "x = some_function(" + "argument, " * 30 + ")\nprint(x)\n```\n\n"
        "> a quote\n\n---\n\nEnd.\n",
        encoding="utf-8",
    )


def test_build_pdf_with_synthetic_sections(tmp_path, capsys):
    src = tmp_path / "report"
    png = tmp_path / "fig.png"
    _make_png(png)
    _write_sections(src, png)
    out = tmp_path / "out" / "report.pdf"

    result = mod.build(src, out)

    assert result == out and out.exists()
    assert out.read_bytes()[:5] == b"%PDF-"
    assert _page_count(out) > 1
    printed = capsys.readouterr().out
    # sections 4..8 are absent: warning, not an error
    assert "missing section file, skipped: sec4_bayesian_optimisation.md" in printed
    # absent figure is reported and rendered as a placeholder
    assert "missing figure: results/plots/does_not_exist.png" in printed


def test_draft_note_and_placeholders_in_text(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    src = tmp_path / "report"
    png = tmp_path / "fig.png"
    _make_png(png)
    _write_sections(src, png)
    out = tmp_path / "r.pdf"
    mod.build(src, out)
    reader = pypdf.PdfReader(str(out))
    first = reader.pages[0].extract_text()
    assert "UE24CS352A" in first
    assert "[fill in]" in first
    assert "Contains draft sections" in first
    all_text = "\n".join(p.extract_text() for p in reader.pages)
    assert "missing figure: results/plots/does_not_exist.png" in all_text


def test_empty_or_absent_source_does_not_raise(tmp_path):
    out = tmp_path / "x.pdf"
    mod.build(tmp_path / "nope", out)
    assert out.read_bytes().startswith(b"%PDF")
