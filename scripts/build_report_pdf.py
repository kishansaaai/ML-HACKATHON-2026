"""Assemble the final PathSnap technical report PDF from Markdown section files.

Usage:
    python scripts/build_report_pdf.py [--src docs/report] [--out PATH]

Input : docs/report/*.md in filename order (sec1_overview.md ... sec8_ai_disclosure.md).
        Missing files are skipped with a printed warning.
Output: output/pdf/PathSnap_technical_report_final.pdf by default.

Only reportlab is required (plus matplotlib's bundled DejaVu fonts when available, so
that Unicode symbols render; otherwise the built-in Helvetica/Courier fonts are used).
A small Markdown subset is rendered by the code below: headings (#..####), paragraphs,
bullet / numbered lists, pipe tables, fenced code blocks, block quotes, horizontal rules,
bold / italic / inline code / links, and standalone images ![alt](path).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO_ROOT / "docs" / "report"
DEFAULT_OUT = REPO_ROOT / "output" / "pdf" / "PathSnap_technical_report_final.pdf"

TITLE = "PathSnap: Distributed Map-Matching and Route Reconstruction"
COURSE = "Machine Learning"
INSTITUTION = "PES University, Department of Computer Science and Engineering, RR Campus"
EXPECTED_FILES = [
    "sec1_overview.md",
    "sec2_data_eda.md",
    "sec3_models.md",
    "sec4_bayes_opt.md",
    "sec5_hmm.md",
    "sec6_folium.md",
    "sec7_api_ui.md",
    "sec8_ai_disclosure.md",
]
DRAFT_MARKER = "DRAFT prepared with AI assistance"
DRAFT_NOTE = "Contains draft sections: the team must rewrite them in its own words."

ACCENT = colors.HexColor("#1f4e79")
GRID = colors.HexColor("#9aa5b1")
HEAD_BG = colors.HexColor("#dce6f1")
CODE_BG = colors.HexColor("#f3f4f6")

# --------------------------------------------------------------------------- fonts

BODY, BODY_B, BODY_I, BODY_BI, MONO = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique", "Courier"
_UNICODE_FONTS = False


def _setup_fonts() -> None:
    """Register DejaVu fonts shipped with matplotlib (Unicode-safe); fall back to Helvetica."""
    global BODY, BODY_B, BODY_I, BODY_BI, MONO, _UNICODE_FONTS
    try:
        import matplotlib

        fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        files = {
            "DejaVuSans": "DejaVuSans.ttf",
            "DejaVuSans-Bold": "DejaVuSans-Bold.ttf",
            "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
            "DejaVuSans-BoldOblique": "DejaVuSans-BoldOblique.ttf",
            "DejaVuSansMono": "DejaVuSansMono.ttf",
            "DejaVuSansMono-Bold": "DejaVuSansMono-Bold.ttf",
        }
        if not all((fdir / f).exists() for f in files.values()):
            return
        for name, f in files.items():
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(fdir / f)))
        pdfmetrics.registerFontFamily(
            "DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
            italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-BoldOblique",
        )
        BODY, BODY_B, BODY_I, BODY_BI, MONO = (
            "DejaVuSans", "DejaVuSans-Bold", "DejaVuSans-Oblique", "DejaVuSans-BoldOblique", "DejaVuSansMono",
        )
        _UNICODE_FONTS = True
    except Exception:  # pragma: no cover - font setup must never break the build
        pass


def _safe(text: str) -> str:
    """With only the built-in fonts, replace characters outside latin-1."""
    if _UNICODE_FONTS:
        return text
    return text.encode("latin-1", "replace").decode("latin-1")


# --------------------------------------------------------------------------- styles

def make_styles() -> dict[str, ParagraphStyle]:
    s: dict[str, ParagraphStyle] = {}
    s["body"] = ParagraphStyle("body", fontName=BODY, fontSize=9.5, leading=13.5, spaceAfter=5)
    s["h1"] = ParagraphStyle("Heading1", parent=s["body"], fontName=BODY_B, fontSize=17, leading=21,
                             textColor=ACCENT, spaceBefore=0, spaceAfter=10, keepWithNext=1)
    s["h2"] = ParagraphStyle("Heading2", parent=s["body"], fontName=BODY_B, fontSize=13, leading=16,
                             textColor=ACCENT, spaceBefore=10, spaceAfter=5, keepWithNext=1)
    s["h3"] = ParagraphStyle("Heading3", parent=s["body"], fontName=BODY_B, fontSize=11, leading=14,
                             spaceBefore=8, spaceAfter=3, keepWithNext=1)
    s["h4"] = ParagraphStyle("Heading4", parent=s["body"], fontName=BODY_BI, fontSize=10, leading=13,
                             spaceBefore=6, spaceAfter=2, keepWithNext=1)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=16, bulletIndent=4, spaceAfter=2)
    s["quote"] = ParagraphStyle("quote", parent=s["body"], fontName=BODY_I, leftIndent=12,
                                borderPadding=(2, 2, 2, 6), textColor=colors.HexColor("#444444"))
    s["code"] = ParagraphStyle("code", fontName=MONO, fontSize=7.5, leading=9.5, backColor=CODE_BG,
                               borderColor=GRID, borderWidth=0.4, borderPadding=4, spaceBefore=4, spaceAfter=8)
    s["caption"] = ParagraphStyle("caption", parent=s["body"], fontName=BODY_I, fontSize=8, leading=10,
                                  alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=8)
    s["missing"] = ParagraphStyle("missing", parent=s["body"], fontName=BODY_B, textColor=colors.HexColor("#b00020"),
                                  backColor=colors.HexColor("#fdecea"), borderColor=colors.HexColor("#b00020"),
                                  borderWidth=0.6, borderPadding=6, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10)
    s["title"] = ParagraphStyle("title", parent=s["body"], fontName=BODY_B, fontSize=26, leading=32,
                                alignment=TA_CENTER, textColor=ACCENT, spaceAfter=10)
    s["subtitle"] = ParagraphStyle("subtitle", parent=s["body"], fontSize=13, leading=18, alignment=TA_CENTER,
                                   spaceAfter=4)
    s["note"] = ParagraphStyle("note", parent=s["body"], fontName=BODY_B, textColor=colors.HexColor("#8a4b00"),
                               backColor=colors.HexColor("#fff4e0"), borderColor=colors.HexColor("#e08a00"),
                               borderWidth=0.8, borderPadding=7, alignment=TA_CENTER, spaceBefore=14, spaceAfter=14)
    s["toc_title"] = ParagraphStyle("toc_title", parent=s["h1"], keepWithNext=0)
    s["toc0"] = ParagraphStyle("toc0", parent=s["body"], fontName=BODY_B, fontSize=10, leading=15, spaceBefore=3,
                               leftIndent=0, rightIndent=14)
    s["toc1"] = ParagraphStyle("toc1", parent=s["body"], fontSize=9, leading=12, leftIndent=14, rightIndent=14)
    s["cell"] = ParagraphStyle("cell", parent=s["body"], fontSize=8, leading=10, spaceAfter=0, splitLongWords=1)
    s["cellh"] = ParagraphStyle("cellh", parent=s["cell"], fontName=BODY_B)
    return s


# --------------------------------------------------------------------------- inline markup

_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITAL_RE = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def inline(text: str) -> str:
    """Convert inline Markdown to reportlab paragraph markup."""
    text = _safe(text)
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = _CODE_RE.sub(stash, text)
    text = escape(text)
    text = _LINK_RE.sub(
        lambda m: (f'<a href="{m.group(2)}" color="#1f4e79">{m.group(1)}</a>'
                   if m.group(2).startswith(("http://", "https://")) else m.group(1)),
        text,
    )
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITAL_RE.sub(r"<i>\1</i>", text)
    text = re.sub(r"\x00(\d+)\x00",
                  lambda m: f'<font face="{MONO}" size="8.2" color="#7a1f3d">{escape(codes[int(m.group(1))])}</font>',
                  text)
    return text


# --------------------------------------------------------------------------- flowables

class Heading(Paragraph):
    """Paragraph that remembers its outline level (used for TOC / bookmarks)."""

    def __init__(self, text: str, style: ParagraphStyle, level: int):
        super().__init__(text, style)
        self.level = level


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str, **kw):
        super().__init__(filename, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                         topMargin=20 * mm, bottomMargin=20 * mm, title=TITLE,
                         author="PathSnap team", subject="UE24CS352A " + COURSE, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([
            PageTemplate(id="first", frames=[frame], onPage=self._no_footer),
            PageTemplate(id="later", frames=[frame], onPage=self._footer),
        ])
        self._key = 0

    def beforeDocument(self):
        self._key = 0  # bookmark keys must be identical on every multiBuild pass

    @staticmethod
    def _no_footer(canvas, doc):
        pass

    @staticmethod
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY, 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(doc.leftMargin, 11 * mm, _safe("PathSnap technical report | UE24CS352A"))
        canvas.drawRightString(doc.leftMargin + doc.width, 11 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.4)
        canvas.line(doc.leftMargin, 14.5 * mm, doc.leftMargin + doc.width, 14.5 * mm)
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Heading) and flowable.level <= 2:
            text = flowable.getPlainText()
            key = f"h{self._key}"
            self._key += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=flowable.level - 1, closed=True)
            self.notify("TOCEntry", (flowable.level - 1, text, self.page, key))


# --------------------------------------------------------------------------- Markdown parsing

_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_BULLET = re.compile(r"^(\s*)([-*+])\s+(.*)")
_NUMBERED = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)")
_IMAGE = re.compile(r"^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$")
_HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*#*\s*$")


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line)]


def resolve_image(ref: str, src_dir: Path) -> Path | None:
    ref = ref.split()[0].strip("<>") if ref.strip() else ref
    p = Path(ref)
    candidates = [p] if p.is_absolute() else [REPO_ROOT / p, src_dir / p, Path.cwd() / p]
    for c in candidates:
        if c.is_file():
            return c
    return None


class Builder:
    def __init__(self, styles: dict[str, ParagraphStyle], width: float, height: float, src_dir: Path):
        self.st = styles
        self.width = width
        self.height = height
        self.src_dir = src_dir
        self.missing_figures: list[str] = []

    # -- blocks
    def image(self, alt: str, ref: str):
        path = resolve_image(ref, self.src_dir)
        if path is None:
            self.missing_figures.append(ref)
            print(f"WARNING: missing figure: {ref}")
            return [Paragraph(escape(_safe(f"missing figure: {ref}")), self.st["missing"])]
        try:
            from reportlab.lib.utils import ImageReader

            iw, ih = ImageReader(str(path)).getSize()
            # scale to page width, but cap the height so the figure always fits on a page
            scale = min(self.width / iw, (self.height * 0.72) / ih)
            img = Image(str(path), width=iw * scale, height=ih * scale)
        except Exception as exc:
            self.missing_figures.append(ref)
            print(f"WARNING: could not read figure {ref}: {exc}")
            return [Paragraph(escape(_safe(f"missing figure: {ref}")), self.st["missing"])]
        items = [img]
        if alt.strip():
            items.append(Paragraph(inline(alt), self.st["caption"]))
        else:
            items.append(Spacer(1, 6))
        return [KeepTogether(items)]

    def table(self, rows: list[list[str]]):
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        plain = [[re.sub(r"[`*]", "", c) for c in r] for r in rows]
        fs = 8 if ncol <= 5 else 7.2 if ncol <= 8 else 6.6
        while True:
            # minimum width per column = longest unbreakable word (header is bold) + padding
            mins = []
            for j in range(ncol):
                w = 0.0
                for i, r in enumerate(plain):
                    face = BODY_B if i == 0 else BODY
                    for word in r[j].split():
                        w = max(w, pdfmetrics.stringWidth(_safe(word), face, fs))
                mins.append(min(w, self.width * 0.45) + 8)
            if sum(mins) <= self.width or fs <= 5.5:
                break
            fs -= 0.4
        lead = fs + 2
        cell = ParagraphStyle("c", parent=self.st["cell"], fontSize=fs, leading=lead)
        cellh = ParagraphStyle("ch", parent=self.st["cellh"], fontSize=fs, leading=lead)
        data = [[Paragraph(inline(c) if c else "", cellh if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
        if sum(mins) >= self.width:  # extreme case: squeeze proportionally (long words then split)
            widths = [self.width * m / sum(mins) for m in mins]
        else:
            weights = [max(len(r[j]) for r in plain) ** 0.7 + 1 for j in range(ncol)]
            extra = self.width - sum(mins)
            widths = [m + extra * w / sum(weights) for m, w in zip(mins, weights)]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, GRID),
            ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
        ]))
        return [t, Spacer(1, 8)]

    def code(self, lines: list[str]):
        style = self.st["code"]
        usable = self.width - 2 * style.borderPadding - 2
        char_w = pdfmetrics.stringWidth("M", MONO, style.fontSize)
        per_line = max(20, int(usable / char_w))
        out: list[str] = []
        for raw in lines:
            raw = _safe(raw.expandtabs(4)).rstrip()
            if not raw:
                out.append("")
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            cont = " " * min(indent + 2, per_line // 3)
            first = True
            while len(raw) > per_line:
                cut = per_line
                out.append(raw[:cut])
                raw = cont + raw[cut:]
                first = False
            out.append(raw)
        return [Preformatted("\n".join(out) if out else " ", style)]

    # -- document
    def parse(self, text: str, number: int | None):
        st = self.st
        flow: list = []
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        i, n = 0, len(lines)
        para: list[str] = []
        list_item: list | None = None  # [indent, bullet, [text lines]]
        first_h1_done = False

        def flush_para():
            nonlocal para
            if para:
                flow.append(Paragraph(inline(" ".join(s.strip() for s in para)), st["body"]))
                para = []

        def flush_item():
            nonlocal list_item
            if list_item:
                indent, bullet, parts = list_item
                style = ParagraphStyle("li", parent=st["bullet"], leftIndent=16 + 14 * indent,
                                       bulletIndent=4 + 14 * indent)
                flow.append(Paragraph(inline(" ".join(parts)), style, bulletText=bullet))
                list_item = None

        def flush_all():
            flush_para()
            flush_item()

        while i < n:
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith("<!--"):  # HTML comment (possibly multi-line)
                while i < n and "-->" not in lines[i]:
                    i += 1
                i += 1
                continue

            if stripped.startswith("```") or stripped.startswith("~~~"):
                flush_all()
                fence = stripped[:3]
                i += 1
                block: list[str] = []
                while i < n and not lines[i].strip().startswith(fence):
                    block.append(lines[i])
                    i += 1
                i += 1
                flow.extend(self.code(block))
                continue

            if not stripped:
                flush_all()
                i += 1
                continue

            m = _HEADING.match(stripped)
            if m:
                flush_all()
                level = len(m.group(1))
                title = m.group(2)
                if level == 1 and number is not None and not re.match(r"^\d+(\.\d+)*[.)]?\s", title):
                    title = f"{number}. {title}"
                if level == 1 and not first_h1_done:
                    first_h1_done = True
                flow.append(Heading(inline(title), st[f"h{level}"], level))
                i += 1
                continue

            if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", line):
                flush_all()
                t = Table([[""]], colWidths=[self.width], rowHeights=[4])
                t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.5, GRID)]))
                flow.append(t)
                i += 1
                continue

            m = _IMAGE.match(line)
            if m:
                flush_all()
                flow.extend(self.image(m.group(1), m.group(2)))
                i += 1
                continue

            if "|" in stripped and i + 1 < n and _TABLE_SEP.match(lines[i + 1]) and "|" in lines[i + 1]:
                flush_all()
                rows = [_split_row(stripped)]
                i += 2
                while i < n and lines[i].strip() and "|" in lines[i]:
                    rows.append(_split_row(lines[i]))
                    i += 1
                flow.extend(self.table(rows))
                continue

            if stripped.startswith(">"):
                flush_all()
                quote = []
                while i < n and lines[i].strip().startswith(">"):
                    quote.append(lines[i].strip().lstrip(">").strip())
                    i += 1
                flow.append(Paragraph(inline(" ".join(quote)), st["quote"]))
                continue

            mb, mn = _BULLET.match(line), _NUMBERED.match(line)
            if mb or mn:
                flush_all()
                mm_ = mb or mn
                indent = min(len(mm_.group(1).expandtabs(4)) // 2, 3)
                bullet = "•" if mb else f"{mm_.group(2)}."
                if not _UNICODE_FONTS and mb:
                    bullet = "-"
                list_item = [indent, bullet, [mm_.group(3).strip()]]
                i += 1
                continue

            if list_item is not None and line.startswith((" ", "\t")):
                list_item[2].append(stripped)
                i += 1
                continue

            flush_item()
            para.append(line)
            i += 1

        flush_all()
        return flow


# --------------------------------------------------------------------------- title page / assembly

def title_page(st: dict[str, ParagraphStyle], width: float, drafts: list[str], date: str) -> list:
    team = Table(
        [[Paragraph("<b>Team</b>", st["body"])],
         [Paragraph("Team member names and SRNs: [fill in]", st["body"])]],
        colWidths=[width * 0.8],
    )
    team.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, ACCENT),
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    flow: list = [
        Spacer(1, 45 * mm),
        Paragraph(inline("PathSnap"), st["title"]),
        Paragraph(escape(_safe("Distributed Map-Matching and Route Reconstruction")), st["subtitle"]),
        Paragraph("Technical Report", st["subtitle"]),
        Spacer(1, 12 * mm),
        Paragraph("Course code: UE24CS352A", st["subtitle"]),
        Paragraph(escape(_safe(COURSE)), st["subtitle"]),
        Paragraph(escape(_safe(INSTITUTION)), st["subtitle"]),
        Spacer(1, 12 * mm),
        team,
        Spacer(1, 8 * mm),
        Paragraph(f"Date: {date}", st["subtitle"]),
    ]
    if drafts:
        names = ", ".join(drafts)
        flow.append(Paragraph(escape(_safe(DRAFT_NOTE)) + f"<br/><font size='8' face='{BODY}'>({escape(_safe(names))})</font>",
                              st["note"]))
    flow.append(NextPageTemplate("later"))
    flow.append(PageBreak())
    return flow


def build(src: Path = DEFAULT_SRC, out: Path = DEFAULT_OUT) -> Path:
    """Build the PDF; returns the output path."""
    _setup_fonts()
    src, out = Path(src), Path(out)
    st = make_styles()
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = ReportDoc(str(out))

    # Which files exist? Expected set first (warn on missing), then any extra sec*.md / *.md present.
    present = {p.name: p for p in sorted(src.glob("*.md"))} if src.is_dir() else {}
    if not src.is_dir():
        print(f"WARNING: source directory not found: {src}")
    for name in EXPECTED_FILES:
        if name not in present:
            print(f"WARNING: missing section file, skipped: {name}")
    files = [present[k] for k in sorted(present)]

    drafts: list[str] = []
    texts: list[tuple[Path, str]] = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if first.lstrip("*_>#[ ").startswith(DRAFT_MARKER):
            drafts.append(p.name)
        texts.append((p, text))

    date = _dt.date.today().strftime("%d %B %Y")
    story: list = title_page(st, doc.width, drafts, date)

    toc = TableOfContents()
    toc.levelStyles = [st["toc0"], st["toc1"]]
    toc.dotsMinLevel = 0
    story.append(Paragraph("Table of Contents", st["toc_title"]))
    story.append(Spacer(1, 4 * mm))
    story.append(toc)

    builder = Builder(st, doc.width, doc.height, src)
    for p, text in texts:
        m = re.match(r"^sec(\d+)", p.name, re.I)
        number = int(m.group(1)) if m else None
        story.append(PageBreak())
        story.extend(builder.parse(text, number))
    if not texts:
        story.append(PageBreak())
        story.append(Paragraph("No section files were found.", st["body"]))

    # first page uses the 'first' template (no footer); NextPageTemplate switches after title page.
    doc.multiBuild(story)
    print(f"Wrote {out}  (sections: {len(texts)}, draft sections: {len(drafts)}, missing figures: "
          f"{len(builder.missing_figures)})")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the PathSnap technical report PDF from Markdown sections.")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="directory with section .md files (default docs/report)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output PDF path")
    args = ap.parse_args(argv)
    build(Path(args.src), Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
