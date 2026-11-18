"""Build the editable Word (.docx) version of the technical report from docs/report/*.md.

    python scripts/build_report_docx.py [--src docs/report] [--out output/pdf/PathSnap_technical_report_final.docx]

Plain academic style: Times New Roman, black text, no coloured headings or table fills. Handles the Markdown subset used
by the report sections: # / ## / ### headings, paragraphs, bullet and numbered lists, pipe tables, standalone images,
fenced code blocks, **bold**, *italic* and `code`.
"""
import argparse
import re
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FONT, MONO = "Times New Roman", "Courier New"
BLACK = RGBColor(0, 0, 0)
INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\n]+?\*)")


def set_font(run, size=None, bold=None, italic=None, mono=False):
    run.font.name = MONO if mono else FONT
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:eastAsia"), MONO if mono else FONT)
    rpr.rFonts.set(qn("w:cs"), MONO if mono else FONT)
    run.font.color.rgb = BLACK
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_inline(par, text, size, bold=False, italic=False):
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            add_inline(par, tok[2:-2], size, True, italic)
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            set_font(par.add_run(tok[1:-1]), size - 1, bold, italic, mono=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            add_inline(par, tok[1:-1], size, bold, True)
        else:
            set_font(par.add_run(tok), size, bold, italic)


def fmt(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=6, left=None, first=None, keep=False, line=1.15):
    pf = par.paragraph_format
    par.alignment = align
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), line
    if left is not None:
        pf.left_indent = Inches(left)
    if first is not None:
        pf.first_line_indent = Inches(first)
    if keep:
        pf.keep_with_next = True
    return par


def add_field(par, instr, size=10):
    for kind, txt in (("begin", None), (None, instr), ("separate", None), (None, "1"), ("end", None)):
        r = par.add_run()
        set_font(r, size)
        if kind:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), kind); r._element.append(fc)
        elif txt == instr:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; r._element.append(it)
        else:
            t = OxmlElement("w:t"); t.text = txt; r._element.append(t)


def set_cell_borders(table):
    tbl = table._tbl
    pr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        b.append(e)
    pr.append(b)


def add_table(doc, rows):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    size = 9.5 if ncol <= 4 else 8.5 if ncol <= 6 else 7.5 if ncol <= 8 else 6.5
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_cell_borders(t)
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 0, 0, line=1.0)
            add_inline(p, text.strip(), size, bold=(i == 0))
    # repeat header row on page breaks
    trpr = t.rows[0]._tr.get_or_add_trPr()
    h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trpr.append(h)
    sp = doc.add_paragraph(); fmt(sp, after=4); sp.paragraph_format.space_before = Pt(0)
    return t


def add_image(doc, path, caption, max_w=6.3, max_h=6.2):
    from PIL import Image
    w, h = Image.open(path).size
    scale = min(max_w / (w / 96), max_h / (h / 96), 1e9)
    wi = min(max_w, max_h * w / h)
    p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 4, 2, keep=True)
    p.add_run().add_picture(str(path), width=Inches(wi))
    if caption.strip():
        c = doc.add_paragraph(); fmt(c, WD_ALIGN_PARAGRAPH.CENTER, 0, 8, line=1.0)
        add_inline(c, caption.strip(), 9.5, italic=True)


def add_code(doc, lines):
    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 2, 8, left=0.1, line=1.0)
    ppr = p._p.get_or_add_pPr()
    bd = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "3"); e.set(qn("w:color"), "000000")
        bd.append(e)
    ppr.append(bd)
    for i, line in enumerate(lines):
        r = p.add_run(line)
        set_font(r, 8, mono=True)
        if i < len(lines) - 1:
            r.add_break(WD_BREAK.LINE)


def parse_blocks(text):
    """Yield ('h',level,text) ('p',text) ('ul',text) ('ol',n,text) ('table',rows) ('img',alt,path) ('code',lines)."""
    lines = text.split("\n")
    i, para = 0, []

    def flush():
        nonlocal para
        if para:
            yield ("p", " ".join(s.strip() for s in para))
            para = []

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            yield from flush()
            i += 1; code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            yield ("code", code); i += 1; continue
        if not ln.strip():
            yield from flush(); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            yield from flush(); yield ("h", len(m.group(1)), m.group(2).strip()); i += 1; continue
        m = re.match(r"^!\[(.*)\]\((.*)\)\s*$", ln.strip())
        if m:
            yield from flush(); yield ("img", m.group(1), m.group(2)); i += 1; continue
        if ln.lstrip().startswith("|"):
            yield from flush(); rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            yield ("table", rows); continue
        m = re.match(r"^\s*[-*]\s+(.*)", ln)
        if m and not ln.startswith("**"):
            yield from flush(); yield ("ul", m.group(1)); i += 1; continue
        m = re.match(r"^\s*(\d+)\.\s+(.*)", ln)
        if m:
            yield from flush(); yield ("ol", m.group(1), m.group(2)); i += 1; continue
        para.append(ln); i += 1
    yield from flush()


def build(src: Path, out: Path):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = FONT; st.font.size = Pt(11); st.font.color.rgb = BLACK
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for a in ("left_margin", "right_margin"):
        setattr(sec, a, Inches(1.0))
    sec.top_margin, sec.bottom_margin = Inches(1.0), Inches(0.9)

    team_file = src / "team.txt"
    team = team_file.read_text(encoding="utf-8").strip().split("\n") if team_file.exists() else []

    # ---- cover page
    for _ in range(5):
        fmt(doc.add_paragraph(), after=12)
    p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 0, 6); set_font(p.add_run("PathSnap"), 30, bold=True)
    for txt, size in (("Distributed Map-Matching and Route Reconstruction", 15), ("Technical Report", 15), ("", 8),
                      ("Course code: UE24CS352A", 13), ("Machine Learning", 13),
                      ("PES University, Department of Computer Science and Engineering, RR Campus", 13), ("", 8)):
        q = doc.add_paragraph(); fmt(q, WD_ALIGN_PARAGRAPH.CENTER, 0, 4)
        if txt:
            set_font(q.add_run(txt), size)
    for k, line in enumerate(team):
        q = doc.add_paragraph(); fmt(q, WD_ALIGN_PARAGRAPH.CENTER, 0, 3, line=1.0)
        set_font(q.add_run(line), 12, bold=(k == 0))
    q = doc.add_paragraph(); fmt(q, WD_ALIGN_PARAGRAPH.CENTER, 14, 0)
    set_font(q.add_run(f"Date: {date.today().day} {date.today():%B %Y}"), 12)

    # ---- contents (field; filled by Word on update)
    pb = doc.add_paragraph(); pb.add_run().add_break(WD_BREAK.PAGE)
    h = doc.add_paragraph(); fmt(h, WD_ALIGN_PARAGRAPH.LEFT, 0, 10); set_font(h.add_run("Table of Contents"), 16, bold=True)
    tp = doc.add_paragraph(); fmt(tp, WD_ALIGN_PARAGRAPH.LEFT, 0, 0)
    add_field(tp, 'TOC \\o "1-2" \\h \\z \\u', 11)

    # no header or footer on any page (the section header/footer stay empty)

    files = sorted(p for p in src.glob("sec*.md"))
    for f in files:
        first_h1 = True
        for blk in parse_blocks(f.read_text(encoding="utf-8")):
            kind = blk[0]
            if kind == "h":
                _, level, text = blk
                if level == 1:
                    doc.add_page_break()
                par = doc.add_heading(level=min(level, 3))
                par.style.font.color.rgb = BLACK
                fmt(par, WD_ALIGN_PARAGRAPH.LEFT, 12 if level > 1 else 0, 6, keep=True, line=1.0)
                run = par.add_run(text)
                set_font(run, {1: 17, 2: 13.5, 3: 12, 4: 11}[min(level, 4)], bold=True, italic=(level == 4))
            elif kind == "p":
                fmt(add_inline_par(doc, blk[1]), keep=False)
            elif kind == "ul":
                p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 3, left=0.35, first=-0.2)
                add_inline(p, "•  " + blk[1], 11)
            elif kind == "ol":
                p = doc.add_paragraph(); fmt(p, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 3, left=0.35, first=-0.25)
                add_inline(p, f"{blk[1]}.  " + blk[2], 11)
            elif kind == "table":
                add_table(doc, blk[1])
            elif kind == "img":
                path = (ROOT / blk[2]) if not Path(blk[2]).is_absolute() else Path(blk[2])
                if not path.exists():
                    path = src / blk[2]
                add_image(doc, path, blk[1])
            elif kind == "code":
                add_code(doc, blk[1])
    # black headings: reset built-in heading styles
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        s = doc.styles[name]; s.font.color.rgb = BLACK; s.font.name = FONT; s.font.bold = True
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return out


def add_inline_par(doc, text):
    p = doc.add_paragraph()
    add_inline(p, text, 11)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "docs" / "report"))
    ap.add_argument("--out", default=str(ROOT / "output" / "pdf" / "PathSnap_technical_report_final.docx"))
    a = ap.parse_args()
    print("wrote", build(Path(a.src), Path(a.out)))
