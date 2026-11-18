"""Backward-compatible entry point for the consolidated report builder.

Use ``python scripts/build_report_pdf.py`` as documented in the README.
This legacy command delegates to the same section-based builder and cannot
create a competing short-form report.
"""
from build_report_pdf import main


if __name__ == "__main__":
    raise SystemExit(main())
