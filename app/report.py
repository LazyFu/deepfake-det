"""用 Jinja2 + WeasyPrint 生成 PDF 审计报告。"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

TEMPLATE_DIR = Path("app/templates")


def render_report(ctx: dict, out_pdf: Path) -> Path:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    html = env.get_template("report.html").render(**ctx)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(Path.cwd())).write_pdf(str(out_pdf))
    return out_pdf
