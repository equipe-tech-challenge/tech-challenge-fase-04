import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

from api.services.storage_service import get_reports_dir


COLOR_DARK = colors.HexColor("#111827")
COLOR_MUTED = colors.HexColor("#6b7280")
COLOR_BORDER = colors.HexColor("#d1d5db")
COLOR_BG = colors.HexColor("#f9fafb")
COLOR_HEADER = colors.HexColor("#0f172a")  

_styles = getSampleStyleSheet()
STYLE_CELL = _styles["BodyText"]
STYLE_CELL.fontName = "Helvetica"
STYLE_CELL.fontSize = 9
STYLE_CELL.leading = 11


def _safe_str(v: Any) -> str:
    if v is None:
        return "N/A"
    return str(v)


def _spacer(h=10):
    return Spacer(1, h)


def _section_title(text: str, styles):
    return KeepTogether([
        Paragraph(text, styles["H2"]),
        Spacer(1, 6),
    ])


def _card(elements: List, padding=10):
    """
    "Card" usando Table com fundo e borda.
    """
    t = Table([[elements]], colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, COLOR_BORDER),
        ("INNERPADDING", (0, 0), (-1, -1), padding),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _kv_table(data: Dict[str, Any], col1="Campo", col2="Valor"):
    rows = [[col1, col2]]

    for k, v in data.items():
        key = Paragraph(_safe_str(k), STYLE_CELL)
        value = Paragraph(_safe_str(v), STYLE_CELL)
        rows.append([key, value])

    t = Table(rows, colWidths=[6.5 * cm, 9.8 * cm])

    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),

        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, COLOR_BG]),
        ("GRID", (0, 0), (-1, -1), 0.25, COLOR_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),

        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    return t


def _events_table(events: List[Dict[str, Any]]):
    rows = [["Tipo", "Score", "Descrição"]]

    for e in events:
        rows.append([
            Paragraph(_safe_str(e.get("type", "")), STYLE_CELL),
            Paragraph(_safe_str(e.get("score", "")), STYLE_CELL),
            Paragraph(_safe_str(e.get("description", "")), STYLE_CELL)
        ])

    t = Table(rows, colWidths=[5.2 * cm, 2.0 * cm, 9 * cm])

    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),

        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),

        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, COLOR_BG]),
        ("GRID", (0, 0), (-1, -1), 0.25, COLOR_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),

        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    return t


# ==========================================================
# Seções
# ==========================================================
def _build_cover(story, styles, job):
    story.append(Spacer(1, 70))
    story.append(Paragraph("Women Health AI", styles["CoverTitle"]))
    story.append(Spacer(1, 22))

    story.append(Paragraph(
        "Sistema de apoio para análise automatizada de vídeo e áudio clínico "
        "(cirurgias, consultas e fisioterapia).",
        styles["CoverBody"]
    ))
    story.append(Spacer(1, 18))

    info = {
        "Job ID": job.id,
        "Tipo": job.job_type,
        "Arquivo": job.input_path,
        "Data": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    }

    story.append(_card([
        Paragraph("Identificação", styles["H3"]),
        Spacer(1, 6),
        _kv_table(info, "Item", "Valor"),
    ]))

    story.append(Spacer(1, 18))
    story.append(_card([
        Paragraph("Aviso", styles["H3"]),
        Spacer(1, 6),
        Paragraph(
            "Este relatório é um sistema de apoio e <b>não substitui avaliação profissional</b>. "
            "Não fornece diagnóstico. Alertas devem ser revisados por um profissional.",
            styles["Body"]
        ),
    ]))

    story.append(PageBreak())


def _build_executive_summary(story, styles, result: Dict[str, Any]):
    story.append(Paragraph("Resumo Executivo", styles["H2"]))
    story.append(Spacer(1, 8))

    scores = result.get("scores", {}) or {}
    metrics = result.get("metrics", {}) or {}
    events = result.get("events", []) or []

    summary_lines = []
    summary_lines.append(f"- Scores gerais encontrados: <b>{len(scores)}</b>")
    summary_lines.append(f"- Métricas gerais extraídas: <b>{len(metrics)}</b>")
    summary_lines.append(f"- Eventos gerais detectados: <b>{len(events)}</b>")

    story.append(_card([
        Paragraph(
            "<br/>".join(summary_lines),
            styles["Body"]
        )
    ]))

    story.append(Spacer(1, 14))


def _build_general_pipeline(story, styles, result: Dict[str, Any]):
    story.append(Paragraph("Resultados do Pipeline (Geral)", styles["H2"]))
    story.append(Spacer(1, 8))

    scores = result.get("scores", {}) or {}
    metrics = result.get("metrics", {}) or {}
    events = result.get("events", []) or []

    story.append(_card([
        Paragraph("Scores", styles["H3"]),
        Spacer(1, 6),
        _kv_table(scores, "Score", "Valor") if scores else Paragraph("Nenhum score calculado.", styles["Body"]),
    ]))
    story.append(Spacer(1, 10))

    story.append(_card([
        Paragraph("Métricas", styles["H3"]),
        Spacer(1, 6),
        _kv_table(metrics, "Métrica", "Valor") if metrics else Paragraph("Nenhuma métrica disponível.", styles["Body"]),
    ]))
    story.append(Spacer(1, 10))

    story.append(_card([
        Paragraph("Eventos", styles["H3"]),
        Spacer(1, 6),
        _events_table(events[:25]) if events else Paragraph("Nenhum evento detectado.", styles["Body"]),
    ]))
    story.append(Spacer(1, 14))


# ==========================================================
# Rodapé
# ==========================================================
def _draw_footer(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(COLOR_MUTED)

    canvas_obj.drawString(2 * cm, 1.2 * cm, "Women Health AI — Relatório gerado automaticamente")
    canvas_obj.drawRightString(19.5 * cm, 1.2 * cm, f"Página {doc.page}")

    canvas_obj.restoreState()


# ==========================================================
# Main
# ==========================================================
def generate_pdf_report(job, result: Dict[str, Any]) -> str:
    reports_dir = get_reports_dir()
    os.makedirs(reports_dir, exist_ok=True)

    pdf_path = os.path.join(reports_dir, f"Report_WomenHealth_{job.id}.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Women Health AI - Relatório",
        author="Women Health AI",
    )

    base = getSampleStyleSheet()

    styles = {
        "CoverTitle": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=COLOR_DARK,
            spaceAfter=2,
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=13,
            textColor=COLOR_MUTED,
            spaceAfter=14,
        ),
        "CoverBody": ParagraphStyle(
            "CoverBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=COLOR_DARK,
        ),

        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=COLOR_DARK,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "H3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#374151"),
            spaceBefore=0,
            spaceAfter=2,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=COLOR_DARK,
        ),
    }

    story = []

    _build_cover(story, styles, job)

    _build_executive_summary(story, styles, result)

    _build_general_pipeline(story, styles, result)

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)

    return pdf_path
