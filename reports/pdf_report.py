import os
from datetime import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from api.services.storage_service import get_reports_dir


def _draw_wrapped_text(c, x, y, text, max_chars=95, line_height=12):
    words = text.split(" ")
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > max_chars:
            c.drawString(x, y, line)
            y -= line_height
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        c.drawString(x, y, line)
        y -= line_height
    return y


def generate_pdf_report(job, result: Dict[str, Any]) -> str:
    reports_dir = get_reports_dir()
    os.makedirs(reports_dir, exist_ok=True)

    pdf_path = f"{reports_dir}/report_{job.id}.pdf"

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    y = height - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Women Health AI - Relatório do MVP")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Job ID: {job.id}")
    y -= 16
    c.drawString(50, y, f"Tipo: {job.job_type}")
    y -= 16
    c.drawString(50, y, f"Arquivo: {job.input_path}")
    y -= 16
    c.drawString(50, y, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Aviso")
    y -= 16

    c.setFont("Helvetica", 10)
    y = _draw_wrapped_text(
        c,
        50,
        y,
        "Este relatório é um sistema de apoio e não substitui avaliação profissional. "
        "Não fornece diagnóstico. Alertas devem ser revisados por um profissional.",
    )
    y -= 10

    # Scores
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Scores (MVP)")
    y -= 16
    c.setFont("Helvetica", 10)

    scores = result.get("scores", {}) or {}
    if not scores:
        c.drawString(50, y, "Nenhum score calculado.")
        y -= 14
    else:
        for k, v in scores.items():
            c.drawString(50, y, f"{k}: {v}")
            y -= 14

    y -= 10

    # Métricas
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Métricas")
    y -= 16
    c.setFont("Helvetica", 10)

    metrics = result.get("metrics", {}) or {}
    if not metrics:
        c.drawString(50, y, "Nenhuma métrica disponível.")
        y -= 14
    else:
        for k, v in metrics.items():
            c.drawString(50, y, f"{k}: {v}")
            y -= 14
            if y < 80:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 10)

    y -= 10

    # Eventos
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Eventos")
    y -= 16
    c.setFont("Helvetica", 10)

    events = result.get("events", []) or []
    if not events:
        c.drawString(50, y, "Nenhum evento detectado.")
        y -= 14
    else:
        for e in events:
            line = f"- {e.get('type')} | score={e.get('score')} | {e.get('description', '')}"
            y = _draw_wrapped_text(c, 50, y, line, max_chars=95, line_height=12)
            if y < 80:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 10)

    c.showPage()
    c.save()

    return pdf_path