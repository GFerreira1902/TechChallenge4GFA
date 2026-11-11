"""Relatorio automatico de desvios em video (PDF com timeline de eventos e frames anotados)."""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

from src.guardrails.safety_rules import HUMAN_VALIDATION_DISCLAIMER
from src.video_analysis.pose_analyzer import EventoVideo, Postura

REPORTS_DIR = Path("outputs/reports")
FIGURES_DIR = Path("outputs/figures")

# pares de keypoints COCO para desenhar o esqueleto
ESQUELETO = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
             (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def _fontes() -> tuple[str, str]:
    reg, bold = Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")
    if reg.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(reg)))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(bold)))
        return "Arial", "Arial-Bold"
    return "Helvetica", "Helvetica-Bold"


def anotar_frame(imagem: np.ndarray, postura: Optional[Postura], evento: Optional[EventoVideo],
                 deteccoes: Optional[list] = None) -> np.ndarray:
    img = imagem.copy()
    for d in deteccoes or []:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cor = (0, 200, 255) if d.classe == "person" else (180, 180, 180)
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(img, f"{d.classe} {d.confianca:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)
    if postura is not None and postura.keypoints is not None:
        kp = postura.keypoints
        for a, b in ESQUELETO:
            if kp[a][0] > 0 and kp[b][0] > 0:
                cv2.line(img, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (0, 255, 0), 2)
        for p in kp:
            if p[0] > 0:
                cv2.circle(img, tuple(p.astype(int)), 3, (0, 0, 255), -1)
    if evento is not None:
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 180), -1)
        cv2.putText(img, f"{evento.tipo.upper()} t={evento.tempo_s:.1f}s", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def salvar_figura(img: np.ndarray, nome: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{re.sub(r'[^A-Za-z0-9_-]', '_', nome)}.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(path)


def gerar_relatorio_pdf(
    paciente_id: str,
    arquivo_video: str,
    info: dict,
    eventos: list[EventoVideo],
    figuras: list[str],
    resumo_objetos: dict[str, int],
    timestamp: Optional[datetime] = None,
) -> str:
    timestamp = timestamp or datetime.now()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", paciente_id)
    pdf = REPORTS_DIR / f"video_{safe}_{timestamp:%Y%m%d_%H%M%S}.pdf"
    reg, bold = _fontes()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["BodyText"], fontName=reg, fontSize=10, leading=14)
    h = ParagraphStyle("h", parent=body, fontName=bold, fontSize=12, spaceBefore=8, spaceAfter=4)
    title = ParagraphStyle("t", parent=styles["Title"], fontName=bold, fontSize=15)

    el = [Paragraph("RELATÓRIO AUTOMÁTICO DE ANÁLISE DE VÍDEO", title), Spacer(1, 0.3 * cm)]
    el.append(Paragraph(
        f"<b>Paciente:</b> {escape(paciente_id)} &nbsp;&nbsp; <b>Arquivo:</b> {escape(os.path.basename(arquivo_video))}"
        f"<br/><b>Gerado em:</b> {timestamp:%d/%m/%Y %H:%M:%S} &nbsp;&nbsp; <b>Duração:</b> {info.get('duracao_s', 0):.1f}s "
        f"({info.get('frames', 0)} frames @ {info.get('fps', 0):.0f} fps, {info.get('largura')}x{info.get('altura')})", body))
    el.append(Paragraph("1. Modelos aplicados", h))
    el.append(Paragraph("Detecção de objetos: YOLOv8n (COCO). Estimativa de postura: YOLOv8n-pose "
                        "(17 keypoints, equivalente funcional ao OpenPose). Regras temporais sobre altura do "
                        "corpo, inclinação do tronco, deslocamento e áreas críticas.", body))
    el.append(Paragraph("2. Objetos detectados (frames com ocorrência)", h))
    el.append(Paragraph(", ".join(f"{k}: {v}" for k, v in sorted(resumo_objetos.items(), key=lambda x: -x[1])) or "nenhum", body))

    el.append(Paragraph("3. Linha do tempo de eventos", h))
    if eventos:
        dados = [["Tempo", "Evento", "Severidade", "Evidência"]] + [
            [f"{e.tempo_s:.1f}s", e.tipo, e.severidade, Paragraph(escape(e.evidencia), body)] for e in eventos]
        tb = Table(dados, colWidths=[1.6 * cm, 3.2 * cm, 2.4 * cm, 9.8 * cm])
        tb.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), bold), ("FONTNAME", (0, 1), (-1, -1), reg), ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE3EA")), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        el.append(tb)
    else:
        el.append(Paragraph("Nenhum desvio detectado. Padrão de movimentação dentro do esperado.", body))

    if figuras:
        el.append(Paragraph("4. Frames anotados", h))
        for f in figuras[:4]:
            el.append(Image(f, width=12 * cm, height=9 * cm, kind="proportional"))
            el.append(Spacer(1, 0.2 * cm))

    el.append(Spacer(1, 0.4 * cm))
    el.append(Paragraph(escape(HUMAN_VALIDATION_DISCLAIMER.strip()), body))
    SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm).build(el)
    return str(pdf)
