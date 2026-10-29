"""Geracao automatica de laudos clinicos (documento formatado), a partir da
sugestao de conduta produzida pelo RAG. Fecha o requisito do enunciado sobre
"modelos de laudos, receitas e procedimentos internos" como tipo de saida do
assistente (o modelo de referencia esta em data/raw/synthetic_laudo_template.md).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any, Optional
from xml.sax.saxutils import escape

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.guardrails.safety_rules import enforce_human_validation

REPORTS_DIR = Path("outputs/reports")


def _format_report_text(
    paciente: dict,
    timestamp: datetime,
    sugestao_tratamento: str,
    fontes: list[dict],
    alerta_emitido: Optional[str],
    llm_text: str,
) -> str:
    """Monta o documento com campos objetivos e o texto do modelo como complemento."""
    exames = paciente.get("exames_pendentes", [])
    exames_texto = "; ".join(exames) if exames else "Nenhum exame pendente."
    protocolos = ", ".join(
        f"{fonte['protocol_id']} - {fonte.get('titulo', 'Protocolo sem título')}"
        for fonte in fontes
    ) or "Nenhum protocolo identificado."
    condutas = [
        f"Sugestão retornada pelo pipeline RAG: {_remove_validation_notice(sugestao_tratamento)}",
        f"Revisar os exames pendentes: {exames_texto}",
        f"Consultar os protocolos internos: {protocolos}",
    ]
    if alerta_emitido:
        condutas.append("Comunicar e acompanhar o alerta crítico registrado para a equipe responsável.")
    condutas.append("Validar qualquer conduta com o profissional de saúde responsável.")

    complemento = _extract_llm_complement(llm_text)
    if complemento:
        condutas.append(f"Complemento gerado pelo assistente: {complemento}")

    return "\n".join(
        [
            "LAUDO CLÍNICO",
            f"Paciente: {paciente['paciente_id']}",
            f"Data: {timestamp:%d/%m/%Y}",
            f"Hora: {timestamp:%H:%M:%S}",
            "",
            "1. Diagnóstico principal:",
            paciente["diagnostico_principal"],
            "",
            "2. Exames pendentes ou realizados:",
            exames_texto,
            "",
            "3. Conduta sugerida:",
            "\n".join(f"- {conduta}" for conduta in condutas),
            "",
            "4. Protocolos internos consultados (fonte):",
            protocolos,
            "",
            "5. Alertas ativos:",
            alerta_emitido or "Nenhum alerta ativo.",
            "",
            "[AVISO] Documento gerado automaticamente. Não substitui a avaliação clínica "
            "de um profissional de saúde responsável. Nenhuma conduta deve ser executada "
            "sem validação humana.",
        ]
    )


def _extract_llm_complement(llm_text: str) -> str:
    """Evita inserir um segundo laudo ou uma frase genérica no documento final."""
    text = llm_text.strip()
    normalized = re.sub(r"\s+", " ", text).lower()
    generic_prefixes = (
        "com base na consulta do profissional de saúde, as seguintes condutas são recomendadas",
        "com base na consulta do profissional de saude, as seguintes condutas sao recomendadas",
    )
    if not text or "laudo clínico" in normalized or "laudo clinico" in normalized:
        return ""
    if normalized.startswith(generic_prefixes):
        return ""
    return text


def _remove_validation_notice(text: str) -> str:
    """Mantém o aviso de validação apenas no rodapé do laudo."""
    cleaned = re.split(r"\[AVISO\]|⚠️", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return cleaned.strip().rstrip("-").strip()


def _register_fonts() -> tuple[str, str]:
    regular_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular_path.exists() and bold_path.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(regular_path)))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(bold_path)))
        return "Arial", "Arial-Bold"
    return "Helvetica", "Helvetica-Bold"


def salvar_laudo_pdf(laudo: str, paciente_id: str, timestamp: datetime) -> str:
    """Salva o texto do laudo como PDF e retorna o caminho gerado."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", paciente_id)
    pdf_path = REPORTS_DIR / f"laudo_{safe_id}_{timestamp:%Y%m%d_%H%M%S}.pdf"
    regular_font, bold_font = _register_fonts()

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName=bold_font,
        alignment=TA_CENTER, spaceAfter=18,
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontName=regular_font,
        leading=15, spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "ReportHeading", parent=body_style, fontName=bold_font,
        fontSize=14, leading=18, spaceBefore=12, spaceAfter=7,
    )

    story = []
    for raw_line in laudo.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.upper() == "LAUDO CLÍNICO":
            story.append(Paragraph(line, title_style))
        elif re.match(r"^[1-5]\.\s", line):
            story.append(Paragraph(escape(line), heading_style))
        else:
            story.append(Paragraph(escape(line), body_style))

    document = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"Laudo clínico {paciente_id}",
        author="Assistente Virtual Médico - Tech Challenge Fase 3",
    )
    document.build(story)
    return str(pdf_path)

LAUDO_SYSTEM_PROMPT = (
    "Você é um assistente que redige laudos clínicos estruturados para um "
    "hospital. Use exatamente este formato, preenchendo cada seção de forma "
    "objetiva e curta, em português:\n\n"
    "LAUDO CLÍNICO\n"
    "Paciente: <id>\n"
    "Data/hora: <data>\n\n"
    "1. Diagnóstico principal\n<texto>\n\n"
    "2. Exames pendentes ou realizados\n<texto>\n\n"
    "3. Conduta sugerida\n<texto>\n\n"
    "4. Protocolos internos consultados (fonte)\n<texto>\n\n"
    "5. Alertas ativos\n<texto>\n\n"
    "Não invente informações que não estejam no contexto fornecido. Você "
    "nunca prescreve tratamentos diretamente sem ressalva de validação humana."
)


def gerar_laudo(
    llm_client: Any,
    paciente: dict,
    sugestao_tratamento: str,
    fontes: list[dict],
    alerta_emitido: Optional[str] = None,
) -> dict:
    """Gera um laudo clinico formatado com base no atendimento processado.

    Retorna um dict com o texto do laudo e os avisos de seguranca levantados.
    """
    protocolos = ", ".join(f["protocol_id"] for f in fontes) or "nenhum"
    exames = ", ".join(paciente.get("exames_pendentes", [])) or "nenhum"

    timestamp = datetime.now()
    user_message = (
        f"Paciente: {paciente['paciente_id']}\n"
        f"Data/hora: {timestamp.isoformat()}\n"
        f"Diagnóstico principal: {paciente['diagnostico_principal']}\n"
        f"Exames pendentes: {exames}\n"
        f"Conduta sugerida pelo assistente: {sugestao_tratamento}\n"
        f"Protocolos consultados: {protocolos}\n"
        f"Alerta crítico: {alerta_emitido or 'nenhum'}\n\n"
        "Redija o laudo clínico completo no formato especificado."
    )

    laudo_bruto = llm_client.ask(
        system_prompt=LAUDO_SYSTEM_PROMPT,
        user_message=user_message,
        metadata={"paciente_id": paciente["paciente_id"], "tipo": "laudo_clinico"},
    )

    laudo_validado, avisos = enforce_human_validation(laudo_bruto)
    laudo_formatado = _format_report_text(
        paciente=paciente,
        timestamp=timestamp,
        sugestao_tratamento=sugestao_tratamento,
        fontes=fontes,
        alerta_emitido=alerta_emitido,
        llm_text=laudo_validado,
    )
    pdf_path = salvar_laudo_pdf(laudo_formatado, paciente["paciente_id"], timestamp)
    return {
        "laudo": laudo_formatado,
        "avisos_seguranca": avisos,
        "pdf_path": pdf_path,
    }
