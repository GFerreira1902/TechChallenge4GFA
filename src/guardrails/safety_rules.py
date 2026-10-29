"""Guardrails de seguranca: garante que o assistente nunca prescreva diretamente
sem validacao humana, conforme exigido no enunciado da Fase 3.
"""
from __future__ import annotations

import re

# Padroes que indicam uma prescricao direta e imperativa (sem ressalva de validacao humana).
DIRECT_PRESCRIPTION_PATTERNS = [
    r"\btome\b",
    r"\badministre\b(?!.*valida)",
    r"\bprescrevo\b",
    r"\baplique\b.*\d+\s*mg",
    r"\binicie\b.*(\d+\s*(mg|ml)|\bdose\b)",
]

HUMAN_VALIDATION_DISCLAIMER = (
    "\n\n[AVISO] Este é um alerta automático de monitoramento e não substitui a "
    "avaliação clínica de um profissional de saúde responsável. Nenhuma conduta "
    "deve ser executada sem validação humana."
)


def contains_direct_prescription(text: str) -> bool:
    """Detecta linguagem de prescricao direta/imperativa na resposta do assistente."""
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in DIRECT_PRESCRIPTION_PATTERNS)


def enforce_human_validation(text: str) -> tuple[str, list[str]]:
    """Garante que toda resposta clinica carregue o aviso de validacao humana.

    Retorna o texto (com o disclaimer anexado, se ainda nao houver um) e a lista
    de avisos de seguranca levantados (ex.: linguagem de prescricao direta detectada).
    """
    warnings: list[str] = []

    if contains_direct_prescription(text):
        warnings.append(
            "Linguagem de prescrição direta detectada na resposta - reforçando disclaimer."
        )

    if "validação humana" not in text.lower() and "validacao humana" not in text.lower():
        text = text + HUMAN_VALIDATION_DISCLAIMER

    return text, warnings
