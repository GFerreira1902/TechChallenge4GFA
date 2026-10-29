"""Estrutura de alertas e consolidacao multimodal."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

from src.guardrails.safety_rules import enforce_human_validation

SEVERIDADES = ("baixa", "media", "alta", "critica")
_PESO = {"baixa": 1, "media": 2, "alta": 4, "critica": 8}


@dataclass
class Alerta:
    paciente: str
    modalidade: str  # vitais | prescricao | movimento | audio | video
    severidade: str
    evidencia: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if self.severidade not in SEVERIDADES:
            raise ValueError(f"Severidade invalida: {self.severidade}")

    def to_dict(self) -> dict:
        return asdict(self)

    def mensagem(self) -> str:
        texto = (
            f"[ALERTA {self.severidade.upper()}] {self.paciente} ({self.modalidade}): "
            f"{self.evidencia}"
        )
        texto, _ = enforce_human_validation(texto)
        return texto


def score_risco(alertas: list[Alerta]) -> float:
    """Score consolidado 0-100; sobe com severidade e diversidade de modalidades."""
    if not alertas:
        return 0.0
    base = sum(_PESO[a.severidade] for a in alertas)
    bonus = 1 + 0.25 * (len({a.modalidade for a in alertas}) - 1)
    return round(min(100.0, base * bonus * 5), 1)


def nivel_risco(score: float) -> str:
    if score >= 60:
        return "critico"
    if score >= 30:
        return "alto"
    if score > 0:
        return "moderado"
    return "normal"
