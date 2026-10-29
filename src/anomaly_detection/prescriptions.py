"""Deteccao de alteracoes inesperadas na evolucao de prescricoes."""
from __future__ import annotations

import re
from datetime import date

from src.anomaly_detection.alerts import Alerta

_DOSE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|ml|ui)", re.I)
_FATOR = {"g": 1000.0, "mg": 1.0, "mcg": 0.001, "ml": 1.0, "ui": 1.0}


def dose_em_mg(dose: str) -> float | None:
    m = _DOSE_RE.search(dose or "")
    if not m:
        return None
    return float(m.group(1).replace(",", ".")) * _FATOR[m.group(2).lower()]


def analisar_prescricoes(
    paciente_id: str,
    historico: list[dict],
    limite_variacao: float = 2.0,
    medicacoes_atuais: list[str] | None = None,
) -> list[Alerta]:
    """Cada item: {data, medicamento, dose}. Sinaliza saltos de dose, suspensoes
    abruptas e medicamentos novos fora do prontuario atual."""
    alertas: list[Alerta] = []
    por_med: dict[str, list[dict]] = {}
    for item in sorted(historico, key=lambda x: x["data"]):
        por_med.setdefault(item["medicamento"].lower(), []).append(item)

    for med, itens in por_med.items():
        for ant, atual in zip(itens, itens[1:]):
            d0, d1 = dose_em_mg(ant["dose"]), dose_em_mg(atual["dose"])
            if not d0 or d1 is None:
                continue
            if d1 == 0:
                alertas.append(Alerta(paciente_id, "prescricao", "alta",
                                      f"{med}: suspensao abrupta em {atual['data']}"))
                continue
            razao = d1 / d0
            if razao >= limite_variacao or razao <= 1 / limite_variacao:
                sev = "critica" if razao >= 2 * limite_variacao else "alta"
                alertas.append(Alerta(
                    paciente_id, "prescricao", sev,
                    f"{med}: dose {ant['dose']} -> {atual['dose']} em {atual['data']} "
                    f"(x{razao:.1f}) - alteracao inesperada",
                ))

    if medicacoes_atuais is not None:
        atuais = " ".join(medicacoes_atuais).lower()
        ult = max((i["data"] for i in historico), default=None)
        for med, itens in por_med.items():
            if itens[-1]["data"] == ult and med.split()[0] not in atuais:
                alertas.append(Alerta(paciente_id, "prescricao", "media",
                                      f"{med} prescrito recentemente e ausente da lista de medicacoes atuais"))
    return alertas
