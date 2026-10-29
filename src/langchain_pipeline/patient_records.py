"""Acesso aos prontuarios sinteticos (dados estruturados de pacientes fictícios).

Simula a "consulta em base de dados estruturadas" exigida no enunciado da Fase 3
(prontuarios e registros), usada para contextualizar as respostas do assistente.
"""
from __future__ import annotations

import json

PATIENTS_PATH = "data/raw/synthetic_patients.json"


def load_patients(path: str = PATIENTS_PATH) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["paciente_id"]: p for p in data["pacientes"]}


def get_patient(paciente_id: str, path: str = PATIENTS_PATH) -> dict | None:
    return load_patients(path).get(paciente_id)


def format_patient_context(patient: dict) -> str:
    """Resumo textual do prontuario, para injetar no prompt do assistente."""
    exames = ", ".join(patient["exames_pendentes"]) or "nenhum"
    alertas = "; ".join(patient["alertas"]) or "nenhum"
    medicacoes = ", ".join(patient["medicacoes_atuais"]) or "nenhuma"

    return (
        f"Paciente {patient['paciente_id']} ({patient['idade']} anos, {patient['sexo']}).\n"
        f"Diagnostico principal: {patient['diagnostico_principal']}.\n"
        f"Historico: {patient['historico']}\n"
        f"Medicacoes atuais: {medicacoes}.\n"
        f"Exames pendentes: {exames}.\n"
        f"Alertas ativos: {alertas}."
    )
