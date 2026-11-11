"""Helpers puros do dashboard (sem Streamlit) - testaveis isoladamente."""
from __future__ import annotations

import glob
import json
import os
from typing import Iterator

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.anomaly_detection.alerts import Alerta

CORES_SEV = {"baixa": "#6c757d", "media": "#f0ad4e", "alta": "#fd7e14", "critica": "#dc3545"}
ICONE_MOD = {"vitais": "[VITAIS]", "prescricao": "[RX]", "audio": "[AUDIO]", "video": "[VIDEO]", "movimento": "[MOV]"}


def pacientes_monitorados(pacientes: dict[str, dict]) -> list[str]:
    return [pid for pid, p in pacientes.items() if p.get("monitoramento")]


def figura_vitais(df: pd.DataFrame, titulo: str = "Sinais vitais") -> go.Figure:
    """3 subplots (FC, PA, SpO2) com pontos anomalos destacados."""
    cols = [("fc", "FC (bpm)", (50, 120)), ("pa_sist", "PA sist. (mmHg)", (90, 160)), ("spo2", "SpO2 (%)", (92, 100))]
    cols = [c for c in cols if c[0] in df and df[c[0]].notna().any()]
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=[c[1] for c in cols], vertical_spacing=0.08)
    anom = df[df.get("anomalia", False) == True]  # noqa: E712
    for i, (c, nome, (lo, hi)) in enumerate(cols, start=1):
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[c], mode="lines", name=nome, line=dict(color="#1f77b4", width=1.5)), row=i, col=1)
        fig.add_hrect(y0=lo, y1=hi, fillcolor="green", opacity=0.06, line_width=0, row=i, col=1)
        # marca apenas os pontos cujo motivo cita esta variavel
        an_c = anom[anom["motivo"].str.contains(c, na=False)] if "motivo" in anom else anom
        if len(an_c):
            fig.add_trace(go.Scatter(x=an_c["timestamp"], y=an_c[c], mode="markers", name=f"anomalia {c}",
                                     marker=dict(color="#dc3545", size=7, symbol="x"), showlegend=(i == 1),
                                     hovertext=an_c.get("motivo")), row=i, col=1)
    fig.update_layout(height=180 * len(cols) + 60, title=titulo, margin=dict(l=40, r=20, t=60, b=30), showlegend=False)
    return fig


def figura_postura(serie: pd.DataFrame, eventos: list[dict]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Altura do corpo (px)", "Inclinacao do tronco (graus)"])
    d = serie[serie["detectada"]] if "detectada" in serie else serie
    fig.add_trace(go.Scatter(x=d["tempo_s"], y=d["altura_bbox"], mode="lines+markers", name="altura"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["tempo_s"], y=d["inclinacao_tronco_deg"], mode="lines+markers", name="inclinacao"), row=2, col=1)
    for e in eventos:
        fig.add_vline(x=e["tempo_s"], line=dict(color=CORES_SEV.get(e["severidade"], "red"), dash="dash"),
                      annotation_text=e["tipo"], annotation_position="top left")
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=30), showlegend=False)
    return fig


def alertas_para_df(alertas: list[Alerta | dict]) -> pd.DataFrame:
    rows = [a.to_dict() if isinstance(a, Alerta) else a for a in alertas]
    if not rows:
        return pd.DataFrame(columns=["timestamp", "modalidade", "severidade", "evidencia"])
    df = pd.DataFrame(rows)[["timestamp", "modalidade", "severidade", "evidencia"]]
    ordem = ["critica", "alta", "media", "baixa"]
    return df.sort_values("severidade", key=lambda s: s.map(ordem.index)).reset_index(drop=True)


def simular_stream(df: pd.DataFrame, janela: int = 30, passo: int = 5) -> Iterator[pd.DataFrame]:
    """Entrega fatias crescentes da serie, simulando chegada de dados em tempo real."""
    for fim in range(janela, len(df) + passo, passo):
        yield df.iloc[: min(fim, len(df))]


def alertas_ate(alertas: list[Alerta], ate_timestamp) -> list[Alerta]:
    """Alertas de vitais cujo episodio ja comecou ate o instante do stream; demais modalidades sempre visiveis."""
    out = []
    for a in alertas:
        if a.modalidade == "vitais":
            if pd.Timestamp(a.timestamp) <= pd.Timestamp(ate_timestamp):
                out.append(a)
        else:
            out.append(a)
    return out


def carregar_auditoria(path: str = "outputs/audit_log.json", n: int = 20) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "prompt", "metadata"])
    with open(path, encoding="utf-8") as f:
        log = json.load(f)
    rows = [{"timestamp": e["timestamp"][:19], "evento": e["prompt"][:80],
             "tipo": e["metadata"].get("tipo") or e["metadata"].get("service", ""),
             "simulado": e["metadata"].get("simulated", "")} for e in log[-n:]]
    return pd.DataFrame(rows[::-1])


def ultimo_relatorio(paciente_id: str, pasta: str = "outputs/alerts") -> dict | None:
    arquivos = sorted(glob.glob(os.path.join(pasta, f"monitoramento_{paciente_id}_*.json")))
    if not arquivos:
        return None
    with open(arquivos[-1], encoding="utf-8") as f:
        return json.load(f)
