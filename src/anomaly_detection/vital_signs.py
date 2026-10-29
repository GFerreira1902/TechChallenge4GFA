"""Detecao de anomalias em series temporais de sinais vitais (FC, PA, SpO2).

Combina limites clinicos, z-score com janela movel e IsolationForest.
Fontes: series sinteticas por paciente (com anomalias injetadas) e CSVs do
PhysioNet BIDMC (validacao).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.anomaly_detection.alerts import Alerta

VITAL_COLS = ["fc", "pa_sist", "spo2"]

# (minimo, maximo, severidade se violado em valor extremo)
LIMITES = {
    "fc": (50, 120),
    "pa_sist": (90, 160),
    "spo2": (92, 100),
}
EXTREMOS = {"fc": (40, 150), "pa_sist": (80, 190), "spo2": (88, 101)}


def gerar_serie_sintetica(
    paciente_id: str, n: int = 240, seed: int | None = None, anomalias: bool = True
) -> pd.DataFrame:
    """Serie 1 amostra/min com ruido; anomalias injetadas e marcadas em `anomalia_real`."""
    if seed is None:
        seed = sum(ord(c) for c in paciente_id)
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01 08:00", periods=n, freq="min")
    df = pd.DataFrame(
        {
            "timestamp": t,
            "fc": rng.normal(78, 3, n),
            "pa_sist": rng.normal(124, 4, n),
            "spo2": np.clip(rng.normal(97, 0.7, n), 80, 100),
            "anomalia_real": False,
        }
    )
    if anomalias:
        a = int(n * 0.5)
        df.loc[a : a + 9, "fc"] += np.linspace(20, 55, 10)
        df.loc[a : a + 9, "anomalia_real"] = True
        b = int(n * 0.75)
        df.loc[b : b + 7, "spo2"] -= np.linspace(4, 9, 8)
        df.loc[b : b + 7, "pa_sist"] -= np.linspace(15, 40, 8)
        df.loc[b : b + 7, "anomalia_real"] = True
    return df


def carregar_bidmc_csv(path: str | Path) -> pd.DataFrame:
    """Le CSV 'Numerics' do BIDMC (colunas HR, SpO2, ...); PA nao existe -> NaN."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["time [s]"], unit="s"),
            "fc": raw["hr"],
            "pa_sist": np.nan,
            "spo2": raw["spo2"],
        }
    )
    df["anomalia_real"] = False
    return df.dropna(subset=["fc", "spo2"]).reset_index(drop=True)


def detectar_vitais(
    df: pd.DataFrame, janela: int = 30, z_limite: float = 4.0, contaminacao: float = 0.05
) -> pd.DataFrame:
    """Adiciona colunas `anomalia`, `motivo` e `score_if` ao DataFrame."""
    df = df.copy().reset_index(drop=True)
    cols = [c for c in VITAL_COLS if df[c].notna().any()]
    motivos: list[list[str]] = [[] for _ in range(len(df))]

    for c in cols:
        lo, hi = LIMITES[c]
        xlo, xhi = EXTREMOS[c]
        # linha de base robusta: mediana/MAD rolantes, deslocadas para nao incluir o ponto atual
        base = df[c].shift(1).rolling(janela, min_periods=max(5, janela // 3))
        med = base.median()
        mad = base.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True) * 1.4826
        z = (df[c] - med) / mad.clip(lower=df[c].std() * 0.25 + 1e-6)
        for i, v in df[c].items():
            if pd.isna(v):
                continue
            if v < lo or v > hi:
                motivos[i].append(f"{c}={v:.0f} fora da faixa {lo}-{hi}")
            elif abs(z.iloc[i]) > z_limite:
                motivos[i].append(f"{c} desvio abrupto (z={z.iloc[i]:.1f})")

    X = df[cols].interpolate().bfill().ffill().to_numpy()
    model = IsolationForest(contamination=contaminacao, random_state=42).fit(X)
    df["score_if"] = -model.score_samples(X)
    outlier_if = model.predict(X) == -1
    for i in np.where(outlier_if)[0]:
        if motivos[i]:
            motivos[i].append("IsolationForest")

    df["motivo"] = ["; ".join(m) for m in motivos]
    df["anomalia"] = [bool(m) for m in motivos]
    return df


def _severidade(row: pd.Series) -> str:
    for c in VITAL_COLS:
        v = row.get(c)
        if pd.notna(v) and (v < EXTREMOS[c][0] or v > EXTREMOS[c][1]):
            return "critica"
    return "alta" if row["motivo"].count(";") >= 1 else "media"


def vitais_para_alertas(paciente_id: str, df: pd.DataFrame) -> list[Alerta]:
    """Agrupa pontos anomalos consecutivos em um alerta por episodio."""
    alertas: list[Alerta] = []
    idx = np.where(df["anomalia"].to_numpy())[0]
    if len(idx) == 0:
        return alertas
    for grupo in np.split(idx, np.where(np.diff(idx) > 3)[0] + 1):
        trecho = df.iloc[grupo]
        pior = trecho.loc[trecho["score_if"].idxmax()]
        sev = max((_severidade(r) for _, r in trecho.iterrows()), key=["media", "alta", "critica"].index)
        ini = trecho["timestamp"].iloc[0].strftime("%H:%M")
        fim = trecho["timestamp"].iloc[-1].strftime("%H:%M")
        alertas.append(
            Alerta(
                paciente=paciente_id,
                modalidade="vitais",
                severidade=sev,
                evidencia=f"Episodio {ini}-{fim}: {pior['motivo']}",
                timestamp=trecho["timestamp"].iloc[0].isoformat(timespec="seconds"),
            )
        )
    return alertas
