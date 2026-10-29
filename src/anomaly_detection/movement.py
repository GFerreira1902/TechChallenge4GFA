"""Padroes de movimentacao do paciente internado (trajetoria/altura do centroide).

Entrada: DataFrame com `t` (s), `x`, `y`, `altura` (px ou m), tipicamente extraido da
pose/deteccao YOLOv8 do modulo video_analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly_detection.alerts import Alerta


def gerar_trajetoria_sintetica(
    n: int = 300, fps: float = 1.0, queda_em: int | None = 200, imobilidade: tuple[int, int] | None = (240, 290),
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    x = np.cumsum(rng.normal(0, 0.05, n))
    y = np.cumsum(rng.normal(0, 0.05, n))
    altura = rng.normal(1.7, 0.03, n)
    if queda_em is not None:
        altura[queda_em : queda_em + 3] = [1.2, 0.7, 0.4]
        altura[queda_em + 3 :] = 0.4 + rng.normal(0, 0.02, n - queda_em - 3)
    if imobilidade:
        a, b = imobilidade
        x[a:b], y[a:b] = x[a], y[a]
    return pd.DataFrame({"t": t, "x": x, "y": y, "altura": altura})


def analisar_movimento(
    paciente_id: str,
    df: pd.DataFrame,
    queda_razao: float = 0.5,
    janela_imobil: float = 30.0,
    vel_min: float = 0.02,
    vel_max_z: float = 5.0,
) -> list[Alerta]:
    alertas: list[Alerta] = []
    t = df["t"].to_numpy()
    dt = np.diff(t, prepend=t[0] - 1)
    vel = np.hypot(np.diff(df["x"], prepend=df["x"][0]), np.diff(df["y"], prepend=df["y"][0])) / dt
    h = df["altura"].to_numpy()

    # queda: perda rapida de altura (>= queda_razao da referencia em 3 s)
    ref = np.median(h[: max(5, len(h) // 10)])
    for i in range(len(h)):
        j = np.searchsorted(t, t[i] + 3)
        if j < len(h) and (ref - h[min(j, len(h) - 1)]) / ref >= queda_razao and h[i] > ref * 0.85:
            alertas.append(Alerta(paciente_id, "movimento", "critica",
                                  f"Possivel queda em t={t[i]:.0f}s (altura {h[i]:.2f} -> {h[j]:.2f})"))
            break

    # imobilidade prolongada
    parado = vel < vel_min
    ini = None
    for i, p in enumerate(parado):
        if p and ini is None:
            ini = i
        if (not p or i == len(parado) - 1) and ini is not None:
            dur = t[i] - t[ini]
            if dur >= janela_imobil:
                alertas.append(Alerta(paciente_id, "movimento", "alta",
                                      f"Imobilidade por {dur:.0f}s a partir de t={t[ini]:.0f}s"))
            ini = None

    # agitacao: velocidade anomala frente a mediana robusta
    med = np.median(vel)
    mad = np.median(np.abs(vel - med)) * 1.4826 + 1e-6
    picos = np.where((vel - med) / mad > vel_max_z * 3)[0]
    if len(picos) >= 3:
        alertas.append(Alerta(paciente_id, "movimento", "media",
                              f"Movimentacao agitada atipica ({len(picos)} amostras acima do padrao)"))
    return alertas
