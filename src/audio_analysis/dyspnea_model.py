"""Modelo supervisionado de dispneia na fala, treinado no SPIRA (USP/HC-FMUSP).

Usa apenas atributos prosodicos robustos a canal de gravacao (o SPIRA tem confusao
total entre classe e canal: pacientes via WhatsApp, controles via app web). Atributos
espectrais (MFCC) foram descartados por aprenderem o canal (AUC=1.0 espuria).
"""
from __future__ import annotations

import os
from typing import Optional

import joblib
import librosa
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dyspnea_logreg.joblib")
FEATURES = ["dur", "pausas", "taxa_fala", "bursts_por_s"]
# Descartados por refletirem o canal de gravacao, nao a fala: MFCC (AUC=1.0 espuria),
# fracao_ativa (mediana 1.0 nos pacientes por piso de ruido do WhatsApp) e cv_energia.


def extrair_prosodia(y: np.ndarray, sr: int = 16000) -> dict:
    """Atributos prosodicos leves (sem pyin) compartilhados por treino e inferencia."""
    y, _ = librosa.effects.trim(y, top_db=30)
    dur = len(y) / sr
    if dur < 0.3:
        return {k: 0.0 for k in FEATURES + ["cv_energia", "fracao_ativa"]}
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    limiar = max(0.1 * np.percentile(rms, 90), 1e-4)
    ativo = rms > max(1e-4, 0.01 * rms.max())
    ref = np.median(rms[ativo]) if ativo.sum() >= 5 else 1.0
    rise = rms[3:] / (rms[:-3] + 1e-6)
    burst = (rms[3:] > 3 * ref) & (rise > 3)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256).astype(np.float64)
    picos = librosa.util.peak_pick(onset, pre_max=3, post_max=3, pre_avg=10, post_avg=10,
                                   delta=float(0.3 * np.std(onset) + 1e-6), wait=4) if onset.max() > 0 else []
    return {
        "dur": dur,
        "pausas": float(np.mean(rms <= limiar)),
        "taxa_fala": len(picos) / dur,
        "bursts_por_s": float(np.sum(np.diff(burst.astype(int)) == 1)) / dur,
        # auxiliares (nao usados pelo modelo; uteis para analise)
        "cv_energia": float(rms.std() / (rms.mean() + 1e-6)),
        "fracao_ativa": float(np.mean(ativo)),
    }


class DyspneaModel:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline = joblib.load(model_path) if os.path.exists(model_path) else None

    @property
    def disponivel(self) -> bool:
        return self.pipeline is not None

    def prever(self, y: np.ndarray, sr: int = 16000) -> Optional[float]:
        """Probabilidade (0-1) de insuficiencia respiratoria na fala; None se modelo ausente."""
        if not self.disponivel:
            return None
        f = extrair_prosodia(y, sr)
        import pandas as pd

        X = pd.DataFrame([[f[c] for c in FEATURES]], columns=FEATURES)
        return float(self.pipeline.predict_proba(X)[0, 1])
