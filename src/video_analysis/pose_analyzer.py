"""Analise de postura com YOLOv8n-pose (equivalente funcional ao OpenPose, 17 keypoints COCO).

Gera uma serie temporal por frame (altura do tronco, inclinacao, centro) e detecta
eventos: queda (perda rapida de altura + corpo horizontal), postura anomala
(tronco inclinado por tempo prolongado), imobilidade e saida/entrada de area critica.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.anomaly_detection.alerts import Alerta
from src.video_analysis.object_detector import MODELS_DIR, ROI, Deteccao, pessoas

# indices COCO
NARIZ, OMBRO_E, OMBRO_D, QUADRIL_E, QUADRIL_D, JOELHO_E, JOELHO_D, TORN_E, TORN_D = 0, 5, 6, 11, 12, 13, 14, 15, 16


@dataclass
class Postura:
    tempo_s: float
    detectada: bool
    centro_x: float = math.nan
    centro_y: float = math.nan
    altura_bbox: float = math.nan
    razao_hw: float = math.nan
    inclinacao_tronco_deg: float = math.nan  # 0 = vertical, 90 = horizontal
    keypoints: Optional[np.ndarray] = field(default=None, repr=False)

    def to_row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "keypoints"}


class PoseAnalyzer:
    def __init__(self, pesos: str = "yolov8n-pose.pt", conf: float = 0.4, device: Optional[str] = None):
        self.pesos, self.conf, self.device = pesos, conf, device
        self._model = None

    def _carregar(self):
        if self._model is None:
            import torch
            from ultralytics import YOLO

            if self.device is None:
                self.device = "0" if torch.cuda.is_available() else "cpu"
            path = os.path.join(MODELS_DIR, self.pesos)
            self._model = YOLO(path if os.path.exists(path) else self.pesos)
        return self._model

    def estimar(self, imagem: np.ndarray, tempo_s: float, pessoa_bbox: Optional[Deteccao] = None) -> Postura:
        """Pose do individuo mais confiante; usa bbox do detector como fallback se a pose falhar."""
        r = self._carregar().predict(imagem, verbose=False, device=self.device, conf=self.conf)[0]
        if r.keypoints is not None and len(r.boxes) > 0:
            i = int(r.boxes.conf.argmax())
            kp = r.keypoints.xy[i].cpu().numpy()
            box = r.boxes.xyxy[i].cpu().numpy()
            return postura_de_keypoints(kp, box, tempo_s)
        if pessoa_bbox is not None:
            return postura_de_bbox(pessoa_bbox, tempo_s)
        return Postura(tempo_s, detectada=False)


def postura_de_bbox(d: Deteccao, tempo_s: float) -> Postura:
    cx, cy = d.centro
    incl = 90.0 if d.razao_hw < 0.8 else (0.0 if d.razao_hw > 1.5 else 45.0)
    return Postura(tempo_s, True, cx, cy, d.altura, d.razao_hw, incl)


def postura_de_keypoints(kp: np.ndarray, box: np.ndarray, tempo_s: float) -> Postura:
    x1, y1, x2, y2 = box
    alt, larg = y2 - y1, max(x2 - x1, 1e-6)
    ombro = _media_valida(kp, (OMBRO_E, OMBRO_D))
    quadril = _media_valida(kp, (QUADRIL_E, QUADRIL_D))
    if ombro is not None and quadril is not None:
        dx, dy = ombro[0] - quadril[0], ombro[1] - quadril[1]
        incl = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
    else:
        incl = 90.0 if alt / larg < 0.8 else 0.0
    return Postura(tempo_s, True, (x1 + x2) / 2, (y1 + y2) / 2, alt, alt / larg, incl, kp)


def _media_valida(kp: np.ndarray, idx: tuple[int, int]) -> Optional[np.ndarray]:
    pts = [kp[i] for i in idx if kp[i][0] > 0 and kp[i][1] > 0]
    return np.mean(pts, axis=0) if pts else None


def posturas_para_df(posturas: list[Postura]) -> pd.DataFrame:
    return pd.DataFrame([p.to_row() for p in posturas])


@dataclass
class EventoVideo:
    tipo: str  # queda | postura_anomala | imobilidade | area_critica | sem_pessoa
    tempo_s: float
    severidade: str
    evidencia: str
    frame_indice: Optional[int] = None
    frame_fim: Optional[int] = None  # frame que confirma o evento (ex.: pessoa no chao)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def detectar_eventos(
    df: pd.DataFrame,
    altura_img: int,
    largura_img: int,
    rois: Optional[list[ROI]] = None,
    janela_queda_s: float = 1.0,
    queda_razao_min: float = 0.35,
    inclinacao_anomala_deg: float = 50.0,
    duracao_postura_s: float = 3.0,
    imobilidade_s: float = 20.0,
    frames_idx: Optional[list[int]] = None,
) -> list[EventoVideo]:
    eventos: list[EventoVideo] = []
    d = df[df["detectada"]].reset_index(drop=True)
    if len(d) < 3:
        if len(df) > 0 and df["detectada"].mean() < 0.2:
            eventos.append(EventoVideo("sem_pessoa", 0.0, "media",
                                       f"Paciente visivel em apenas {df['detectada'].mean():.0%} dos frames"))
        return eventos
    t = d["tempo_s"].to_numpy()
    alt = d["altura_bbox"].to_numpy()
    incl = d["inclinacao_tronco_deg"].to_numpy()
    ref_alt = np.nanpercentile(alt[: max(3, len(alt) // 4)], 75)

    # queda: altura cai >= queda_razao_min da referencia dentro de janela_queda_s e tronco fica horizontal
    for i in range(len(d)):
        j = np.searchsorted(t, t[i] + janela_queda_s)
        if j >= len(d):
            break
        if alt[i] >= 0.75 * ref_alt and (alt[i] - alt[j]) / max(alt[i], 1e-6) >= queda_razao_min and incl[j] >= inclinacao_anomala_deg:
            eventos.append(EventoVideo("queda", float(t[i]), "critica",
                                       f"Queda detectada em t={t[i]:.1f}s: altura {alt[i]:.0f}->{alt[j]:.0f}px, "
                                       f"tronco {incl[j]:.0f} graus", _idx(frames_idx, d, i), _idx(frames_idx, d, j)))
            break

    # postura anomala sustentada (sem contar o pos-queda ja reportado)
    t_queda = eventos[0].tempo_s if eventos and eventos[0].tipo == "queda" else math.inf
    inicio = None
    for i in range(len(d)):
        anom = incl[i] >= inclinacao_anomala_deg and t[i] < t_queda
        if anom and inicio is None:
            inicio = i
        if (not anom or i == len(d) - 1) and inicio is not None:
            dur = t[i] - t[inicio]
            if dur >= duracao_postura_s:
                eventos.append(EventoVideo("postura_anomala", float(t[inicio]), "alta",
                                           f"Tronco inclinado >= {inclinacao_anomala_deg:.0f} graus por {dur:.1f}s "
                                           f"a partir de t={t[inicio]:.1f}s", _idx(frames_idx, d, inicio)))
            inicio = None

    # imobilidade: deslocamento do centro abaixo de 2% da imagem por imobilidade_s
    cx, cy = d["centro_x"].to_numpy(), d["centro_y"].to_numpy()
    lim = 0.02 * max(altura_img, largura_img)
    inicio = 0
    for i in range(1, len(d)):
        desloc = math.hypot(cx[i] - cx[inicio], cy[i] - cy[inicio])
        if desloc > lim:
            inicio = i
        elif t[i] - t[inicio] >= imobilidade_s:
            eventos.append(EventoVideo("imobilidade", float(t[inicio]), "alta",
                                       f"Sem deslocamento por {t[i]-t[inicio]:.0f}s a partir de t={t[inicio]:.1f}s",
                                       _idx(frames_idx, d, inicio)))
            break

    # area critica
    for roi in rois or []:
        dentro = [roi.contem((cx[i], cy[i]), largura_img, altura_img) for i in range(len(d))]
        for i in range(1, len(d)):
            if dentro[i] and not dentro[i - 1]:
                eventos.append(EventoVideo("area_critica", float(t[i]), "alta",
                                           f"Paciente entrou na area critica '{roi.nome}' em t={t[i]:.1f}s",
                                           _idx(frames_idx, d, i)))
                break
    return sorted(eventos, key=lambda e: e.tempo_s)


def _idx(frames_idx: Optional[list[int]], d: pd.DataFrame, i: int) -> Optional[int]:
    if frames_idx is None or "frame_indice" not in d:
        return None
    return int(d["frame_indice"].iloc[i])


def eventos_para_alertas(paciente_id: str, eventos: list[EventoVideo], arquivo: str) -> list[Alerta]:
    return [Alerta(paciente_id, "video", e.severidade, f"[{os.path.basename(arquivo)}] {e.evidencia}") for e in eventos]
