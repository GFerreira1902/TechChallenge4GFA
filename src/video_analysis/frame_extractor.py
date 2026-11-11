"""Extracao de frames amostrados de um video (OpenCV)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    indice: int
    tempo_s: float
    imagem: np.ndarray


def info_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Nao foi possivel abrir o video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "largura": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "altura": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duracao_s"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    cap.release()
    return info


def extrair_frames(path: str, passo: int = 3, max_frames: int | None = None) -> Iterator[Frame]:
    """Itera sobre 1 a cada `passo` frames (passo=3 em 30 fps -> 10 amostras/s)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Nao foi possivel abrir o video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx, emitidos = 0, 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % passo == 0:
                yield Frame(idx, idx / fps, img)
                emitidos += 1
                if max_frames and emitidos >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
