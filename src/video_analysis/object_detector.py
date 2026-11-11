"""Deteccao de objetos (YOLOv8n) com ROI de area critica configuravel.

O modelo e carregado sob demanda; em testes pode ser substituido por um stub
que implemente `detectar(imagem) -> list[Deteccao]`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
CLASSES_INTERESSE = {"person", "bed", "chair", "couch", "bench", "bottle", "cell phone", "laptop", "tv"}


@dataclass
class Deteccao:
    classe: str
    confianca: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def centro(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def altura(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def largura(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def razao_hw(self) -> float:
        """altura/largura: ~2+ em pe, <1 deitado."""
        return self.altura / max(self.largura, 1e-6)


@dataclass
class ROI:
    """Area critica em coordenadas normalizadas (0-1)."""
    nome: str
    x1: float
    y1: float
    x2: float
    y2: float

    def contem(self, ponto_xy: tuple[float, float], largura: int, altura: int) -> bool:
        x, y = ponto_xy[0] / largura, ponto_xy[1] / altura
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class ObjectDetector:
    def __init__(self, pesos: str = "yolov8n.pt", conf: float = 0.4, device: Optional[str] = None):
        self.pesos = pesos
        self.conf = conf
        self.device = device
        self._model = None
        self.names: dict[int, str] = {}

    def _carregar(self):
        if self._model is None:
            import torch
            from ultralytics import YOLO

            if self.device is None:
                self.device = "0" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)
            path = os.path.join(MODELS_DIR, self.pesos)
            self._model = YOLO(path if os.path.exists(path) else self.pesos)
            self.names = self._model.names
        return self._model

    def detectar(self, imagem: np.ndarray, apenas_interesse: bool = True) -> list[Deteccao]:
        model = self._carregar()
        r = model.predict(imagem, verbose=False, device=self.device, conf=self.conf)[0]
        out = []
        for box, cls, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            nome = self.names[int(cls)]
            if apenas_interesse and nome not in CLASSES_INTERESSE:
                continue
            out.append(Deteccao(nome, float(cf), tuple(float(v) for v in box)))
        return out


def pessoas(deteccoes: list[Deteccao]) -> list[Deteccao]:
    return [d for d in deteccoes if d.classe == "person"]
