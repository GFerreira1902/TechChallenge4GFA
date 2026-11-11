"""Orquestra extracao de frames -> YOLO objetos -> YOLO pose -> eventos -> alertas -> PDF."""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from src.anomaly_detection.alerts import Alerta
from src.guardrails.audit_logger import AuditLogger
from src.video_analysis.frame_extractor import extrair_frames, info_video
from src.video_analysis.object_detector import ROI, ObjectDetector, pessoas
from src.video_analysis.pose_analyzer import (
    EventoVideo,
    PoseAnalyzer,
    Postura,
    detectar_eventos,
    eventos_para_alertas,
    posturas_para_df,
)
from src.video_analysis.video_report import anotar_frame, gerar_relatorio_pdf, salvar_figura


@dataclass
class LaudoVideo:
    paciente: str
    arquivo: str
    info: dict
    eventos: list[EventoVideo]
    objetos: dict[str, int]
    serie: pd.DataFrame = field(repr=False)
    figuras: list[str] = field(default_factory=list)
    pdf: Optional[str] = None
    alertas: list[Alerta] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paciente": self.paciente, "arquivo": self.arquivo, "info": self.info,
            "eventos": [e.to_dict() for e in self.eventos], "objetos": self.objetos,
            "figuras": self.figuras, "pdf": self.pdf, "alertas": [a.to_dict() for a in self.alertas],
        }


class VideoAnalyzer:
    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        pose: Optional[PoseAnalyzer] = None,
        audit_logger: Optional[AuditLogger] = None,
        passo: int = 3,
        rois: Optional[list[ROI]] = None,
        gerar_pdf: bool = True,
    ):
        self.detector = detector or ObjectDetector()
        self.pose = pose or PoseAnalyzer()
        self.audit_logger = audit_logger or AuditLogger()
        self.passo = passo
        self.rois = rois or []
        self.gerar_pdf = gerar_pdf

    def analisar(self, video_path: str, paciente_id: str, max_frames: Optional[int] = None) -> LaudoVideo:
        info = info_video(video_path)
        posturas: list[Postura] = []
        frames_cache: dict[int, tuple] = {}
        objetos: Counter = Counter()

        for fr in extrair_frames(video_path, self.passo, max_frames):
            dets = self.detector.detectar(fr.imagem)
            objetos.update({d.classe for d in dets})
            pes = pessoas(dets)
            melhor = max(pes, key=lambda d: d.confianca) if pes else None
            p = self.pose.estimar(fr.imagem, fr.tempo_s, melhor)
            p.frame_indice = fr.indice  # type: ignore[attr-defined]
            posturas.append(p)
            frames_cache[fr.indice] = (fr.imagem, p, dets)

        df = posturas_para_df(posturas)
        if len(df):
            df["frame_indice"] = [p.frame_indice for p in posturas]  # type: ignore[attr-defined]
        eventos = detectar_eventos(df, info["altura"], info["largura"], self.rois, frames_idx=list(df.get("frame_indice", [])))

        figuras = []
        base = os.path.splitext(os.path.basename(video_path))[0]
        for e in eventos:
            for k, sufixo in ((e.frame_indice, "inicio"), (e.frame_fim, "fim")):
                if k is not None and k in frames_cache:
                    img, p, dets = frames_cache[k]
                    figuras.append(salvar_figura(anotar_frame(img, p, e, dets), f"{base}_{e.tipo}_{sufixo}_{k}"))
        if not eventos and frames_cache:
            k = sorted(frames_cache)[len(frames_cache) // 2]
            img, p, dets = frames_cache[k]
            figuras.append(salvar_figura(anotar_frame(img, p, None, dets), f"{base}_normal_{k}"))

        laudo = LaudoVideo(paciente_id, os.path.basename(video_path), info, eventos, dict(objetos), df, figuras)
        laudo.alertas = eventos_para_alertas(paciente_id, eventos, video_path)
        if self.gerar_pdf:
            laudo.pdf = gerar_relatorio_pdf(paciente_id, video_path, info, eventos, figuras, dict(objetos))

        self.audit_logger.log(
            prompt=f"[video_analysis] {os.path.basename(video_path)} paciente={paciente_id}",
            response=f"{len(eventos)} evento(s): " + "; ".join(f"{e.tipo}@{e.tempo_s:.1f}s" for e in eventos),
            metadata={"service": "yolov8", "frames_analisados": len(posturas), "pdf": laudo.pdf},
        )
        return laudo
