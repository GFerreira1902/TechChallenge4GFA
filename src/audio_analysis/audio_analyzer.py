"""Analisador de audio: acustica (librosa) + Azure Speech + Azure Language -> laudo + alertas."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.anomaly_detection.alerts import Alerta
from src.audio_analysis.acoustic_features import avaliar_indicadores, carregar_audio, extrair_atributos
from src.audio_analysis.azure_language_client import AzureLanguageClient
from src.audio_analysis.azure_speech_client import AzureSpeechClient
from src.audio_analysis.dyspnea_model import DyspneaModel
from src.guardrails.audit_logger import AuditLogger

_ORDEM = ["baixa", "media", "alta", "critica"]
LIMIAR_DISPNEIA = 0.7


@dataclass
class LaudoAudio:
    paciente: str
    arquivo: str
    atributos: dict
    indicadores_acusticos: list[dict]
    transcricao: str
    transcricao_simulada: bool
    sentimento: str
    scores_sentimento: dict
    frases_chave: list[str]
    termos_criticos: list[dict]
    analise_simulada: bool
    prob_dispneia: Optional[float] = None
    alertas: list[Alerta] = field(default_factory=list)
    erros: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alertas"] = [a.to_dict() for a in self.alertas]
        return d


class AudioAnalyzer:
    def __init__(
        self,
        speech_client: Optional[AzureSpeechClient] = None,
        language_client: Optional[AzureLanguageClient] = None,
        audit_logger: Optional[AuditLogger] = None,
        transcrever: bool = True,
        dyspnea_model: Optional[DyspneaModel] = None,
    ):
        self.audit_logger = audit_logger or AuditLogger()
        self.speech = speech_client or AzureSpeechClient(audit_logger=self.audit_logger)
        self.language = language_client or AzureLanguageClient(audit_logger=self.audit_logger)
        self.transcrever = transcrever
        self.dyspnea = dyspnea_model or DyspneaModel()

    def analisar(self, audio_path: str, paciente_id: str, contem_fala: bool = True) -> LaudoAudio:
        y, sr = carregar_audio(audio_path)
        atributos = extrair_atributos(y, sr)
        indicadores = avaliar_indicadores(atributos)
        erros: list[str] = []
        prob = self.dyspnea.prever(y, sr) if contem_fala else None

        texto, trans_sim = "", True
        if self.transcrever and contem_fala:
            t = self.speech.transcrever(audio_path)
            texto, trans_sim = t.texto, t.simulado
            if t.erro:
                erros.append(f"speech: {t.erro}")

        if texto.strip():
            an = self.language.analisar(texto)
            if an.erro:
                erros.append(f"language: {an.erro}")
            sent, scores, frases, termos, an_sim = (an.sentimento, an.scores, an.frases_chave,
                                                    an.termos_criticos, an.simulado)
        else:
            sent, scores, frases, termos, an_sim = "neutral", {}, [], [], True

        laudo = LaudoAudio(paciente_id, os.path.basename(audio_path), atributos.to_dict(), indicadores,
                           texto, trans_sim, sent, scores, frases, termos, an_sim, prob_dispneia=prob, erros=erros)
        laudo.alertas = self._gerar_alertas(laudo)
        return laudo

    @staticmethod
    def _gerar_alertas(l: LaudoAudio) -> list[Alerta]:
        alertas: list[Alerta] = []
        if l.termos_criticos:
            sev = max((t["severidade"] for t in l.termos_criticos), key=_ORDEM.index)
            termos = ", ".join(sorted({t["termo"] for t in l.termos_criticos}))
            alertas.append(Alerta(l.paciente, "audio", sev,
                                  f"Termos criticos na fala: {termos}. Transcricao: \"{l.transcricao[:160]}\""))
        neg = l.scores_sentimento.get("negative", 0)
        if l.sentimento == "negative" and neg >= 0.7 and not l.termos_criticos:
            alertas.append(Alerta(l.paciente, "audio", "media",
                                  f"Sentimento negativo ({neg:.0%}) na fala do paciente"))
        if l.prob_dispneia is not None and l.prob_dispneia >= LIMIAR_DISPNEIA:
            alertas.append(Alerta(l.paciente, "audio", "alta" if l.prob_dispneia >= 0.85 else "media",
                                  f"Padrao de fala compativel com dificuldade respiratoria "
                                  f"(modelo SPIRA, p={l.prob_dispneia:.2f})"))
        if l.indicadores_acusticos:
            sev = max((i["severidade"] for i in l.indicadores_acusticos), key=_ORDEM.index)
            ev = "; ".join(i["evidencia"] for i in l.indicadores_acusticos)
            alertas.append(Alerta(l.paciente, "audio", sev, f"Alteracao vocal/respiratoria: {ev}"))
        return alertas
