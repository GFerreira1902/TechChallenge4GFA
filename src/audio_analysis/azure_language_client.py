"""Cliente Azure AI Language (Text Analytics): sentimento, frases-chave,
entidades de saude e termos criticos. mock_mode sem AZURE_LANGUAGE_KEY.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from src.guardrails.audit_logger import AuditLogger

# Termos que, presentes na fala do paciente, elevam o risco (comparacao sem acento)
TERMOS_CRITICOS = {
    "critica": ["falta de ar", "nao consigo respirar", "dor no peito", "desmaiei", "desmaio",
                "sangramento", "convulsao"],
    "alta": ["tontura", "tonto", "sufoco", "sufocando", "aperto no peito", "palpitacao",
             "confuso", "confusa", "vomito", "vomitando", "febre alta"],
    "media": ["cansaco", "cansado", "cansada", "fadiga", "fraqueza", "dor", "enjoo", "nausea",
              "tosse", "febre", "chiado", "sem forca"],
}


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def encontrar_termos_criticos(texto: str) -> list[dict]:
    norm = _normalizar(texto)
    achados = []
    for sev, termos in TERMOS_CRITICOS.items():
        for t in termos:
            if re.search(rf"\b{re.escape(t)}\b", norm):
                achados.append({"termo": t, "severidade": sev})
    return achados


@dataclass
class AnaliseTexto:
    sentimento: str  # positive | neutral | negative | mixed
    scores: dict
    frases_chave: list[str]
    entidades: list[dict]
    termos_criticos: list[dict]
    simulado: bool
    erro: Optional[str] = None
    extras: dict = field(default_factory=dict)


class AzureLanguageClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        language: str = "pt",
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.api_key = api_key or os.environ.get("AZURE_LANGUAGE_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_LANGUAGE_ENDPOINT")
        self.language = language
        self.mock_mode = not (self.api_key and self.endpoint)
        self.audit_logger = audit_logger or AuditLogger()
        self._client = None

    def analisar(self, texto: str) -> AnaliseTexto:
        termos = encontrar_termos_criticos(texto)
        if self.mock_mode or not texto.strip():
            result = self._mock(texto, termos)
        else:
            result = self._analisar_azure(texto, termos)

        self.audit_logger.log(
            prompt=f"[azure_language] analisar: {texto[:200]}",
            response=f"sentimento={result.sentimento}; termos={[t['termo'] for t in result.termos_criticos]}",
            metadata={"service": "azure_language", "simulated": result.simulado, "erro": result.erro},
        )
        return result

    def _mock(self, texto: str, termos: list[dict]) -> AnaliseTexto:
        # heuristica local: sentimento negativo se houver termo critico
        neg = min(1.0, 0.3 + 0.25 * len(termos)) if termos else 0.1
        sent = "negative" if neg >= 0.5 else "neutral"
        palavras = [p for p in re.findall(r"\w{5,}", texto.lower())][:8]
        return AnaliseTexto(sent, {"positive": round(1 - neg, 2) * 0.3, "neutral": 0.2, "negative": round(neg, 2)},
                            palavras, [], termos, simulado=True)

    def _cliente(self):
        if self._client is None:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential
            self._client = TextAnalyticsClient(self.endpoint, AzureKeyCredential(self.api_key))
        return self._client

    def _analisar_azure(self, texto: str, termos: list[dict]) -> AnaliseTexto:
        try:
            cli = self._cliente()
            docs = [texto]
            sent = cli.analyze_sentiment(docs, language=self.language)[0]
            kp = cli.extract_key_phrases(docs, language=self.language)[0]
            try:
                ents = cli.recognize_entities(docs, language=self.language)[0]
                entidades = [{"texto": e.text, "categoria": e.category, "confianca": round(e.confidence_score, 2)}
                             for e in ents.entities]
            except Exception:
                entidades = []
            cs = sent.confidence_scores
            return AnaliseTexto(
                sentimento=sent.sentiment,
                scores={"positive": cs.positive, "neutral": cs.neutral, "negative": cs.negative},
                frases_chave=list(kp.key_phrases),
                entidades=entidades,
                termos_criticos=termos,
                simulado=False,
            )
        except Exception as exc:
            r = self._mock(texto, termos)
            r.simulado = False
            r.erro = str(exc)
            return r
