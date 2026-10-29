"""Cliente LLM (Groq/Llama) com fallback seguro e auditoria automática.

Padrão adaptado da Fase 2 (ver docs/reference/fase2_llm_explainer.py): se a
GROQ_API_KEY não estiver disponível no ambiente, o cliente entra em modo
simulado (mock_mode) ao invés de falhar, garantindo continuidade do fluxo.
Toda interação (real ou simulada) é registrada via AuditLogger para atender
ao requisito de explicabilidade/auditoria do assistente.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from groq import Groq

from src.guardrails.audit_logger import AuditLogger

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class MedicalAssistantLLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.mock_mode = not self.api_key
        self.client = None if self.mock_mode else Groq(api_key=self.api_key, timeout=20.0)
        self.audit_logger = audit_logger or AuditLogger()

    def ask(
        self,
        system_prompt: str,
        user_message: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Envia a mensagem ao LLM (ou gera resposta simulada) e audita o resultado."""
        if self.mock_mode:
            response = (
                "*[MODO SIMULADO - GROQ_API_KEY AUSENTE]*\n\n"
                f"{user_message}\n\n"
                "Configure a variável de ambiente GROQ_API_KEY para obter respostas "
                "geradas pelo LLM real."
            )
        else:
            try:
                completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    model=self.model,
                    temperature=0.4,
                    max_tokens=800,
                )
                response = completion.choices[0].message.content
            except Exception as exc:
                response = f"Erro ao contatar a API da Groq: {exc}"

        self.audit_logger.log(
            prompt=user_message,
            response=response,
            metadata={**(metadata or {}), "model": self.model, "simulated": self.mock_mode},
        )
        return response
