"""Auditoria/explicabilidade: histórico persistente de interações com o LLM.

Padrão adaptado da Fase 2 (ver docs/reference/fase2_llm_explainer.py), generalizado
para qualquer resposta gerada pelo assistente (RAG, fluxos LangGraph, etc).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional


class AuditLogger:
    def __init__(self, history_path: str = "outputs/audit_log.json"):
        self.history_path = history_path
        dirname = os.path.dirname(self.history_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def log(self, prompt: str, response: str, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Registra prompt/resposta/metadados e retorna a entrada persistida."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
        }

        history = []
        if os.path.exists(self.history_path):
            with open(self.history_path, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []

        history.append(entry)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

        return entry
