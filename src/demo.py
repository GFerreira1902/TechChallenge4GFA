"""Demonstracao ponta a ponta do assistente virtual medico (Fase 3).

Roteiro pensado para gravacao do video de demonstracao, cobrindo os 4 pontos
pedidos no enunciado:
  1) Funcionamento da LLM personalizada (fine-tuning) - compara base vs fine-tuned;
  2) Execucao de um fluxo automatizado (LangGraph);
  3) Resposta a perguntas clinicas contextualizadas (RAG + prontuario);
  4) Logs e validacao das respostas (auditoria + guardrails).

Uso (recomendado, para evitar problemas de acentuacao no console do Windows):
    chcp 65001
    $env:PYTHONIOENCODING="utf-8"
    python -m src.demo
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch

from src.fine_tuning.compare_base_vs_finetuned import QUESTIONS as COMPARE_QUESTIONS
from src.fine_tuning.compare_base_vs_finetuned import generate as generate_answer
from src.fine_tuning.train import BASE_MODEL
from src.langgraph_flows.clinical_flow import build_clinical_flow
from src.langchain_pipeline.rag_chain import MedicalAssistantRAG

ADAPTER_PATH = "outputs/models/qwen2.5-1.5b-lora-medquad-v2/final_adapter"
AUDIT_LOG_PATH = "outputs/audit_log.json"


def secao(titulo: str) -> None:
    print("\n" + "=" * 90)
    print(f" {titulo}")
    print("=" * 90)


def parte_1_fine_tuning() -> None:
    secao("1) LLM PERSONALIZADA - comparacao base vs. fine-tuned (Qwen2.5-1.5B + LoRA/MedQuAD)")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map={"": 0}
    )

    question = COMPARE_QUESTIONS[1]  # tratamento de hipertensao
    print(
        "\nNota: esta pergunta esta em ingles de proposito - o fine-tuning foi "
        "feito 100% com dados do MedQuAD (ingles), entao testamos aqui no "
        "formato exato do treino. Nas proximas secoes, o assistente completo "
        "responde em portugues (contexto real de uso)."
    )
    print(f"\nPergunta: {question}\n")

    base_answer = generate_answer(base_model, tokenizer, question)
    print(f"--- Modelo BASE (sem fine-tuning) ---\n{base_answer}\n")

    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    ft_answer = generate_answer(ft_model, tokenizer, question)
    print(f"--- Modelo FINE-TUNED (MedQuAD) ---\n{ft_answer}\n")

    # Libera a VRAM usada por esta copia manual do modelo antes das proximas secoes,
    # que carregam o modelo novamente via LocalFineTunedLLMClient.
    del base_model, ft_model, tokenizer
    torch.cuda.empty_cache()


def parte_2_fluxo_automatizado() -> None:
    secao("2) FLUXO AUTOMATIZADO (LangGraph) - paciente com alerta critico")

    flow = build_clinical_flow()
    resultado = flow.invoke({"paciente_id": "PAC-001"})
    print(resultado["resumo"])


def parte_3_pergunta_contextualizada() -> None:
    secao("3) PERGUNTA CLINICA CONTEXTUALIZADA (RAG + prontuario) - paciente com sepse")

    assistant = MedicalAssistantRAG()
    resultado = assistant.ask(
        "Quais os passos do pacote da primeira hora para sepse?", paciente_id="PAC-002"
    )
    print(f"Resposta:\n{resultado['resposta']}\n")
    print("Fontes citadas:", ", ".join(f["protocol_id"] for f in resultado["fontes"]))


def parte_4_logs_e_validacao() -> None:
    secao("4) LOGS E VALIDACAO DAS RESPOSTAS (auditoria)")

    if not os.path.exists(AUDIT_LOG_PATH):
        print("Nenhum log de auditoria encontrado ainda.")
        return

    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    print(f"Total de interacoes auditadas: {len(history)}\n")
    for entry in history[-2:]:
        print(f"[{entry['timestamp']}] metadata={entry['metadata']}")
        print(f"resposta: {entry['response'][:200]}...\n")


def main() -> None:
    parte_1_fine_tuning()
    parte_2_fluxo_automatizado()
    parte_3_pergunta_contextualizada()
    parte_4_logs_e_validacao()


if __name__ == "__main__":
    main()
