"""Fluxo de decisao clinica automatizado (LangGraph).

Ao receber os dados de um paciente, o fluxo:
1. Recupera o prontuario (registro estruturado sintetico);
2. Verifica exames pendentes;
3. Verifica alertas criticos e, se houver, emite alerta para a equipe (auditado);
4. Sugere conduta/tratamento via RAG sobre os protocolos internos, sempre com
   validacao humana obrigatoria (guardrails);
5. Gera um laudo clinico formatado (documento) com base no atendimento;
6. Consolida um resumo final do atendimento.
"""
from __future__ import annotations

import argparse
import random
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.guardrails.audit_logger import AuditLogger
from src.guardrails.safety_rules import enforce_human_validation
from src.langchain_pipeline.local_llm_client import LocalFineTunedLLMClient
from src.langchain_pipeline.patient_records import get_patient, load_patients
from src.langchain_pipeline.rag_chain import MedicalAssistantRAG
from src.langchain_pipeline.report_generator import gerar_laudo


class ClinicalFlowState(TypedDict, total=False):
    paciente_id: str
    pergunta: Optional[str]
    paciente: Optional[dict]
    exames_pendentes: list[str]
    tem_exames_pendentes: bool
    alertas_criticos: list[str]
    tem_alerta_critico: bool
    alerta_emitido: Optional[str]
    sugestao_tratamento: Optional[str]
    fontes: list[dict]
    avisos_seguranca: list[str]
    laudo: Optional[str]
    laudo_pdf_path: Optional[str]
    resumo: str


def node_receber_paciente(state: ClinicalFlowState) -> dict:
    paciente = get_patient(state["paciente_id"])
    if paciente is None:
        raise ValueError(f"Paciente {state['paciente_id']} nao encontrado no prontuario.")
    return {"paciente": paciente}


def node_verificar_exames_pendentes(state: ClinicalFlowState) -> dict:
    exames = state["paciente"]["exames_pendentes"]
    return {"exames_pendentes": exames, "tem_exames_pendentes": len(exames) > 0}


def node_verificar_alertas(state: ClinicalFlowState) -> dict:
    alertas = state["paciente"].get("alertas_criticos", [])
    return {"alertas_criticos": alertas, "tem_alerta_critico": len(alertas) > 0}


def node_emitir_alerta(state: ClinicalFlowState, audit_logger: Optional[AuditLogger] = None) -> dict:
    audit_logger = audit_logger or AuditLogger()
    mensagem = (
        f"ALERTA CRITICO - Paciente {state['paciente_id']}: "
        + "; ".join(state["alertas_criticos"])
        + ". Conforme PROT-010, notificar medico assistente e enfermeiro responsavel "
        "imediatamente e registrar confirmacao de recebimento em ate 30 minutos."
    )
    audit_logger.log(
        prompt=f"[fluxo_clinico] alerta critico - paciente {state['paciente_id']}",
        response=mensagem,
        metadata={"paciente_id": state["paciente_id"], "tipo": "alerta_critico"},
    )
    return {"alerta_emitido": mensagem}


def node_sugerir_tratamento(state: ClinicalFlowState) -> dict:
    pergunta = state.get("pergunta") or (
        f"Com base no diagnostico principal ({state['paciente']['diagnostico_principal']}), "
        "quais sao as proximas condutas recomendadas pelos protocolos internos?"
    )
    assistant = MedicalAssistantRAG()
    resultado = assistant.ask(pergunta, paciente_id=state["paciente_id"])

    fontes = resultado["fontes"]
    if fontes and fontes[0].get("conteudo"):
        resposta_baseada_em_protocolo = (
            f"Conforme {fontes[0]['protocol_id']} - {fontes[0]['titulo']}: "
            f"{fontes[0]['conteudo']}"
        )
    else:
        resposta_baseada_em_protocolo = resultado["resposta"]

    resposta_validada, avisos = enforce_human_validation(resposta_baseada_em_protocolo)
    return {
        "sugestao_tratamento": resposta_validada,
        "fontes": fontes,
        "avisos_seguranca": avisos,
    }


def node_gerar_laudo(state: ClinicalFlowState, llm_client: Optional[object] = None) -> dict:
    llm_client = llm_client or LocalFineTunedLLMClient()
    resultado = gerar_laudo(
        llm_client=llm_client,
        paciente=state["paciente"],
        sugestao_tratamento=state["sugestao_tratamento"],
        fontes=state.get("fontes", []),
        alerta_emitido=state.get("alerta_emitido"),
    )
    return {"laudo": resultado["laudo"], "laudo_pdf_path": resultado["pdf_path"]}


def node_finalizar(state: ClinicalFlowState) -> dict:
    linhas = [f"Resumo do atendimento - Paciente {state['paciente_id']}"]

    if state.get("tem_exames_pendentes"):
        linhas.append(f"Exames pendentes: {', '.join(state['exames_pendentes'])}")
    else:
        linhas.append("Exames pendentes: nenhum")

    if state.get("alerta_emitido"):
        linhas.append(f"Alerta emitido: {state['alerta_emitido']}")
    else:
        linhas.append("Alertas críticos: nenhum")

    linhas.append(f"Sugestão de conduta: {state['sugestao_tratamento']}")
    fontes = ", ".join(
        f"{fonte['protocol_id']} - {fonte.get('titulo', 'Protocolo sem título')}"
        for fonte in state.get("fontes", [])
    )
    linhas.append(f"Fontes consultadas: {fontes or 'nenhuma'}")
    linhas.append(f"PDF do laudo: {state.get('laudo_pdf_path', 'nao gerado')}")
    linhas.append(f"\nLaudo clínico gerado:\n{state.get('laudo', 'nao gerado')}")

    return {"resumo": "\n".join(linhas)}


def route_apos_alertas(state: ClinicalFlowState) -> str:
    return "emitir_alerta" if state.get("tem_alerta_critico") else "sugerir_tratamento"


def pergunta_para_paciente(paciente: dict) -> str:
    """Cria uma pergunta coerente com o diagnóstico do paciente sorteado."""
    return (
        f"Quais são as próximas condutas recomendadas para o diagnóstico de "
        f"{paciente['diagnostico_principal']}?"
    )


def selecionar_paciente(paciente_id: Optional[str] = None) -> tuple[str, str]:
    """Seleciona um paciente manualmente ou sorteia um dos prontuários fictícios."""
    pacientes = load_patients()
    if paciente_id:
        if paciente_id not in pacientes:
            raise ValueError(f"Paciente {paciente_id} nao encontrado no prontuario.")
        paciente = pacientes[paciente_id]
    else:
        paciente = random.choice(list(pacientes.values()))

    return paciente["paciente_id"], pergunta_para_paciente(paciente)


def build_clinical_flow():
    graph = StateGraph(ClinicalFlowState)

    graph.add_node("receber_paciente", node_receber_paciente)
    graph.add_node("verificar_exames_pendentes", node_verificar_exames_pendentes)
    graph.add_node("verificar_alertas", node_verificar_alertas)
    graph.add_node("emitir_alerta", node_emitir_alerta)
    graph.add_node("sugerir_tratamento", node_sugerir_tratamento)
    graph.add_node("gerar_laudo", node_gerar_laudo)
    graph.add_node("finalizar", node_finalizar)

    graph.add_edge(START, "receber_paciente")
    graph.add_edge("receber_paciente", "verificar_exames_pendentes")
    graph.add_edge("verificar_exames_pendentes", "verificar_alertas")
    graph.add_conditional_edges(
        "verificar_alertas",
        route_apos_alertas,
        {"emitir_alerta": "emitir_alerta", "sugerir_tratamento": "sugerir_tratamento"},
    )
    graph.add_edge("emitir_alerta", "sugerir_tratamento")
    graph.add_edge("sugerir_tratamento", "gerar_laudo")
    graph.add_edge("gerar_laudo", "finalizar")
    graph.add_edge("finalizar", END)

    return graph.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o fluxo clínico com paciente fictício.")
    parser.add_argument(
        "--paciente-id",
        help="Usa um paciente específico; sem esse argumento, sorteia um paciente a cada execução.",
    )
    args = parser.parse_args()
    paciente_id, pergunta = selecionar_paciente(args.paciente_id)
    print(f"Paciente selecionado: {paciente_id}")
    print(f"Problema selecionado: {pergunta}\n")

    flow = build_clinical_flow()
    resultado = flow.invoke({"paciente_id": paciente_id, "pergunta": pergunta})
    print(resultado["resumo"])
