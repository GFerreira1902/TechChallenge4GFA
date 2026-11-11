"""Fluxo de monitoramento multimodal (LangGraph).

coletar_vitais -> analisar_prescricoes -> analisar_audio -> analisar_video
    -> consolidar_anomalias -> [condicional] emitir_alerta -> gerar_relatorio

Cada no e uma funcao pura sobre o estado; dependencias externas (analisadores,
audit logger) sao injetaveis para testes. Todo alerta passa por
`enforce_human_validation` e e registrado no `AuditLogger`.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from functools import partial
from typing import Any, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.anomaly_detection.alerts import Alerta, nivel_risco, score_risco
from src.anomaly_detection.movement import analisar_movimento
from src.anomaly_detection.prescriptions import analisar_prescricoes
from src.anomaly_detection.vital_signs import (
    carregar_bidmc_csv,
    detectar_vitais,
    gerar_serie_sintetica,
    vitais_para_alertas,
)
from src.guardrails.audit_logger import AuditLogger
from src.guardrails.safety_rules import enforce_human_validation
from src.langchain_pipeline.patient_records import get_patient

ALERTS_DIR = "outputs/alerts"
LIMIAR_ALERTA = 30.0  # score de risco a partir do qual a equipe e notificada


class MonitoringState(TypedDict, total=False):
    paciente_id: str
    paciente: dict
    monitoramento: dict
    vitais_df: Any  # pd.DataFrame com colunas de deteccao
    laudo_audio: Optional[dict]
    laudo_video: Optional[dict]
    alertas: list[Alerta]
    score_risco: float
    nivel_risco: str
    deve_alertar: bool
    mensagem_alerta: Optional[str]
    avisos_seguranca: list[str]
    relatorio: Optional[dict]
    relatorio_path: Optional[str]
    erros: list[str]


# ---------------------------------------------------------------- nos
def node_receber_paciente(state: MonitoringState) -> dict:
    pac = get_patient(state["paciente_id"])
    if pac is None:
        raise ValueError(f"Paciente {state['paciente_id']} nao encontrado.")
    return {"paciente": pac, "monitoramento": pac.get("monitoramento", {}), "alertas": [], "erros": []}


def node_coletar_vitais(state: MonitoringState) -> dict:
    cfg = state["monitoramento"].get("sinais_vitais") or {"fonte": "sintetico", "anomalias": False}
    if cfg.get("fonte") == "bidmc" and os.path.exists(cfg.get("arquivo", "")):
        df = carregar_bidmc_csv(cfg["arquivo"])
    else:
        df = gerar_serie_sintetica(state["paciente_id"], seed=cfg.get("seed"), anomalias=cfg.get("anomalias", True))
    df = detectar_vitais(df)
    return {"vitais_df": df, "alertas": state["alertas"] + vitais_para_alertas(state["paciente_id"], df)}


def node_analisar_prescricoes(state: MonitoringState) -> dict:
    hist = state["monitoramento"].get("historico_prescricoes", [])
    novos = analisar_prescricoes(state["paciente_id"], hist, medicacoes_atuais=state["paciente"].get("medicacoes_atuais"))
    return {"alertas": state["alertas"] + novos}


def node_analisar_audio(state: MonitoringState, audio_analyzer=None) -> dict:
    cfgs = [c for c in (state["monitoramento"].get("audio"), state["monitoramento"].get("audio_extra")) if c]
    if not cfgs or audio_analyzer is None:
        return {"laudo_audio": None}
    alertas, laudos, erros = [], [], list(state.get("erros", []))
    for cfg in cfgs:
        if not os.path.exists(cfg["arquivo"]):
            erros.append(f"audio ausente: {cfg['arquivo']}")
            continue
        try:
            laudo = audio_analyzer.analisar(cfg["arquivo"], state["paciente_id"], contem_fala=cfg.get("contem_fala", True))
            alertas += laudo.alertas
            laudos.append(laudo.to_dict())
        except Exception as exc:
            erros.append(f"audio {os.path.basename(cfg['arquivo'])}: {exc}")
    return {"laudo_audio": laudos[0] if laudos else None, "alertas": state["alertas"] + alertas, "erros": erros,
            **({"laudos_audio": laudos} if len(laudos) > 1 else {})}


def node_analisar_video(state: MonitoringState, video_analyzer=None) -> dict:
    cfg = state["monitoramento"].get("video")
    if not cfg or video_analyzer is None or not os.path.exists(cfg["arquivo"]):
        return {"laudo_video": None}
    try:
        laudo = video_analyzer.analisar(cfg["arquivo"], state["paciente_id"])
    except Exception as exc:
        return {"laudo_video": None, "erros": state.get("erros", []) + [f"video: {exc}"]}
    alertas = list(laudo.alertas)
    # padroes de movimentacao: trajetoria do centroide alimenta o detector de movimento
    serie = laudo.serie
    if len(serie) and serie["detectada"].sum() >= 3:
        traj = serie[serie["detectada"]].rename(columns={"tempo_s": "t", "centro_x": "x", "centro_y": "y", "altura_bbox": "altura"})
        traj = traj[["t", "x", "y", "altura"]].reset_index(drop=True)
        traj[["x", "y", "altura"]] = traj[["x", "y", "altura"]] / max(laudo.info.get("altura", 1), 1)
        mov = analisar_movimento(state["paciente_id"], traj, janela_imobil=10.0, vel_min=0.01)
        # evita duplicar queda ja reportada pelo modulo de video
        tipos_video = {e.tipo for e in laudo.eventos}
        for a in mov:
            if ("queda" in a.evidencia.lower() and "queda" in tipos_video):
                continue
            alertas.append(a)
    return {"laudo_video": laudo.to_dict(), "alertas": state["alertas"] + alertas}


def node_consolidar_anomalias(state: MonitoringState) -> dict:
    alertas = state["alertas"]
    score = score_risco(alertas)
    return {"score_risco": score, "nivel_risco": nivel_risco(score), "deve_alertar": score >= LIMIAR_ALERTA
            or any(a.severidade == "critica" for a in alertas)}


def decidir_alerta(state: MonitoringState) -> str:
    return "emitir_alerta" if state["deve_alertar"] else "gerar_relatorio"


def node_emitir_alerta(state: MonitoringState, audit_logger: Optional[AuditLogger] = None) -> dict:
    audit_logger = audit_logger or AuditLogger()
    pac = state["paciente"]
    por_mod: dict[str, list[Alerta]] = {}
    for a in state["alertas"]:
        por_mod.setdefault(a.modalidade, []).append(a)
    linhas = [f"[ALERTA {state['nivel_risco'].upper()}] Paciente {pac['paciente_id']} ({pac['idade']} anos) - "
              f"{pac['diagnostico_principal']} - score de risco {state['score_risco']:.0f}/100",
              f"Modalidades com anomalia: {', '.join(sorted(por_mod))}"]
    for mod, lst in por_mod.items():
        pior = max(lst, key=lambda a: ["baixa", "media", "alta", "critica"].index(a.severidade))
        linhas.append(f"  - {mod} ({pior.severidade}): {pior.evidencia[:180]}")
    linhas.append("Acao: notificar medico assistente e enfermagem; confirmar recebimento em ate 15 minutos.")
    mensagem, avisos = enforce_human_validation("\n".join(linhas))
    audit_logger.log(prompt=f"[monitoring_flow] alerta {pac['paciente_id']}", response=mensagem,
                     metadata={"tipo": "alerta_equipe", "score": state["score_risco"], "nivel": state["nivel_risco"],
                               "n_alertas": len(state["alertas"]), "modalidades": sorted(por_mod)})
    return {"mensagem_alerta": mensagem, "avisos_seguranca": avisos}


def node_gerar_relatorio(state: MonitoringState, audit_logger: Optional[AuditLogger] = None) -> dict:
    audit_logger = audit_logger or AuditLogger()
    ts = datetime.now()
    df = state.get("vitais_df")
    rel = {
        "paciente_id": state["paciente_id"],
        "gerado_em": ts.isoformat(timespec="seconds"),
        "cenario": state["monitoramento"].get("cenario"),
        "score_risco": state["score_risco"],
        "nivel_risco": state["nivel_risco"],
        "alerta_emitido": state.get("mensagem_alerta") is not None,
        "mensagem_alerta": state.get("mensagem_alerta"),
        "alertas": [a.to_dict() for a in state["alertas"]],
        "vitais": {"amostras": int(len(df)), "anomalias": int(df["anomalia"].sum())} if df is not None else None,
        "audio": _resumo_audio(state.get("laudo_audio")),
        "video": _resumo_video(state.get("laudo_video")),
        "erros": state.get("erros", []),
    }
    os.makedirs(ALERTS_DIR, exist_ok=True)
    path = os.path.join(ALERTS_DIR, f"monitoramento_{state['paciente_id']}_{ts:%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2, default=str)
    audit_logger.log(prompt=f"[monitoring_flow] relatorio {state['paciente_id']}",
                     response=f"score={rel['score_risco']} nivel={rel['nivel_risco']} alertas={len(rel['alertas'])}",
                     metadata={"tipo": "relatorio", "path": path})
    return {"relatorio": rel, "relatorio_path": path}


def _resumo_audio(l: Optional[dict]) -> Optional[dict]:
    if not l:
        return None
    return {k: l.get(k) for k in ("arquivo", "transcricao", "transcricao_simulada", "sentimento", "scores_sentimento",
                                  "termos_criticos", "prob_dispneia", "indicadores_acusticos")}


def _resumo_video(l: Optional[dict]) -> Optional[dict]:
    if not l:
        return None
    return {k: l.get(k) for k in ("arquivo", "eventos", "objetos", "pdf", "figuras")}


# ---------------------------------------------------------------- grafo
def build_monitoring_graph(audio_analyzer=None, video_analyzer=None, audit_logger: Optional[AuditLogger] = None):
    g = StateGraph(MonitoringState)
    g.add_node("receber_paciente", node_receber_paciente)
    g.add_node("coletar_vitais", node_coletar_vitais)
    g.add_node("analisar_prescricoes", node_analisar_prescricoes)
    g.add_node("analisar_audio", partial(node_analisar_audio, audio_analyzer=audio_analyzer))
    g.add_node("analisar_video", partial(node_analisar_video, video_analyzer=video_analyzer))
    g.add_node("consolidar_anomalias", node_consolidar_anomalias)
    g.add_node("emitir_alerta", partial(node_emitir_alerta, audit_logger=audit_logger))
    g.add_node("gerar_relatorio", partial(node_gerar_relatorio, audit_logger=audit_logger))

    g.add_edge(START, "receber_paciente")
    g.add_edge("receber_paciente", "coletar_vitais")
    g.add_edge("coletar_vitais", "analisar_prescricoes")
    g.add_edge("analisar_prescricoes", "analisar_audio")
    g.add_edge("analisar_audio", "analisar_video")
    g.add_edge("analisar_video", "consolidar_anomalias")
    g.add_conditional_edges("consolidar_anomalias", decidir_alerta,
                            {"emitir_alerta": "emitir_alerta", "gerar_relatorio": "gerar_relatorio"})
    g.add_edge("emitir_alerta", "gerar_relatorio")
    g.add_edge("gerar_relatorio", END)
    return g.compile()


def executar_monitoramento(paciente_id: str, com_audio: bool = True, com_video: bool = True,
                           audit_logger: Optional[AuditLogger] = None) -> MonitoringState:
    """Conveniencia: instancia analisadores reais (Azure/YOLO, com mock quando sem chave) e roda o grafo."""
    audit_logger = audit_logger or AuditLogger()
    audio = video = None
    if com_audio:
        from src.audio_analysis.audio_analyzer import AudioAnalyzer
        audio = AudioAnalyzer(audit_logger=audit_logger)
    if com_video:
        from src.video_analysis.video_analyzer import VideoAnalyzer
        video = VideoAnalyzer(audit_logger=audit_logger)
    graph = build_monitoring_graph(audio, video, audit_logger)
    return graph.invoke({"paciente_id": paciente_id})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fluxo de monitoramento multimodal")
    ap.add_argument("--paciente-id", default="PAC-003")
    ap.add_argument("--sem-audio", action="store_true")
    ap.add_argument("--sem-video", action="store_true")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    st = executar_monitoramento(args.paciente_id, not args.sem_audio, not args.sem_video)
    print(f"\n{st['paciente_id']} | score {st['score_risco']} | nivel {st['nivel_risco']} | alertas {len(st['alertas'])}")
    for a in st["alertas"]:
        print(f"  - [{a.modalidade}/{a.severidade}] {a.evidencia[:140]}")
    if st.get("mensagem_alerta"):
        print("\n" + st["mensagem_alerta"])
    print("\nrelatorio:", st["relatorio_path"])
