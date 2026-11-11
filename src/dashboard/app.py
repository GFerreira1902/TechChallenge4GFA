"""Dashboard Streamlit - Monitoramento Multimodal de Pacientes (Tech Challenge Fase 4).

Executar: streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.anomaly_detection.alerts import nivel_risco, score_risco
from src.dashboard.helpers import (
    CORES_SEV,
    alertas_ate,
    alertas_para_df,
    carregar_auditoria,
    figura_postura,
    figura_vitais,
    pacientes_monitorados,
    simular_stream,
)
from src.guardrails.audit_logger import AuditLogger
from src.langchain_pipeline.patient_records import load_patients

load_dotenv()
st.set_page_config(page_title="Monitoramento Multimodal", page_icon=":hospital:", layout="wide")

AUDIT_PATH = "outputs/audit_log.json"


@st.cache_resource(show_spinner="Carregando modelos (YOLOv8, Azure clients)...")
def _recursos():
    from src.audio_analysis.audio_analyzer import AudioAnalyzer
    from src.fusion.monitoring_flow import build_monitoring_graph
    from src.video_analysis.video_analyzer import VideoAnalyzer

    logger = AuditLogger(AUDIT_PATH)
    audio, video = AudioAnalyzer(audit_logger=logger), VideoAnalyzer(audit_logger=logger)
    return {"audio": audio, "video": video, "graph": build_monitoring_graph(audio, video, logger), "logger": logger}


def _badge(sev: str) -> str:
    return f"<span style='background:{CORES_SEV[sev]};color:white;padding:2px 8px;border-radius:8px;font-size:0.8em'>{sev.upper()}</span>"


# ------------------------------------------------------------------ sidebar
pacientes = load_patients()
ids = pacientes_monitorados(pacientes)
with st.sidebar:
    st.title("Monitoramento Multimodal")
    pid = st.selectbox("Paciente", ids, index=ids.index("PAC-003") if "PAC-003" in ids else 0,
                       format_func=lambda i: f"{i} - {pacientes[i]['diagnostico_principal'][:34]}")
    pac = pacientes[pid]
    st.caption(pac["monitoramento"].get("cenario", ""))
    st.divider()
    rec = _recursos()
    st.markdown(f"**Azure Speech:** {'REAL' if not rec['audio'].speech.mock_mode else 'SIMULADO'}  \n"
                f"**Azure Language:** {'REAL' if not rec['audio'].language.mock_mode else 'SIMULADO'}  \n"
                f"**YOLOv8:** {rec['video'].detector.device or 'auto'}")
    st.divider()
    rodar = st.button("Rodar fluxo completo (LangGraph)", type="primary", use_container_width=True)
    stream = st.button("Simular stream de vitais", use_container_width=True)
    st.caption("Dados ficticios (PAC-00X) + amostras publicas reais: SPIRA, Common Voice, ESC-50, UR Fall, PhysioNet BIDMC.")

# ------------------------------------------------------------------ cabecalho
c1, c2, c3 = st.columns([3, 1, 1])
c1.subheader(f"{pid} - {pac['nome']} ({pac['idade']} anos, {pac['sexo']})")
c1.write(f"**Diagnostico:** {pac['diagnostico_principal']}  |  **Historico:** {pac['historico']}")
c1.write(f"**Medicacoes:** {', '.join(pac['medicacoes_atuais']) or 'nenhuma'}")

estado_key = f"estado_{pid}"
if rodar:
    with st.spinner("Executando: vitais -> prescricoes -> audio (Azure) -> video (YOLO) -> fusao -> alerta..."):
        t0 = time.time()
        st.session_state[estado_key] = rec["graph"].invoke({"paciente_id": pid})
        st.session_state[f"{estado_key}_t"] = time.time() - t0

estado = st.session_state.get(estado_key)
if estado:
    c2.metric("Score de risco", f"{estado['score_risco']:.0f}/100")
    c3.metric("Nivel", estado["nivel_risco"].upper())
    st.caption(f"Fluxo executado em {st.session_state.get(f'{estado_key}_t', 0):.1f}s")
else:
    c2.metric("Score de risco", "-"); c3.metric("Nivel", "-")
    st.info("Clique em **Rodar fluxo completo** na barra lateral para analisar as modalidades deste paciente.")

# ------------------------------------------------------------------ abas
tab_alertas, tab_vitais, tab_audio, tab_video, tab_audit = st.tabs(
    ["Alertas", "Sinais vitais", "Audio", "Video", "Auditoria"])

with tab_alertas:
    if estado:
        if estado.get("mensagem_alerta"):
            st.error(estado["mensagem_alerta"].replace("\n", "  \n"))
        else:
            st.success("Nenhum alerta emitido - paciente dentro do padrao esperado.")
        df_al = alertas_para_df(estado["alertas"])
        if len(df_al):
            for _, r in df_al.iterrows():
                st.markdown(f"{_badge(r['severidade'])} **{r['modalidade']}** - {r['evidencia']}", unsafe_allow_html=True)
        st.caption(f"Relatorio JSON: `{estado.get('relatorio_path')}`")
    else:
        st.write("Sem execucao ainda.")

with tab_vitais:
    if estado and estado.get("vitais_df") is not None:
        df = estado["vitais_df"]
        fonte = pac["monitoramento"].get("sinais_vitais", {}).get("fonte", "sintetico")
        st.caption(f"Fonte: {fonte} | {len(df)} amostras | {int(df['anomalia'].sum())} pontos anomalos")
        if stream:
            ph_fig, ph_al = st.empty(), st.empty()
            vistos: set[str] = set()
            for fatia in simular_stream(df, janela=40, passo=8):
                ph_fig.plotly_chart(figura_vitais(fatia, "Stream em tempo real (simulado)"), use_container_width=True)
                ativos = alertas_ate([a for a in estado["alertas"] if a.modalidade == "vitais"], fatia["timestamp"].iloc[-1])
                novos = [a for a in ativos if a.evidencia not in vistos]
                for a in novos:
                    vistos.add(a.evidencia)
                    rec["logger"].log(prompt=f"[dashboard-stream] {pid}", response=a.mensagem(),
                                      metadata={"tipo": "alerta_stream", "severidade": a.severidade})
                if ativos:
                    ph_al.error("  \n".join(f"[{a.severidade.upper()}] {a.evidencia}" for a in ativos))
                time.sleep(0.25)
            st.success("Stream concluido.")
        else:
            st.plotly_chart(figura_vitais(df), use_container_width=True)
    else:
        st.write("Execute o fluxo para carregar a serie.")

with tab_audio:
    la = estado.get("laudo_audio") if estado else None
    if la:
        cfg = pac["monitoramento"].get("audio", {})
        st.caption(f"Arquivo: {cfg.get('arquivo')} | fonte: {cfg.get('fonte')}")
        if cfg.get("arquivo") and os.path.exists(cfg["arquivo"]):
            st.audio(cfg["arquivo"])
        a1, a2, a3 = st.columns(3)
        a1.metric("Sentimento (Azure Language)", la["sentimento"], f"neg {la['scores_sentimento'].get('negative', 0):.2f}")
        a2.metric("Prob. dispneia (modelo SPIRA)", f"{la['prob_dispneia']:.2f}" if la.get("prob_dispneia") is not None else "-")
        a3.metric("Fonte da transcricao", "simulada" if la["transcricao_simulada"] else "Azure (real)")
        st.markdown(f"**Transcricao:** _{la['transcricao']}_")
        st.write("**Termos criticos:**", ", ".join(t["termo"] for t in la["termos_criticos"]) or "nenhum")
        st.write("**Frases-chave:**", ", ".join(la.get("frases_chave", [])[:10]) or "-")
        if la["indicadores_acusticos"]:
            st.dataframe(pd.DataFrame(la["indicadores_acusticos"]), use_container_width=True, hide_index=True)
        with st.expander("Atributos acusticos (librosa)"):
            st.json(la["atributos"])
        if estado.get("laudos_audio") and len(estado["laudos_audio"]) > 1:
            ex = estado["laudos_audio"][1]
            st.caption(f"Audio adicional: {ex['arquivo']} - indicadores: {[i['indicador'] for i in ex['indicadores_acusticos']]}")
    else:
        st.write("Paciente sem audio ou fluxo nao executado.")

with tab_video:
    lv = estado.get("laudo_video") if estado else None
    if lv:
        cfg = pac["monitoramento"].get("video", {})
        st.caption(f"Arquivo: {cfg.get('arquivo')} | fonte: {cfg.get('fonte')} | objetos: {lv['objetos']}")
        v1, v2 = st.columns([1, 1])
        if cfg.get("arquivo") and os.path.exists(cfg["arquivo"]):
            v1.video(cfg["arquivo"])
        for f in lv.get("figuras", [])[:2]:
            if os.path.exists(f):
                v2.image(f, caption=os.path.basename(f), use_container_width=True)
        if lv["eventos"]:
            for e in lv["eventos"]:
                st.markdown(f"{_badge(e['severidade'])} **{e['tipo']}** @ {e['tempo_s']:.1f}s - {e['evidencia']}", unsafe_allow_html=True)
        else:
            st.success("Nenhum desvio detectado no video.")
        serie = pd.DataFrame(lv.get("serie", [])) if isinstance(lv.get("serie"), list) else None
        if lv.get("pdf"):
            st.caption(f"Relatorio PDF: `{lv['pdf']}`")
    else:
        st.write("Paciente sem video ou fluxo nao executado.")

with tab_audit:
    st.caption(f"Ultimos registros de `{AUDIT_PATH}` (toda chamada Azure/YOLO e todo alerta sao auditados)")
    st.dataframe(carregar_auditoria(AUDIT_PATH, 25), use_container_width=True, hide_index=True)
