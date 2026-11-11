"""Demonstracao ponta a ponta do monitoramento multimodal (Fase 4).

Roteiro para a gravacao do video, cobrindo os 4 pontos do enunciado:
  1) Analise de audio (librosa + Azure Speech to Text + Azure AI Language);
  2) Analise de video (YOLOv8n + YOLOv8n-pose, relatorio PDF de desvios);
  3) Deteccao de anomalias (vitais, prescricoes, movimento) e resposta (fusao LangGraph);
  4) Integracao Azure + fluxo final do alerta a equipe medica (auditoria e guardrails).

Uso (console Windows):
    chcp 65001
    $env:PYTHONIOENCODING="utf-8"
    python -m src.demo [--paciente PAC-003] [--rapido]
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv

load_dotenv()

from src.audio_analysis.audio_analyzer import AudioAnalyzer
from src.fusion.monitoring_flow import build_monitoring_graph
from src.guardrails.audit_logger import AuditLogger
from src.langchain_pipeline.patient_records import get_patient
from src.video_analysis.video_analyzer import VideoAnalyzer

AUDIT_LOG_PATH = "outputs/audit_log.json"


def secao(titulo: str) -> None:
    print("\n" + "=" * 90)
    print(f" {titulo}")
    print("=" * 90)


def parte_1_audio(audio: AudioAnalyzer, pac: dict) -> None:
    secao("1) ANALISE DE AUDIO - voz do paciente (Azure Speech + Azure Language + librosa)")
    cfg = pac["monitoramento"].get("audio")
    if not cfg:
        print("[INFO] paciente sem audio associado"); return
    print(f"Arquivo: {cfg['arquivo']}  (fonte: {cfg.get('fonte')})")
    print(f"Azure Speech em modo {'SIMULADO' if audio.speech.mock_mode else 'REAL'} | "
          f"Azure Language em modo {'SIMULADO' if audio.language.mock_mode else 'REAL'}")
    t0 = time.time()
    laudo = audio.analisar(cfg["arquivo"], pac["paciente_id"], contem_fala=cfg.get("contem_fala", True))
    print(f"\nTranscricao ({time.time()-t0:.1f}s): \"{laudo.transcricao}\"")
    print(f"Sentimento: {laudo.sentimento} {laudo.scores_sentimento}")
    print(f"Termos criticos: {[t['termo'] for t in laudo.termos_criticos] or 'nenhum'}")
    if laudo.prob_dispneia is not None:
        print(f"Prob. dispneia (modelo SPIRA): {laudo.prob_dispneia:.2f}")
    a = laudo.atributos
    print(f"Acustica: f0={a['f0_media']:.0f}Hz pausas={a['proporcao_pausas']:.0%} taxa_fala={a['taxa_fala']:.1f}/s "
          f"eventos_impulsivos={a['eventos_impulsivos']}")
    for i in laudo.indicadores_acusticos:
        print(f"  [{i['severidade']}] {i['indicador']}: {i['evidencia']}")
    for al in laudo.alertas:
        print(f"[ALERTA {al.severidade.upper()}] {al.evidencia[:150]}")


def parte_2_video(video: VideoAnalyzer, pac: dict) -> None:
    secao("2) ANALISE DE VIDEO - YOLOv8n (objetos) + YOLOv8n-pose (postura)")
    cfg = pac["monitoramento"].get("video")
    if not cfg:
        print("[INFO] paciente sem video associado"); return
    print(f"Arquivo: {cfg['arquivo']}  (fonte: {cfg.get('fonte')})")
    t0 = time.time()
    laudo = video.analisar(cfg["arquivo"], pac["paciente_id"])
    s = laudo.serie
    print(f"Frames analisados: {len(s)} em {time.time()-t0:.1f}s | pessoa detectada em {s['detectada'].mean():.0%}")
    print(f"Objetos: {laudo.objetos}")
    if laudo.eventos:
        for e in laudo.eventos:
            print(f"[EVENTO {e.severidade.upper()}] {e.tipo} @ {e.tempo_s:.1f}s - {e.evidencia}")
    else:
        print("Nenhum desvio detectado - padrao de movimentacao normal.")
    print(f"Relatorio PDF: {laudo.pdf}")
    print(f"Frames anotados: {laudo.figuras}")


def parte_3_fusao(graph, pac: dict) -> dict:
    secao("3) FUSAO MULTIMODAL (LangGraph) - vitais + prescricoes + audio + video -> score de risco")
    print("Fluxo: receber_paciente -> coletar_vitais -> analisar_prescricoes -> analisar_audio -> "
          "analisar_video -> consolidar_anomalias -> [score>=30 ou critico?] emitir_alerta -> gerar_relatorio")
    print(f"Cenario: {pac['monitoramento'].get('cenario')}")
    t0 = time.time()
    st = graph.invoke({"paciente_id": pac["paciente_id"]})
    df = st["vitais_df"]
    print(f"\nVitais: {len(df)} amostras, {int(df['anomalia'].sum())} pontos anomalos "
          f"(fonte: {pac['monitoramento'].get('sinais_vitais', {}).get('fonte')})")
    print(f"Alertas consolidados ({len(st['alertas'])}):")
    for a in st["alertas"]:
        print(f"  - [{a.modalidade}/{a.severidade}] {a.evidencia[:130]}")
    print(f"\nScore de risco: {st['score_risco']} -> nivel {st['nivel_risco'].upper()} ({time.time()-t0:.1f}s)")
    return st


def parte_4_alerta(st: dict) -> None:
    secao("4) RESPOSTA - alerta a equipe medica, guardrails e auditoria")
    if st.get("mensagem_alerta"):
        print(st["mensagem_alerta"])
    else:
        print("[OK] Nenhum alerta emitido - paciente dentro do padrao. Relatorio gerado mesmo assim.")
    print(f"\nRelatorio JSON: {st['relatorio_path']}")
    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, encoding="utf-8") as f:
            log = json.load(f)
        print(f"Auditoria: {len(log)} registros em {AUDIT_LOG_PATH}. Ultimos 3:")
        for e in log[-3:]:
            print(f"  {e['timestamp'][:19]} | {e['prompt'][:60]} | {json.dumps(e['metadata'], ensure_ascii=False)[:90]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paciente", default="PAC-003")
    ap.add_argument("--rapido", action="store_true", help="pula as partes 1 e 2 (so a fusao)")
    args = ap.parse_args()

    pac = get_patient(args.paciente)
    if not pac or "monitoramento" not in pac:
        raise SystemExit(f"Paciente {args.paciente} sem dados de monitoramento.")
    secao(f"MONITORAMENTO MULTIMODAL - {pac['paciente_id']} ({pac['idade']} anos) - {pac['diagnostico_principal']}")

    logger = AuditLogger(AUDIT_LOG_PATH)
    audio = AudioAnalyzer(audit_logger=logger)
    video = VideoAnalyzer(audit_logger=logger)
    if not args.rapido:
        parte_1_audio(audio, pac)
        parte_2_video(video, pac)
    st = parte_3_fusao(build_monitoring_graph(audio, video, logger), pac)
    parte_4_alerta(st)


if __name__ == "__main__":
    main()
