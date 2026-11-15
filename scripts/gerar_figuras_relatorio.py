"""Regenera as figuras usadas em docs/relatorio_tecnico.md a partir dos dados do projeto.

Uso: python scripts/gerar_figuras_relatorio.py [--sem-video]

Saidas em docs/figures/:
  vitais_pac003.png      serie sintetica do PAC-003 com anomalias marcadas (vital_signs.detectar_vitais)
  vitais_bidmc05.png     PhysioNet BIDMC paciente 05 (dados reais) com anomalias marcadas
  spira_roc.png          curva ROC do modelo de dispneia no split de teste do SPIRA
  spira_features.png     distribuicao das 3 medidas prosodicas por classe (SPIRA)
  video_queda_inicio.jpg frame anotado (YOLOv8 + pose) do inicio da queda em UR Fall fall-01
  video_queda_fim.jpg    frame anotado do paciente no chao
  video_normal.jpg       frame anotado de atividade normal em UR Fall adl-05
A captura do dashboard (dashboard_alertas.png) e feita manualmente no navegador.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.anomaly_detection.vital_signs import carregar_bidmc_csv, detectar_vitais, gerar_serie_sintetica  # noqa: E402

FIG = "docs/figures"
SPIRA = "data/samples/audio/spira"
BIDMC = "data/samples/vitals/bidmc_05_Numerics.csv"


def plot_vitais(df: pd.DataFrame, titulo: str, path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    series = [("fc", "FC (bpm)", (50, 120)), ("pa_sist", "PA sistolica (mmHg)", (90, 160)), ("spo2", "SpO2 (%)", (92, 100))]
    for ax, (c, nome, (lo, hi)) in zip(axes, series):
        if df[c].isna().all():
            ax.set_visible(False)
            continue
        ax.plot(df["timestamp"], df[c], lw=1, color="#1f77b4")
        ax.axhspan(lo, hi, color="green", alpha=0.07)
        an = df[df["anomalia"] & df["motivo"].str.contains(c)]
        ax.scatter(an["timestamp"], an[c], color="red", marker="x", s=30, zorder=3, label="anomalia detectada")
        ax.set_ylabel(nome)
    axes[0].legend(loc="upper left")
    axes[0].set_title(titulo)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print("[ok]", path)


def figuras_vitais() -> None:
    df = detectar_vitais(gerar_serie_sintetica("PAC-003", seed=33))
    plot_vitais(df, "PAC-003 (serie sintetica) - episodios de taquicardia e queda de SpO2/PA detectados",
                f"{FIG}/vitais_pac003.png")
    r, p = df["anomalia_real"], df["anomalia"]
    print(f"     recall={(r & p).sum() / r.sum():.2f} falsos_positivos={int((~r & p).sum())}")
    if os.path.exists(BIDMC):
        plot_vitais(detectar_vitais(carregar_bidmc_csv(BIDMC)),
                    "PhysioNet BIDMC paciente 05 (dados reais) - queda breve de SpO2 detectada", f"{FIG}/vitais_bidmc05.png")
    else:
        print("[aviso] BIDMC ausente; rode scripts/download_samples.py --skip-video --skip-audio")


def figuras_spira() -> None:
    """Delega ao script de treino, que ja gera ROC e boxplots em outputs/figures/."""
    if not os.path.exists(f"{SPIRA}/metadata_test.csv"):
        print("[aviso] SPIRA ausente; figuras spira_*.png nao regeneradas (ver data/samples/README.md)")
        return
    import subprocess

    subprocess.run([sys.executable, "scripts/train_dyspnea_model.py"], check=True)
    for n in ("spira_roc.png", "spira_features.png"):
        shutil.copy(f"outputs/figures/{n}", f"{FIG}/{n}")
        print("[ok]", f"{FIG}/{n}")


def figuras_video() -> None:
    from src.video_analysis.video_analyzer import VideoAnalyzer

    va = VideoAnalyzer(gerar_pdf=False)
    pares = [("fall-01", "PAC-004", {"queda_inicio": "video_queda_inicio.jpg", "queda_fim": "video_queda_fim.jpg"}),
             ("adl-05", "PAC-003", {"normal": "video_normal.jpg"})]
    for seq, pid, mapa in pares:
        path = f"data/samples/video/{seq}.mp4"
        if not os.path.exists(path):
            print("[aviso] video ausente:", path)
            continue
        laudo = va.analisar(path, pid)
        for fig in laudo.figuras:
            for chave, destino in mapa.items():
                if chave in os.path.basename(fig):
                    shutil.copy(fig, f"{FIG}/{destino}")
                    print("[ok]", f"{FIG}/{destino}", "<-", os.path.basename(fig))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-video", action="store_true", help="pula YOLO (sem GPU)")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    figuras_vitais()
    figuras_spira()
    if not args.sem_video:
        figuras_video()
