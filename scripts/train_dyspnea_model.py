"""Treina o modelo de dispneia (regressao logistica prosodica) no SPIRA.

Uso: python scripts/train_dyspnea_model.py
Requer data/samples/audio/spira/ (ver data/samples/README.md). Gera
src/audio_analysis/models/dyspnea_logreg.joblib e outputs/figures/spira_*.png.
"""
from __future__ import annotations

import os
import sys
import warnings

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.audio_analysis.dyspnea_model import FEATURES, MODEL_PATH, extrair_prosodia  # noqa: E402

warnings.filterwarnings("ignore")
SPIRA = "data/samples/audio/spira"


def carregar_features() -> pd.DataFrame:
    cache = "outputs/spira_prosodia.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for split in ("train", "eval", "test"):
        meta = pd.read_csv(f"{SPIRA}/metadata_{split}.csv")
        for _, r in meta.iterrows():
            y, sr = librosa.load(f"{SPIRA}/{r['file_path']}", sr=16000, mono=True)
            f = extrair_prosodia(y, sr)
            f.update(split=split, classe=int(r["class"]), nivel=r["nivel_falta_de_ar"], file=r["file_path"])
            rows.append(f)
        print(f"[features] {split}: {len(meta)} arquivos")
    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def main() -> None:
    df = carregar_features()
    y = df["classe"]
    treino, teste = df["split"] != "test", df["split"] == "test"
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))

    cv = cross_val_score(pipe, df[FEATURES], y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
    pipe.fit(df.loc[treino, FEATURES], y[treino])
    p = pipe.predict_proba(df.loc[teste, FEATURES])[:, 1]
    auc = roc_auc_score(y[teste], p)
    print(f"\nAUC teste (split oficial) = {auc:.3f} | AUC CV5 = {cv.mean():.3f} +- {cv.std():.3f}")
    print(classification_report(y[teste], p > 0.5, digits=3))
    coef = dict(zip(FEATURES, pipe[-1].coef_[0].round(2)))
    print("coeficientes padronizados:", coef)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print("[ok] modelo salvo em", MODEL_PATH)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("outputs/figures", exist_ok=True)
        fpr, tpr, _ = roc_curve(y[teste], p)
        plt.figure(figsize=(4.5, 4)); plt.plot(fpr, tpr, label=f"AUC={auc:.2f}"); plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("SPIRA - dispneia (teste)"); plt.legend(); plt.tight_layout()
        plt.savefig("outputs/figures/spira_roc.png", dpi=120)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax, c in zip(axes, ["dur", "bursts_por_s", "pausas"]):
            df.boxplot(column=c, by="classe", ax=ax); ax.set_title(c); ax.set_xlabel("classe (0=controle, 1=paciente)")
        plt.suptitle(""); plt.tight_layout(); plt.savefig("outputs/figures/spira_features.png", dpi=120)
        print("[ok] figuras em outputs/figures/")
    except Exception as exc:
        print("[aviso] figuras nao geradas:", exc)


if __name__ == "__main__":
    main()
