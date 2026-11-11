"""Baixa as amostras publicas usadas no projeto (ver data/samples/README.md).

Uso: python scripts/download_samples.py [--skip-video] [--skip-audio] [--skip-vitals]
Common Voice e Pexels exigem download manual.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import urllib.request
import zipfile

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


def _get(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dest)


def vitals() -> None:
    for i in ("01", "05", "09"):
        _get(f"https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_{i}_Numerics.csv",
             os.path.join(BASE, "vitals", f"bidmc_{i}_Numerics.csv"))


def audio() -> None:
    api = "https://api.github.com/repos/karolpiczak/ESC-50/contents/audio"
    with urllib.request.urlopen(api) as r:
        names = [x["name"] for x in json.load(r)]
    for n in names:
        if n.endswith(("-23.wav", "-24.wav")):
            _get(f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{n}",
                 os.path.join(BASE, "audio", "esc50", n))


def video() -> None:
    import cv2

    for seq in ("fall-01", "fall-05", "adl-01", "adl-05"):
        mp4 = os.path.join(BASE, "video", f"{seq}.mp4")
        if os.path.exists(mp4):
            continue
        z = os.path.join(BASE, "video", f"{seq}-cam0-rgb.zip")
        _get(f"http://fenix.ur.edu.pl/~mkepski/ds/data/{seq}-cam0-rgb.zip", z)
        folder = z[:-4]
        with zipfile.ZipFile(z) as zf:
            zf.extractall(folder)
        frames = sorted(glob.glob(os.path.join(folder, "**", "*.png"), recursive=True))
        h, w = cv2.imread(frames[0]).shape[:2]
        raw = mp4 + ".raw.mp4"
        out = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for f in frames:
            out.write(cv2.imread(f))
        out.release()
        _para_h264(raw, mp4)
        print(f"[video] {seq}.mp4 ({len(frames)} frames)")


def _para_h264(origem: str, destino: str) -> None:
    """Reencoda para H.264 (reproduzivel no navegador/Streamlit); mantem mp4v se nao houver ffmpeg."""
    try:
        import subprocess

        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", origem, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", destino], check=True)
        os.remove(origem)
    except Exception as exc:
        print("[aviso] sem ffmpeg, mantendo mp4v:", exc)
        os.replace(origem, destino)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--skip-vitals", action="store_true")
    a = p.parse_args()
    if not a.skip_vitals:
        vitals()
    if not a.skip_audio:
        audio()
    if not a.skip_video:
        video()
    print("[ok] amostras em", os.path.abspath(BASE))
