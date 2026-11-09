"""Extracao de atributos acusticos (librosa) e heuristica de fadiga/dispneia.

Indicadores usados (literatura de analise vocal clinica, aproximados):
- f0 media/desvio, jitter (variacao ciclo a ciclo de f0) e shimmer (de amplitude)
- taxa de fala aproximada (picos de energia por segundo) e proporcao de pausas
- energia RMS media e taxa de cruzamento por zero (ruido respiratorio)
- eventos de tosse/respiracao ofegante: picos curtos de alta energia e banda larga
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np

SR = 16000


@dataclass
class AtributosAcusticos:
    duracao_s: float
    f0_media: float
    f0_desvio: float
    jitter: float
    shimmer: float
    taxa_fala: float          # silabas aprox./s
    proporcao_pausas: float   # 0-1
    energia_rms: float
    zcr_media: float
    eventos_impulsivos: int   # tosse/ofego
    fracao_vozeada: float
    fracao_ativa: float = 0.0  # frames com energia acima do piso de silencio

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def carregar_audio(path: str, sr: int = SR) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def extrair_atributos(y: np.ndarray, sr: int = SR) -> AtributosAcusticos:
    dur = len(y) / sr
    if dur < 0.2:
        return AtributosAcusticos(dur, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0)

    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024)
    f0v = f0[~np.isnan(f0)]
    f0_media = float(np.mean(f0v)) if len(f0v) else 0.0
    f0_desvio = float(np.std(f0v)) if len(f0v) else 0.0
    jitter = float(np.mean(np.abs(np.diff(f0v))) / f0_media) if len(f0v) > 2 and f0_media else 0.0
    fracao_vozeada = float(np.mean(voiced)) if len(voiced) else 0.0

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_v = rms[rms > np.percentile(rms, 30)]
    shimmer = float(np.mean(np.abs(np.diff(rms_v))) / np.mean(rms_v)) if len(rms_v) > 2 else 0.0

    # pausa = energia abaixo de 10% do nivel tipico de fala (p90), com piso absoluto
    limiar = max(0.1 * np.percentile(rms, 90), 1e-4)
    pausas = float(np.mean(rms <= limiar))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256).astype(np.float64)
    if np.max(onset_env) > 0:
        picos = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=10, post_avg=10,
                                       delta=float(0.3 * np.std(onset_env) + 1e-6), wait=4)
        taxa_fala = len(picos) / dur
    else:
        taxa_fala = 0.0

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)))

    # eventos impulsivos (tosse/ofego): burst >= 3x a mediana dos frames ativos com subida rapida (~50 ms)
    ativo = rms > max(1e-4, 0.01 * np.max(rms))
    fracao_ativa = float(np.mean(ativo))
    if ativo.sum() >= 5:
        ref = np.median(rms[ativo])
        rise = rms[3:] / (rms[:-3] + 1e-6)
        burst = (rms[3:] > 3 * ref) & (rise > 3)
        eventos = int(np.sum(np.diff(burst.astype(int)) == 1))
    else:
        eventos = 0

    return AtributosAcusticos(dur, f0_media, f0_desvio, jitter, shimmer, taxa_fala, pausas,
                              float(np.mean(rms)), zcr, eventos, fracao_vozeada, fracao_ativa)


def avaliar_indicadores(a: AtributosAcusticos) -> list[dict]:
    """Regras heuristicas -> lista de {indicador, severidade, evidencia}."""
    achados = []
    if a.duracao_s >= 2 and a.proporcao_pausas > 0.45 and a.fracao_vozeada > 0.25:
        achados.append({"indicador": "pausas_excessivas", "severidade": "media",
                        "evidencia": f"{a.proporcao_pausas:.0%} do tempo em pausa (possivel fadiga/dispneia)"})
    if a.fracao_vozeada > 0.25 and a.taxa_fala < 2.0 and a.duracao_s >= 3:
        achados.append({"indicador": "fala_lenta", "severidade": "media",
                        "evidencia": f"taxa de fala {a.taxa_fala:.1f} picos/s (abaixo de 2.0)"})
    if a.jitter > 0.04 and a.fracao_vozeada > 0.2:
        achados.append({"indicador": "instabilidade_vocal", "severidade": "alta",
                        "evidencia": f"jitter {a.jitter:.3f} (possivel disartria/fadiga vocal)"})
    if a.shimmer > 0.35 and a.fracao_vozeada > 0.2:
        achados.append({"indicador": "tremor_amplitude", "severidade": "media",
                        "evidencia": f"shimmer {a.shimmer:.2f}"})
    if a.eventos_impulsivos >= 1 and a.fracao_vozeada < 0.5:
        achados.append({"indicador": "tosse_ou_ofego", "severidade": "alta" if a.eventos_impulsivos >= 2 else "media",
                        "evidencia": f"{a.eventos_impulsivos} evento(s) impulsivo(s) (tosse/respiracao ofegante)"})
    elif a.fracao_vozeada < 0.1 and a.fracao_ativa > 0.5 and a.duracao_s >= 2:
        achados.append({"indicador": "respiracao_ruidosa", "severidade": "media",
                        "evidencia": f"energia sustentada sem voz em {a.fracao_ativa:.0%} do tempo (respiracao audivel)"})
    return achados
