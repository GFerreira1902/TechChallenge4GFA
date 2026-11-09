"""Cliente Azure Speech to Text (transcricao pt-BR) com mock_mode e auditoria.

Mesmo padrao do llm_client da Fase 3: sem AZURE_SPEECH_KEY o cliente nao quebra,
retorna uma transcricao simulada e marca `simulated=True` na auditoria.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from src.guardrails.audit_logger import AuditLogger

MOCK_TRANSCRIPT = (
    "[MODO SIMULADO - AZURE_SPEECH_KEY AUSENTE] Doutor, estou com muita falta de ar "
    "desde ontem e sinto um cansaco que nao passa."
)


@dataclass
class Transcricao:
    texto: str
    idioma: str
    confianca: Optional[float]
    duracao_s: Optional[float]
    simulado: bool
    erro: Optional[str] = None
    segmentos: list[dict] = field(default_factory=list)


class AzureSpeechClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        language: str = "pt-BR",
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.api_key = api_key or os.environ.get("AZURE_SPEECH_KEY")
        self.region = region or os.environ.get("AZURE_SPEECH_REGION", "brazilsouth")
        self.language = language
        self.mock_mode = not self.api_key
        self.audit_logger = audit_logger or AuditLogger()

    def transcrever(self, audio_path: str) -> Transcricao:
        if self.mock_mode:
            result = Transcricao(MOCK_TRANSCRIPT, self.language, None, None, simulado=True)
        else:
            result = self._transcrever_azure(audio_path)

        self.audit_logger.log(
            prompt=f"[azure_speech] transcrever {os.path.basename(audio_path)}",
            response=result.texto,
            metadata={"service": "azure_speech", "simulated": result.simulado,
                      "region": self.region, "erro": result.erro},
        )
        return result

    def _transcrever_azure(self, audio_path: str) -> Transcricao:
        """Reconhecimento continuo (audios > 15 s) via SDK; concatena os segmentos."""
        tmp = None
        try:
            import azure.cognitiveservices.speech as speechsdk

            audio_path, tmp = _preparar_wav(audio_path)
            cfg = speechsdk.SpeechConfig(subscription=self.api_key, region=self.region)
            cfg.speech_recognition_language = self.language
            cfg.output_format = speechsdk.OutputFormat.Detailed
            audio = speechsdk.AudioConfig(filename=audio_path)
            rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)

            segmentos: list[dict] = []
            erros: list[str] = []
            done = False

            def on_recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                    segmentos.append({
                        "texto": evt.result.text,
                        "offset_s": evt.result.offset / 1e7,
                        "duracao_s": evt.result.duration / 1e7,
                    })

            def on_stop(evt):
                nonlocal done
                done = True

            def on_canceled(evt):
                nonlocal done
                if evt.reason == speechsdk.CancellationReason.Error:
                    erros.append(evt.error_details)
                done = True

            rec.recognized.connect(on_recognized)
            rec.session_stopped.connect(on_stop)
            rec.canceled.connect(on_canceled)
            rec.start_continuous_recognition()
            import time
            t0 = time.time()
            while not done and time.time() - t0 < 300:
                time.sleep(0.2)
            rec.stop_continuous_recognition()
            del rec, audio

            texto = " ".join(s["texto"] for s in segmentos).strip()
            dur = (segmentos[-1]["offset_s"] + segmentos[-1]["duracao_s"]) if segmentos else None
            return Transcricao(texto, self.language, None, dur, simulado=False,
                               erro=erros[0] if erros else None, segmentos=segmentos)
        except Exception as exc:  # nunca derruba o fluxo
            return Transcricao("", self.language, None, None, simulado=False, erro=str(exc))
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass  # SDK pode manter handle aberto no Windows; o SO limpa o temp


def _preparar_wav(audio_path: str) -> tuple[str, Optional[str]]:
    """O SDK exige WAV PCM; converte mp3/ogg/etc. (e reamostra) para WAV 16 kHz mono temporario."""
    import soundfile as sf

    if audio_path.lower().endswith(".wav"):
        try:
            info = sf.info(audio_path)
            if info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16":
                return audio_path, None
        except Exception:
            pass
    import tempfile

    import librosa

    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp, y, 16000, subtype="PCM_16")
    return tmp, tmp
