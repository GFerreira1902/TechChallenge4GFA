import numpy as np
import pytest
import soundfile as sf

from src.audio_analysis import acoustic_features as af
from src.audio_analysis.audio_analyzer import AudioAnalyzer
from src.audio_analysis.azure_language_client import AzureLanguageClient, encontrar_termos_criticos
from src.audio_analysis.azure_speech_client import AzureSpeechClient, Transcricao
from src.guardrails.audit_logger import AuditLogger


@pytest.fixture
def sem_chaves(monkeypatch):
    for k in ("AZURE_SPEECH_KEY", "AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def wav_tom(tmp_path):
    """Tom de 150 Hz modulado (vozeado) com pausas: 3 s a 16 kHz."""
    sr = 16000
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    env = ((t % 1.0) < 0.5).astype(float)
    y = 0.3 * np.sin(2 * np.pi * 150 * t) * env
    p = tmp_path / "tom.wav"
    sf.write(p, y, sr)
    return str(p)


@pytest.fixture
def logger(tmp_path):
    return AuditLogger(str(tmp_path / "audit.json"))


def test_extrair_atributos_tom(wav_tom):
    y, sr = af.carregar_audio(wav_tom)
    a = af.extrair_atributos(y, sr)
    assert 2.9 < a.duracao_s < 3.1
    assert 120 < a.f0_media < 180
    assert 0.3 < a.proporcao_pausas < 0.7


def test_extrair_atributos_silencio():
    a = af.extrair_atributos(np.zeros(16000), 16000)
    assert a.f0_media == 0 and a.proporcao_pausas >= 0.9
    assert af.avaliar_indicadores(a) == []


def test_avaliar_indicadores_regras():
    a = af.AtributosAcusticos(10, 120, 10, 0.08, 0.1, 1.0, 0.6, 0.05, 0.1, 0, 0.5)
    nomes = {i["indicador"] for i in af.avaliar_indicadores(a)}
    assert {"pausas_excessivas", "fala_lenta", "instabilidade_vocal"} <= nomes


def test_termos_criticos_sem_acento():
    achados = encontrar_termos_criticos("Estou com FALTA DE AR e muita tontura, doutor.")
    termos = {a["termo"]: a["severidade"] for a in achados}
    assert termos["falta de ar"] == "critica" and termos["tontura"] == "alta"
    assert encontrar_termos_criticos("Tudo bem, me sinto otimo.") == []


def test_speech_mock_mode(sem_chaves, logger, wav_tom):
    cli = AzureSpeechClient(audit_logger=logger)
    assert cli.mock_mode
    t = cli.transcrever(wav_tom)
    assert t.simulado and "SIMULADO" in t.texto
    assert logger.log.__self__ is logger


def test_language_mock_mode(sem_chaves, logger):
    cli = AzureLanguageClient(audit_logger=logger)
    assert cli.mock_mode
    an = cli.analisar("sinto dor no peito e falta de ar")
    assert an.simulado and an.sentimento == "negative" and len(an.termos_criticos) >= 2


def test_language_falha_api_cai_em_heuristica(logger, mocker):
    cli = AzureLanguageClient(api_key="x", endpoint="https://fake.local", audit_logger=logger)
    mocker.patch.object(cli, "_cliente", side_effect=RuntimeError("401"))
    an = cli.analisar("falta de ar")
    assert an.erro == "401" and an.termos_criticos


def test_audio_analyzer_integra_e_gera_alertas(logger, wav_tom, mocker):
    speech = mocker.Mock()
    speech.transcrever.return_value = Transcricao(
        "Doutor, estou com falta de ar e muito cansaco.", "pt-BR", 0.9, 3.0, simulado=False)
    lang = AzureLanguageClient(api_key=None, endpoint=None, audit_logger=logger)
    an = AudioAnalyzer(speech_client=speech, language_client=lang, audit_logger=logger)

    laudo = an.analisar(wav_tom, "PAC-002")
    assert laudo.transcricao.startswith("Doutor")
    assert any(a.severidade == "critica" and a.modalidade == "audio" for a in laudo.alertas)
    assert "falta de ar" in laudo.alertas[0].evidencia
    assert laudo.to_dict()["alertas"][0]["paciente"] == "PAC-002"


def test_audio_analyzer_sem_fala_nao_transcreve(logger, wav_tom, mocker):
    speech = mocker.Mock()
    an = AudioAnalyzer(speech_client=speech, language_client=mocker.Mock(), audit_logger=logger)
    laudo = an.analisar(wav_tom, "PAC-001", contem_fala=False)
    speech.transcrever.assert_not_called()
    assert laudo.transcricao == "" and laudo.termos_criticos == []


def test_dyspnea_model_disponivel_e_prob_valida(wav_tom):
    from src.audio_analysis.dyspnea_model import DyspneaModel, FEATURES, extrair_prosodia
    y, sr = af.carregar_audio(wav_tom)
    f = extrair_prosodia(y, sr)
    assert set(FEATURES) <= set(f)
    m = DyspneaModel()
    assert m.disponivel
    p = m.prever(y, sr)
    assert 0.0 <= p <= 1.0


def test_dyspnea_model_ausente_retorna_none(tmp_path, wav_tom):
    from src.audio_analysis.dyspnea_model import DyspneaModel
    m = DyspneaModel(model_path=str(tmp_path / "nao_existe.joblib"))
    assert not m.disponivel and m.prever(np.zeros(16000), 16000) is None


def test_audio_analyzer_alerta_dispneia(logger, wav_tom, mocker):
    dm = mocker.Mock(); dm.prever.return_value = 0.9
    an = AudioAnalyzer(speech_client=mocker.Mock(), language_client=mocker.Mock(), audit_logger=logger,
                       transcrever=False, dyspnea_model=dm)
    laudo = an.analisar(wav_tom, "PAC-003")
    assert laudo.prob_dispneia == 0.9
    assert any("dificuldade respiratoria" in a.evidencia and a.severidade == "alta" for a in laudo.alertas)
