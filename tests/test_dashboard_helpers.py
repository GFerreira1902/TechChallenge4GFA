import json

import pandas as pd

from src.anomaly_detection.alerts import Alerta
from src.anomaly_detection.vital_signs import detectar_vitais, gerar_serie_sintetica
from src.dashboard import helpers as h


def test_pacientes_monitorados():
    pacs = {"A": {"monitoramento": {"x": 1}}, "B": {}, "C": {"monitoramento": None}}
    assert h.pacientes_monitorados(pacs) == ["A"]


def test_figura_vitais_marca_so_variavel_do_motivo():
    df = detectar_vitais(gerar_serie_sintetica("PAC-001"))
    fig = h.figura_vitais(df)
    nomes = [t.name for t in fig.data]
    assert "FC (bpm)" in nomes and "SpO2 (%)" in nomes
    assert any(n.startswith("anomalia") for n in nomes)


def test_figura_vitais_sem_pa():
    df = detectar_vitais(gerar_serie_sintetica("PAC-005", anomalias=False)).assign(pa_sist=float("nan"))
    fig = h.figura_vitais(df)
    assert "PA sist. (mmHg)" not in [t.name for t in fig.data]


def test_figura_postura():
    serie = pd.DataFrame({"tempo_s": [0, 1, 2], "detectada": [True, True, True],
                          "altura_bbox": [300, 200, 100], "inclinacao_tronco_deg": [5, 40, 85]})
    fig = h.figura_postura(serie, [{"tempo_s": 1.0, "tipo": "queda", "severidade": "critica"}])
    assert len(fig.data) == 2


def test_alertas_para_df_ordena_por_severidade():
    al = [Alerta("P", "vitais", "media", "a"), Alerta("P", "video", "critica", "b"), {"paciente": "P", "modalidade": "audio",
          "severidade": "alta", "evidencia": "c", "timestamp": "2026-01-01T00:00:00"}]
    df = h.alertas_para_df(al)
    assert list(df["severidade"]) == ["critica", "alta", "media"]
    assert h.alertas_para_df([]).empty


def test_simular_stream_e_alertas_ate():
    df = gerar_serie_sintetica("PAC-001", n=100)
    fatias = list(h.simular_stream(df, janela=30, passo=20))
    assert [len(f) for f in fatias] == [30, 50, 70, 90, 100]
    al = [Alerta("P", "vitais", "alta", "x", timestamp=str(df["timestamp"].iloc[60])),
          Alerta("P", "audio", "alta", "y")]
    vis = h.alertas_ate(al, df["timestamp"].iloc[30])
    assert [a.modalidade for a in vis] == ["audio"]
    assert len(h.alertas_ate(al, df["timestamp"].iloc[99])) == 2


def test_carregar_auditoria_e_ultimo_relatorio(tmp_path):
    assert h.carregar_auditoria(str(tmp_path / "nada.json")).empty
    log = tmp_path / "audit.json"
    log.write_text(json.dumps([{"timestamp": "2026-01-01T10:00:00", "prompt": "p", "response": "r",
                                "metadata": {"service": "azure_speech", "simulated": False}}]), encoding="utf-8")
    df = h.carregar_auditoria(str(log))
    assert df.iloc[0]["tipo"] == "azure_speech"
    assert h.ultimo_relatorio("PAC-X", str(tmp_path)) is None
    (tmp_path / "monitoramento_PAC-X_20260101_000000.json").write_text('{"score_risco": 1}', encoding="utf-8")
    assert h.ultimo_relatorio("PAC-X", str(tmp_path))["score_risco"] == 1
