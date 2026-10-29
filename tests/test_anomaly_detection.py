import pandas as pd

from src.anomaly_detection.alerts import Alerta, nivel_risco, score_risco
from src.anomaly_detection.movement import analisar_movimento, gerar_trajetoria_sintetica
from src.anomaly_detection.prescriptions import analisar_prescricoes, dose_em_mg
from src.anomaly_detection.vital_signs import (
    detectar_vitais,
    gerar_serie_sintetica,
    vitais_para_alertas,
)


def test_vitais_detecta_anomalias_injetadas():
    df = detectar_vitais(gerar_serie_sintetica("PAC-001"))
    real, det = df["anomalia_real"], df["anomalia"]
    recall = (real & det).sum() / real.sum()
    falsos = (~real & det).sum() / (~real).sum()
    assert recall >= 0.7
    assert falsos <= 0.05


def test_vitais_serie_normal_gera_poucos_alertas():
    df = detectar_vitais(gerar_serie_sintetica("PAC-002", anomalias=False))
    assert len(vitais_para_alertas("PAC-002", df)) <= 1


def test_vitais_alertas_por_episodio():
    df = detectar_vitais(gerar_serie_sintetica("PAC-001"))
    alertas = vitais_para_alertas("PAC-001", df)
    assert 1 <= len(alertas) <= 4
    assert all(a.modalidade == "vitais" for a in alertas)


def test_dose_em_mg():
    assert dose_em_mg("Losartana 50mg") == 50
    assert dose_em_mg("1 g EV") == 1000
    assert dose_em_mg("sem dose") is None


def test_prescricao_salto_de_dose():
    hist = [
        {"data": "2026-01-01", "medicamento": "Furosemida", "dose": "40mg"},
        {"data": "2026-01-02", "medicamento": "Furosemida", "dose": "160mg"},
    ]
    alertas = analisar_prescricoes("PAC-003", hist)
    assert len(alertas) == 1 and alertas[0].severidade == "critica"


def test_prescricao_estavel_nao_alerta():
    hist = [
        {"data": "2026-01-01", "medicamento": "Losartana", "dose": "50mg"},
        {"data": "2026-01-02", "medicamento": "Losartana", "dose": "50mg"},
    ]
    assert analisar_prescricoes("PAC-001", hist) == []


def test_prescricao_medicamento_novo_fora_da_lista():
    hist = [{"data": "2026-01-01", "medicamento": "Warfarina", "dose": "5mg"}]
    alertas = analisar_prescricoes("PAC-001", hist, medicacoes_atuais=["Losartana 50mg"])
    assert len(alertas) == 1


def test_movimento_detecta_queda_e_imobilidade():
    alertas = analisar_movimento("PAC-004", gerar_trajetoria_sintetica())
    textos = " ".join(a.evidencia for a in alertas)
    assert "queda" in textos.lower()
    assert "Imobilidade" in textos


def test_movimento_normal_sem_queda():
    df = gerar_trajetoria_sintetica(queda_em=None, imobilidade=None)
    assert not [a for a in analisar_movimento("PAC-004", df) if "queda" in a.evidencia.lower()]


def test_alerta_mensagem_tem_disclaimer_e_severidade_valida():
    a = Alerta("PAC-001", "vitais", "alta", "FC elevada")
    assert "validação humana" in a.mensagem()
    try:
        Alerta("PAC-001", "vitais", "xyz", "x")
        assert False
    except ValueError:
        pass


def test_score_risco_cresce_com_modalidades():
    um = [Alerta("P", "vitais", "alta", "x")]
    dois = um + [Alerta("P", "video", "alta", "y")]
    assert score_risco([]) == 0
    assert score_risco(dois) > score_risco(um)
    assert nivel_risco(score_risco(dois)) in {"moderado", "alto", "critico"}
