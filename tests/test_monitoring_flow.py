import json

import pytest

from src.anomaly_detection.alerts import Alerta
from src.fusion import monitoring_flow as mf
from src.guardrails.audit_logger import AuditLogger


class _Laudo:
    def __init__(self, alertas, extra=None):
        self.alertas = alertas
        self._extra = extra or {}

    def to_dict(self):
        return {"alertas": [a.to_dict() for a in self.alertas], **self._extra}


class _AudioStub:
    def __init__(self, alertas=None):
        self.alertas, self.chamadas = alertas or [], []

    def analisar(self, path, pid, contem_fala=True):
        self.chamadas.append((path, contem_fala))
        return _Laudo(self.alertas, {"arquivo": path, "transcricao": "ok", "sentimento": "neutral"})


class _VideoStub:
    def __init__(self, alertas=None):
        import pandas as pd
        self.alertas = alertas or []
        self.eventos = []
        self.serie = pd.DataFrame({"tempo_s": [], "detectada": [], "centro_x": [], "centro_y": [], "altura_bbox": []})
        self.info = {"altura": 480}

    def analisar(self, path, pid):
        l = _Laudo(self.alertas, {"arquivo": path, "eventos": [], "objetos": {}})
        l.serie, l.info, l.eventos = self.serie, self.info, self.eventos
        return l


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "ALERTS_DIR", str(tmp_path / "alerts"))
    monkeypatch.setattr("src.fusion.monitoring_flow.get_patient", lambda pid: _pacientes().get(pid))
    return AuditLogger(str(tmp_path / "audit.json"))


def _pacientes():
    base = {"idade": 70, "sexo": "M", "diagnostico_principal": "Teste", "medicacoes_atuais": ["Furosemida 40mg"]}
    return {
        "PAC-NORMAL": {**base, "paciente_id": "PAC-NORMAL", "monitoramento": {
            "sinais_vitais": {"fonte": "sintetico", "seed": 1, "anomalias": False},
            "historico_prescricoes": [{"data": "2026-01-01", "medicamento": "Furosemida", "dose": "40mg"}],
            "audio": None, "video": None}},
        "PAC-GRAVE": {**base, "paciente_id": "PAC-GRAVE", "monitoramento": {
            "sinais_vitais": {"fonte": "sintetico", "seed": 2, "anomalias": True},
            "historico_prescricoes": [{"data": "2026-01-01", "medicamento": "Furosemida", "dose": "40mg"},
                                      {"data": "2026-01-02", "medicamento": "Furosemida", "dose": "160mg"}],
            "audio": {"arquivo": __file__, "contem_fala": True},
            "video": {"arquivo": __file__}}},
    }


def test_fluxo_normal_nao_alerta(ambiente):
    g = mf.build_monitoring_graph(audit_logger=ambiente)
    st = g.invoke({"paciente_id": "PAC-NORMAL"})
    assert st["nivel_risco"] == "normal" and not st["deve_alertar"]
    assert st.get("mensagem_alerta") is None
    assert st["relatorio"]["alerta_emitido"] is False
    assert json.load(open(st["relatorio_path"], encoding="utf-8"))["paciente_id"] == "PAC-NORMAL"


def test_fluxo_grave_emite_alerta_multimodal(ambiente):
    audio = _AudioStub([Alerta("PAC-GRAVE", "audio", "alta", "dispneia")])
    video = _VideoStub([Alerta("PAC-GRAVE", "video", "critica", "queda")])
    g = mf.build_monitoring_graph(audio, video, ambiente)
    st = g.invoke({"paciente_id": "PAC-GRAVE"})
    mods = {a.modalidade for a in st["alertas"]}
    assert {"vitais", "prescricao", "audio", "video"} <= mods
    assert st["deve_alertar"] and st["nivel_risco"] == "critico"
    assert "validação humana" in st["mensagem_alerta"]
    assert audio.chamadas and audio.chamadas[0][1] is True
    log = json.load(open(ambiente.history_path, encoding="utf-8"))
    assert any(e["metadata"].get("tipo") == "alerta_equipe" for e in log)
    assert st["relatorio"]["audio"]["transcricao"] == "ok" and st["relatorio"]["video"]["arquivo"]


def test_alerta_critico_isolado_e_critico(ambiente):
    video = _VideoStub([Alerta("PAC-NORMAL", "video", "critica", "queda")])
    pac = _pacientes()["PAC-NORMAL"]; pac["monitoramento"]["video"] = {"arquivo": __file__}
    import src.fusion.monitoring_flow as m
    m.get_patient = lambda pid: pac
    st = mf.build_monitoring_graph(None, video, ambiente).invoke({"paciente_id": "PAC-NORMAL"})
    assert st["nivel_risco"] == "critico" and st["deve_alertar"]


def test_audio_ausente_registra_erro_sem_quebrar(ambiente):
    pac = _pacientes()["PAC-NORMAL"]; pac["monitoramento"]["audio"] = {"arquivo": "nao/existe.wav"}
    import src.fusion.monitoring_flow as m
    m.get_patient = lambda pid: pac
    st = mf.build_monitoring_graph(_AudioStub(), None, ambiente).invoke({"paciente_id": "PAC-NORMAL"})
    assert st["laudo_audio"] is None and any("audio ausente" in e for e in st["erros"])


def test_decidir_alerta():
    assert mf.decidir_alerta({"deve_alertar": True}) == "emitir_alerta"
    assert mf.decidir_alerta({"deve_alertar": False}) == "gerar_relatorio"
