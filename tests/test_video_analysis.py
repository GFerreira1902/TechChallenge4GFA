import numpy as np
import pandas as pd
import pytest

from src.video_analysis.object_detector import ROI, Deteccao, pessoas
from src.video_analysis.pose_analyzer import (
    Postura,
    detectar_eventos,
    eventos_para_alertas,
    postura_de_bbox,
    postura_de_keypoints,
    posturas_para_df,
)
from src.video_analysis.video_analyzer import VideoAnalyzer
from src.video_analysis.video_report import anotar_frame, gerar_relatorio_pdf
from src.guardrails.audit_logger import AuditLogger


def _serie(n=60, dt=0.1, queda_em=None, deitado_desde=None, parado=False):
    """Serie sintetica de posturas: pessoa em pe (alt 300, incl 5) andando."""
    rows = []
    for i in range(n):
        t = i * dt
        alt, incl = 300.0, 5.0
        cx = 100 + (0 if parado else i * 5)
        if queda_em is not None and t >= queda_em:
            k = min(1.0, (t - queda_em) / 0.6)
            alt, incl = 300 - 170 * k, 5 + 75 * k
        if deitado_desde is not None and t >= deitado_desde:
            incl = 80.0
        rows.append(Postura(t, True, cx, 200.0, alt, alt / 120, incl))
    return posturas_para_df(rows)


def test_deteccao_propriedades():
    d = Deteccao("person", 0.9, (10, 20, 60, 220))
    assert d.centro == (35, 120) and d.altura == 200 and d.razao_hw == pytest.approx(4.0)
    assert pessoas([d, Deteccao("chair", 0.8, (0, 0, 1, 1))]) == [d]


def test_roi_contem():
    roi = ROI("leito", 0.5, 0.5, 1.0, 1.0)
    assert roi.contem((400, 400), 640, 480) and not roi.contem((100, 100), 640, 480)


def test_postura_de_keypoints_inclinacao():
    kp = np.zeros((17, 2))
    kp[5], kp[6] = (100, 100), (120, 100)      # ombros
    kp[11], kp[12] = (100, 200), (120, 200)    # quadris -> tronco vertical
    p = postura_de_keypoints(kp, np.array([80, 90, 140, 300]), 1.0)
    assert p.inclinacao_tronco_deg == pytest.approx(0, abs=1)
    kp[11], kp[12] = (200, 100), (220, 100)    # quadris ao lado -> horizontal
    assert postura_de_keypoints(kp, np.array([80, 90, 240, 120]), 1.0).inclinacao_tronco_deg > 80


def test_postura_de_bbox_fallback():
    p = postura_de_bbox(Deteccao("person", 0.9, (0, 0, 200, 100)), 2.0)
    assert p.detectada and p.inclinacao_tronco_deg == 90.0


def test_detectar_queda():
    ev = detectar_eventos(_serie(queda_em=3.0), 480, 640)
    tipos = [e.tipo for e in ev]
    assert "queda" in tipos and ev[tipos.index("queda")].severidade == "critica"
    assert ev[tipos.index("queda")].tempo_s == pytest.approx(3.0, abs=1.0)


def test_sem_queda_em_serie_normal():
    assert [e.tipo for e in detectar_eventos(_serie(), 480, 640)] == []


def test_postura_anomala_sustentada_sem_queda():
    ev = detectar_eventos(_serie(n=100, deitado_desde=2.0), 480, 640)
    assert [e.tipo for e in ev] == ["postura_anomala"]


def test_imobilidade():
    ev = detectar_eventos(_serie(n=300, parado=True), 480, 640, imobilidade_s=20)
    assert "imobilidade" in [e.tipo for e in ev]


def test_area_critica():
    df = _serie(n=40)
    roi = ROI("porta", 0.4, 0.0, 1.0, 1.0)  # x > 256 de 640
    ev = detectar_eventos(df, 480, 640, rois=[roi])
    assert any(e.tipo == "area_critica" and "porta" in e.evidencia for e in ev)


def test_sem_pessoa():
    df = posturas_para_df([Postura(i * 0.1, False) for i in range(30)])
    ev = detectar_eventos(df, 480, 640)
    assert ev and ev[0].tipo == "sem_pessoa"


def test_eventos_para_alertas():
    ev = detectar_eventos(_serie(queda_em=3.0), 480, 640)
    al = eventos_para_alertas("PAC-004", ev, "x/fall.mp4")
    assert al[0].modalidade == "video" and "fall.mp4" in al[0].evidencia


def test_anotar_frame_e_pdf(tmp_path, monkeypatch):
    import src.video_analysis.video_report as vr
    monkeypatch.setattr(vr, "REPORTS_DIR", tmp_path / "reports")
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    kp = np.zeros((17, 2)); kp[5] = (50, 30); kp[11] = (50, 80)
    p = Postura(1.0, True, 50, 60, 80, 2, 5, kp)
    ev = detectar_eventos(_serie(queda_em=3.0), 480, 640)
    out = anotar_frame(img, p, ev[0], [Deteccao("person", 0.9, (10, 10, 90, 110))])
    assert out.shape == img.shape and out.any()
    pdf = gerar_relatorio_pdf("PAC-004", "fall.mp4", {"fps": 30, "frames": 160, "duracao_s": 5.3, "largura": 640, "altura": 480},
                              ev, [], {"person": 40})
    assert pdf.endswith(".pdf") and (tmp_path / "reports").exists()


class _DetStub:
    def __init__(self, seq):
        self.seq, self.i = seq, 0

    def detectar(self, img):
        alt = self.seq[min(self.i, len(self.seq) - 1)]; self.i += 1
        return [Deteccao("person", 0.9, (100, 300 - alt, 180, 300)), Deteccao("bed", 0.8, (0, 0, 50, 50))]


class _PoseStub:
    def estimar(self, img, t, bbox=None):
        return postura_de_bbox(bbox, t)


def test_video_analyzer_com_stubs(tmp_path, monkeypatch):
    import cv2
    import src.video_analysis.video_report as vr
    monkeypatch.setattr(vr, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(vr, "FIGURES_DIR", tmp_path / "figs")
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(60):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()
    seq = [200] * 15 + [200 - 150 * min(1, k / 3) for k in range(45)]  # queda apos 1.5 s (passo 1)
    va = VideoAnalyzer(detector=_DetStub(seq), pose=_PoseStub(), audit_logger=AuditLogger(str(tmp_path / "a.json")), passo=1)
    laudo = va.analisar(str(video), "PAC-004")
    assert laudo.info["frames"] == 60 and laudo.objetos["person"] == 60
    assert any(e.tipo == "queda" for e in laudo.eventos)
    assert laudo.pdf and laudo.alertas[0].severidade == "critica" and len(laudo.figuras) >= 1
