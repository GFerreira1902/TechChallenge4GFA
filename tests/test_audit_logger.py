import json
import os

from src.guardrails.audit_logger import AuditLogger


def test_log_creates_history_file(tmp_path):
    """Garante que o diretório e o arquivo de auditoria são criados no primeiro registro."""
    history_file = tmp_path / "audit" / "history.json"
    logger = AuditLogger(history_path=str(history_file))

    logger.log(prompt="Qual o protocolo para dor torácica?", response="Ver protocolo X.")

    assert os.path.exists(history_file)
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    assert len(history) == 1
    assert history[0]["prompt"] == "Qual o protocolo para dor torácica?"
    assert history[0]["response"] == "Ver protocolo X."


def test_log_appends_multiple_entries(tmp_path):
    """Confirma que registros sucessivos são acumulados, não sobrescritos."""
    history_file = tmp_path / "history.json"
    logger = AuditLogger(history_path=str(history_file))

    logger.log(prompt="p1", response="r1", metadata={"source": "protocolo_a"})
    logger.log(prompt="p2", response="r2", metadata={"source": "protocolo_b"})

    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    assert len(history) == 2
    assert history[1]["metadata"]["source"] == "protocolo_b"
