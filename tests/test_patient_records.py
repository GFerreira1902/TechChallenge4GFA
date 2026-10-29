import json

from src.langchain_pipeline.patient_records import format_patient_context, get_patient, load_patients

SAMPLE_DATA = {
    "pacientes": [
        {
            "paciente_id": "PAC-TEST",
            "nome": "Fictício - Teste",
            "idade": 50,
            "sexo": "F",
            "diagnostico_principal": "Condição de teste",
            "historico": "Sem comorbidades.",
            "medicacoes_atuais": ["Remedio A"],
            "exames_pendentes": ["Exame X"],
            "alertas": ["Alerta de teste"],
        }
    ]
}


def _write_sample_file(tmp_path):
    path = tmp_path / "synthetic_patients.json"
    path.write_text(json.dumps(SAMPLE_DATA, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_load_patients_indexes_by_id(tmp_path):
    path = _write_sample_file(tmp_path)
    patients = load_patients(path)

    assert "PAC-TEST" in patients
    assert patients["PAC-TEST"]["idade"] == 50


def test_get_patient_returns_none_for_unknown_id(tmp_path):
    path = _write_sample_file(tmp_path)
    assert get_patient("PAC-INEXISTENTE", path) is None


def test_format_patient_context_includes_key_fields(tmp_path):
    path = _write_sample_file(tmp_path)
    patient = get_patient("PAC-TEST", path)

    context = format_patient_context(patient)

    assert "PAC-TEST" in context
    assert "Exame X" in context
    assert "Alerta de teste" in context
    assert "Remedio A" in context


def test_format_patient_context_handles_empty_lists(tmp_path):
    data = json.loads(json.dumps(SAMPLE_DATA))
    data["pacientes"][0]["exames_pendentes"] = []
    data["pacientes"][0]["alertas"] = []
    path = tmp_path / "synthetic_patients.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    patient = get_patient("PAC-TEST", str(path))
    context = format_patient_context(patient)

    assert "nenhum" in context.lower()
