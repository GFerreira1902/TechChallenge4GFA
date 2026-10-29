from src.guardrails.safety_rules import contains_direct_prescription, enforce_human_validation


def test_contains_direct_prescription_detects_imperative_language():
    """Frases imperativas de dosagem devem ser sinalizadas como prescrição direta."""
    assert contains_direct_prescription("Tome 500mg de dipirona a cada 6 horas.")
    assert contains_direct_prescription("Inicie 40mg de furosemida imediatamente.")


def test_contains_direct_prescription_false_for_informative_text():
    """Texto meramente informativo/explicativo não deve disparar o guardrail."""
    text = "O protocolo PROT-004 recomenda meta glicemica entre 140-180 mg/dL."
    assert not contains_direct_prescription(text)


def test_enforce_human_validation_adds_disclaimer_when_missing():
    text = "Considere avaliar reposição volêmica conforme o protocolo de sepse."
    resultado, avisos = enforce_human_validation(text)

    assert "validação humana" in resultado.lower()
    assert resultado.startswith(text)


def test_enforce_human_validation_does_not_duplicate_existing_disclaimer():
    text = "Resposta já validada. Requer validação humana antes de qualquer conduta."
    resultado, _ = enforce_human_validation(text)

    assert resultado.count("validação humana") == 1


def test_enforce_human_validation_flags_direct_prescription_warning():
    text = "Tome 1g de paracetamol agora."
    _, avisos = enforce_human_validation(text)

    assert len(avisos) == 1
    assert "prescri" in avisos[0].lower()
