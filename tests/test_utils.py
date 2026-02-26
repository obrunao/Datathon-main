"""Testes para o modulo de utilidades."""
import pytest
from src.utils import (
    explain_prediction,
    risk_score,
    intervention_suggestion,
)


class TestExplainPrediction:
    def test_gap_trigger(self):
        features = {"idade": 18, "fase": 5, "inde": 7.0}
        reasons = explain_prediction(features)
        assert any("acima do esperado" in r for r in reasons)

    def test_low_inde_trigger(self):
        features = {"idade": 14, "fase": 7, "inde": 4.0}
        reasons = explain_prediction(features)
        assert any("INDE" in r for r in reasons)

    def test_low_iaa_trigger(self):
        features = {"idade": 14, "fase": 7, "inde": 7.0, "iaa": 3.0}
        reasons = explain_prediction(features)
        assert any("IAA" in r for r in reasons)

    def test_low_ieg_trigger(self):
        features = {"idade": 14, "fase": 7, "inde": 7.0, "ieg": 3.0}
        reasons = explain_prediction(features)
        assert any("IEG" in r for r in reasons)

    def test_no_risk(self):
        features = {"idade": 14, "fase": 7, "inde": 8.0, "iaa": 8.0, "ieg": 8.0}
        reasons = explain_prediction(features)
        assert any("esperado" in r for r in reasons)

    def test_returns_list(self):
        features = {"idade": 14, "fase": 7, "inde": 7.0}
        result = explain_prediction(features)
        assert isinstance(result, list)
        assert len(result) > 0


class TestRiskScore:
    def test_alto(self):
        features = {"idade": 20, "fase": 5, "inde": 3.0}
        assert risk_score(features) == "Alto"

    def test_medio(self):
        # gap = 14 - (7+6) = 1 > 0 => score += 2, inde ok => Medio
        features = {"idade": 14, "fase": 7, "inde": 8.0}
        assert risk_score(features) == "Medio"

    def test_baixo(self):
        # gap = 12 - (7+6) = -1 <= 0 => score 0, inde ok => Baixo
        features = {"idade": 12, "fase": 7, "inde": 8.0}
        assert risk_score(features) == "Baixo"

    def test_returns_string(self):
        features = {"idade": 14, "fase": 7, "inde": 7.0}
        assert isinstance(risk_score(features), str)


class TestInterventionSuggestion:
    def test_alto(self):
        result = intervention_suggestion("Alto")
        assert "intensivo" in result

    def test_medio(self):
        result = intervention_suggestion("Medio")
        assert "continuo" in result

    def test_baixo(self):
        result = intervention_suggestion("Baixo")
        assert "regular" in result

    def test_unknown(self):
        result = intervention_suggestion("Desconhecido")
        assert "regular" in result
