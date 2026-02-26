"""Testes para o modulo de avaliacao."""
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.evaluate import evaluate_model, format_report


@pytest.fixture
def trained_model():
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X[:20], y[:20]


class TestEvaluateModel:
    def test_returns_dict(self, trained_model):
        model, X_test, y_test = trained_model
        result = evaluate_model(model, X_test, y_test)
        assert isinstance(result, dict)

    def test_has_expected_keys(self, trained_model):
        model, X_test, y_test = trained_model
        result = evaluate_model(model, X_test, y_test)
        expected_keys = {"classification_report", "auc", "f1", "precision", "recall", "confusion_matrix"}
        assert expected_keys.issubset(set(result.keys()))

    def test_auc_between_0_and_1(self, trained_model):
        model, X_test, y_test = trained_model
        result = evaluate_model(model, X_test, y_test)
        assert 0 <= result["auc"] <= 1

    def test_f1_between_0_and_1(self, trained_model):
        model, X_test, y_test = trained_model
        result = evaluate_model(model, X_test, y_test)
        assert 0 <= result["f1"] <= 1

    def test_confusion_matrix_shape(self, trained_model):
        model, X_test, y_test = trained_model
        result = evaluate_model(model, X_test, y_test)
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


class TestFormatReport:
    def test_returns_string(self):
        metrics = {"auc": 0.95, "f1": 0.90, "precision": 0.88, "recall": 0.92}
        result = format_report(metrics)
        assert isinstance(result, str)

    def test_contains_all_metrics(self):
        metrics = {"auc": 0.95, "f1": 0.90, "precision": 0.88, "recall": 0.92}
        result = format_report(metrics)
        assert "AUC" in result
        assert "F1" in result
        assert "Precision" in result
        assert "Recall" in result
