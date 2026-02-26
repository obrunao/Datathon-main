"""Testes para a API FastAPI."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_predict_minimum_fields(self, client):
        payload = {"idade": 14, "fase": 4, "inde": 6.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        payload = {"idade": 14, "fase": 4, "inde": 6.5}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "risco_defasagem" in data
        assert "probabilidade" in data
        assert "nivel_risco" in data
        assert "explicacao" in data
        assert "sugestao_intervencao" in data

    def test_predict_returns_binary(self, client):
        payload = {"idade": 14, "fase": 4, "inde": 6.5}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert data["risco_defasagem"] in [0, 1]

    def test_predict_with_optional_fields(self, client):
        payload = {
            "idade": 14, "fase": 4, "inde": 6.5,
            "iaa": 7.0, "ieg": 6.5, "matem": 8.0, "portug": 7.0,
            "genero": "M",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_invalid_age(self, client):
        payload = {"idade": 2, "fase": 4, "inde": 6.5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_probability_range(self, client):
        payload = {"idade": 14, "fase": 4, "inde": 6.5}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert 0 <= data["probabilidade"] <= 1


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_has_model_info(self, client):
        response = client.get("/metrics")
        data = response.json()
        assert "best_model" in data
        assert "test_metrics" in data
