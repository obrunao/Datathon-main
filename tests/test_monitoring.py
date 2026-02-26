"""Testes para o modulo de monitoramento."""
import pytest
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.monitoring import (
    setup_logging,
    compute_reference_stats,
    detect_drift,
    log_prediction,
)


class TestSetupLogging:
    def test_creates_logger(self, tmp_path):
        logger = setup_logging(str(tmp_path))
        assert isinstance(logger, logging.Logger)
        assert logger.name == "passos_magicos"

    def test_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / "test_logs"
        setup_logging(str(log_dir))
        assert log_dir.exists()

    def test_returns_same_logger(self, tmp_path):
        # Limpa handlers para evitar duplicacao
        logger = logging.getLogger("passos_magicos")
        logger.handlers.clear()
        l1 = setup_logging(str(tmp_path))
        l2 = setup_logging(str(tmp_path))
        assert l1.name == l2.name


class TestComputeReferenceStats:
    def test_returns_dict(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        stats = compute_reference_stats(df)
        assert isinstance(stats, dict)

    def test_has_expected_keys(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        stats = compute_reference_stats(df)
        assert "a" in stats
        for key in ["mean", "std", "min", "max", "q25", "q75"]:
            assert key in stats["a"]

    def test_correct_values(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
        stats = compute_reference_stats(df)
        assert stats["x"]["mean"] == 20.0
        assert stats["x"]["min"] == 10.0
        assert stats["x"]["max"] == 30.0

    def test_ignores_non_numeric(self):
        df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
        stats = compute_reference_stats(df)
        assert "num" in stats
        assert "cat" not in stats


class TestDetectDrift:
    def test_no_drift(self):
        ref = {"x": {"mean": 5.0, "std": 1.0}}
        new_data = pd.DataFrame({"x": [4.8, 5.2, 5.0]})
        alerts = detect_drift(new_data, ref)
        assert len(alerts) == 0

    def test_detects_drift(self):
        ref = {"x": {"mean": 5.0, "std": 1.0}}
        new_data = pd.DataFrame({"x": [15.0, 16.0, 14.0]})
        alerts = detect_drift(new_data, ref)
        assert "x" in alerts
        assert alerts["x"]["z_score"] > 2.0

    def test_returns_dict(self):
        ref = {"x": {"mean": 5.0, "std": 1.0}}
        new_data = pd.DataFrame({"x": [5.0]})
        result = detect_drift(new_data, ref)
        assert isinstance(result, dict)

    def test_custom_threshold(self):
        ref = {"x": {"mean": 5.0, "std": 1.0}}
        new_data = pd.DataFrame({"x": [8.0, 9.0]})
        # Com threshold baixo, deve detectar
        alerts = detect_drift(new_data, ref, threshold=1.0)
        assert "x" in alerts


class TestLogPrediction:
    def test_logs_prediction(self, caplog):
        logger = logging.getLogger("passos_magicos.test")
        with caplog.at_level(logging.INFO, logger="passos_magicos.test"):
            log_prediction(logger, {"idade": 14}, 1, 0.85)
        assert "prediction" in caplog.text
