"""Monitoramento: logging estruturado e deteccao de drift."""
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configura logging estruturado para o pipeline de ML."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("passos_magicos")

    # Evita adicionar handlers duplicados
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Handler de arquivo
    fh = logging.FileHandler(
        Path(log_dir) / f"pipeline_{datetime.now():%Y%m%d}.log",
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    # Handler de console
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def compute_reference_stats(X_train: pd.DataFrame) -> dict:
    """Calcula estatisticas de referencia para deteccao de drift."""
    stats = {}
    for col in X_train.select_dtypes(include=[np.number]).columns:
        series = X_train[col].dropna()
        if len(series) == 0:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
        }
    return stats


def detect_drift(new_data: pd.DataFrame, reference_stats: dict,
                 threshold: float = 2.0) -> dict:
    """Deteccao de drift por z-score.

    Retorna alertas para features cuja media desvia mais que
    `threshold` desvios-padrao da referencia.
    """
    alerts = {}
    for col, ref in reference_stats.items():
        if col in new_data.columns and ref["std"] > 0:
            new_mean = float(new_data[col].mean())
            z_score = abs(new_mean - ref["mean"]) / ref["std"]
            if z_score > threshold:
                alerts[col] = {
                    "reference_mean": ref["mean"],
                    "new_mean": new_mean,
                    "z_score": round(z_score, 2),
                }
    return alerts


def log_prediction(logger, input_data: dict, prediction: int,
                   probability: float):
    """Registra cada predicao no log para auditoria."""
    logger.info(json.dumps({
        "event": "prediction",
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
        "timestamp": datetime.now().isoformat(),
    }))
