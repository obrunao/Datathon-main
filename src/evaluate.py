"""Avaliacao do modelo: calculo de metricas e relatorio."""
import logging
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger("passos_magicos.evaluate")


def evaluate_model(model, X_test, y_test) -> dict:
    """Calcula metricas de classificacao no conjunto de teste."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    logger.info(f"AUC: {auc:.3f}")
    logger.info(f"F1: {f1:.3f}")
    logger.info(f"Precision: {prec:.3f}")
    logger.info(f"Recall: {rec:.3f}")
    logger.info(f"Confusion matrix:\n{cm}")

    return {
        "classification_report": report,
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "confusion_matrix": cm.tolist(),
    }


def format_report(metrics: dict) -> str:
    """Formata metricas em string legivel."""
    lines = [
        f"AUC-ROC:   {metrics['auc']:.3f}",
        f"F1 Score:  {metrics['f1']:.3f}",
        f"Precision: {metrics['precision']:.3f}",
        f"Recall:    {metrics['recall']:.3f}",
    ]
    return "\n".join(lines)
