"""Pipeline de treinamento: preprocessamento, feature engineering, selecao de modelo."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

from src.config import (
    MODEL_PATH, PIPELINE_PATH, METRICS_PATH, MODEL_DIR,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS,
)
from src.preprocessing import preprocess
from src.feature_engineering import FeatureEngineer
from src.evaluate import evaluate_model
from src.monitoring import setup_logging, compute_reference_stats

logger = logging.getLogger("passos_magicos.train")


def get_candidate_models() -> dict:
    """Retorna dict de nome -> instancia de modelo para comparacao."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model) -> Pipeline:
    """Constroi Pipeline sklearn: FeatureEngineer -> Imputer -> Scaler -> Modelo."""
    return Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def select_best_model(X_train, y_train) -> tuple:
    """Compara modelos com cross-validation estratificada, retorna o melhor."""
    candidates = get_candidate_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=["f1", "roc_auc", "accuracy"],
            return_train_score=False,
        )
        results[name] = {
            "f1_mean": float(np.mean(scores["test_f1"])),
            "f1_std": float(np.std(scores["test_f1"])),
            "auc_mean": float(np.mean(scores["test_roc_auc"])),
            "auc_std": float(np.std(scores["test_roc_auc"])),
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        }
        logger.info(
            f"{name}: F1={results[name]['f1_mean']:.3f} "
            f"(+/-{results[name]['f1_std']:.3f}), "
            f"AUC={results[name]['auc_mean']:.3f}"
        )

    # Seleciona pelo maior F1 medio
    best_name = max(results, key=lambda k: results[k]["f1_mean"])
    logger.info(f"Melhor modelo: {best_name}")
    return best_name, candidates[best_name], results


def train(data_path: str = None) -> dict:
    """Pipeline completa de treinamento. Retorna dict de metricas."""
    setup_logging()
    logger.info("=== Iniciando pipeline de treinamento ===")

    # 1. Preprocessamento
    X, y = preprocess(path=data_path)
    logger.info(f"Shape: {X.shape}, Balanco target: {y.mean():.2%} positivo")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # 3. Selecao de modelo com cross-validation
    best_name, best_model, cv_results = select_best_model(X_train, y_train)

    # 4. Pipeline final treinada no conjunto de treino completo
    final_pipeline = build_pipeline(best_model)
    final_pipeline.fit(X_train, y_train)

    # 5. Avaliacao no conjunto de teste
    test_metrics = evaluate_model(final_pipeline, X_test, y_test)
    logger.info(f"AUC no teste: {test_metrics['auc']:.3f}")

    # 6. Salvar artefatos
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, str(PIPELINE_PATH))
    joblib.dump(final_pipeline, str(MODEL_PATH))
    logger.info(f"Pipeline salvo em {PIPELINE_PATH}")

    # 7. Estatisticas de referencia para drift
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_fe),
        columns=X_train_fe.columns if hasattr(X_train_fe, "columns") else None,
    )
    ref_stats = compute_reference_stats(X_train_imp)

    # 8. Salvar metricas
    all_metrics = {
        "best_model": best_name,
        "cv_results": cv_results,
        "test_metrics": {
            "auc": test_metrics["auc"],
            "f1": test_metrics["f1"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "classification_report": test_metrics["classification_report"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "reference_stats": ref_stats,
        "dataset_shape": list(X.shape),
        "target_balance": float(y.mean()),
    }
    with open(str(METRICS_PATH), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Metricas salvas em {METRICS_PATH}")

    return all_metrics


if __name__ == "__main__":
    train()
