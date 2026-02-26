"""Funcoes utilitarias: predicao, explicacao, score de risco."""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("passos_magicos.utils")

# Colunas de entrada do modelo (sem features de leakage: idade, fase, ian)
_MODEL_INPUT_COLS = [
    "inde", "iaa", "ieg", "ips", "ida", "ipv",
    "matem", "portug", "genero", "pedra_22",
    "ano_ingresso", "cg", "cf", "ct", "n_av",
]


def _build_model_input(features: dict) -> pd.DataFrame:
    """Constroi DataFrame com as features que o pipeline espera.

    Features de leakage (idade, fase, ian) sao excluidas.
    Features nao fornecidas sao preenchidas com NaN para o imputer tratar.
    """
    row = {}
    for col in _MODEL_INPUT_COLS:
        row[col] = features.get(col, np.nan)
    return pd.DataFrame([row])


def predict_risk(model, features: dict) -> int:
    """Prediz risco usando o pipeline completo."""
    df = _build_model_input(features)
    return int(model.predict(df)[0])


def prediction_confidence(model, features: dict) -> float:
    """Retorna probabilidade da classe positiva (risco) em porcentagem."""
    df = _build_model_input(features)
    proba = model.predict_proba(df)[0][1]
    return round(proba * 100, 2)


def explain_prediction(features: dict) -> list:
    """Explicacao baseada em regras sobre o risco do aluno."""
    reasons = []

    idade = features.get("idade", 0)
    fase = features.get("fase", 0)
    inde = features.get("inde", 10)
    iaa = features.get("iaa", 10)
    ieg = features.get("ieg", 10)
    matem = features.get("matem", 10)
    portug = features.get("portug", 10)

    gap = idade - (fase + 6)
    if gap > 0:
        reasons.append(f"Idade {gap} ano(s) acima do esperado para a fase")
    if inde < 6:
        reasons.append("INDE abaixo de 6 (desempenho geral baixo)")
    if iaa < 5:
        reasons.append("IAA baixo (auto-aprendizagem insuficiente)")
    if ieg < 5:
        reasons.append("IEG baixo (engajamento insuficiente)")
    if matem < 5:
        reasons.append("Nota de matematica abaixo de 5")
    if portug < 5:
        reasons.append("Nota de portugues abaixo de 5")

    if not reasons:
        reasons.append("Indicadores dentro do esperado")

    return reasons


def risk_score(features: dict) -> str:
    """Calcula nivel de risco categorico a partir das features."""
    score = 0

    idade = features.get("idade", 0)
    fase = features.get("fase", 0)
    inde = features.get("inde", 10)

    gap = idade - (fase + 6)
    if gap > 2:
        score += 3
    elif gap > 0:
        score += 2

    if inde < 4:
        score += 3
    elif inde < 6:
        score += 2

    if score >= 4:
        return "Alto"
    elif score >= 2:
        return "Medio"
    else:
        return "Baixo"


def intervention_suggestion(risk_level: str) -> str:
    """Sugere intervencao baseada no nivel de risco."""
    suggestions = {
        "Alto": "Encaminhar para acompanhamento pedagogico intensivo",
        "Medio": "Monitoramento continuo e reforco escolar",
        "Baixo": "Acompanhamento regular",
    }
    return suggestions.get(risk_level, "Acompanhamento regular")
