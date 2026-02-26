"""Rotas da API: /predict, /health, /metrics."""
import json
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd

from src.config import PIPELINE_PATH, MODEL_PATH, METRICS_PATH
from src.monitoring import log_prediction
from src.utils import explain_prediction, risk_score, intervention_suggestion

router = APIRouter()
logger = logging.getLogger("passos_magicos.api")

# Carrega pipeline (tenta novo path, fallback para antigo)
_pipeline_path = PIPELINE_PATH if PIPELINE_PATH.exists() else MODEL_PATH
model = joblib.load(str(_pipeline_path))

# Descobre as features que o pipeline espera (antes do feature engineering)
# Pega os nomes do imputer e reconstroi as features de entrada
_fe_step = model.named_steps["feature_engineering"]
_imputer_step = model.named_steps["imputer"]
_EXPECTED_FEATURES = list(_imputer_step.feature_names_in_)

# Carrega stats de referencia para drift
_ref_stats = {}
if METRICS_PATH.exists():
    with open(str(METRICS_PATH)) as f:
        _metrics = json.load(f)
        _ref_stats = _metrics.get("reference_stats", {})


class AlunoInput(BaseModel):
    """Input do aluno para predicao. Campos obrigatorios sao o minimo necessario."""
    idade: int = Field(..., ge=6, le=30, description="Idade do aluno")
    fase: int = Field(..., ge=0, le=8, description="Fase escolar atual")
    inde: float = Field(..., ge=0, le=10, description="Indice de desenvolvimento")
    iaa: Optional[float] = Field(None, ge=0, le=10)
    ieg: Optional[float] = Field(None, ge=0, le=10)
    ips: Optional[float] = Field(None, ge=0, le=10)
    ida: Optional[float] = Field(None, ge=0, le=10)
    ipv: Optional[float] = Field(None, ge=0, le=10)
    ian: Optional[float] = Field(None, ge=0, le=10)
    matem: Optional[float] = Field(None, ge=0, le=10)
    portug: Optional[float] = Field(None, ge=0, le=10)
    genero: Optional[str] = Field(None, pattern="^[MF]$")
    ano_ingresso: Optional[int] = Field(None, ge=2010, le=2025)
    pedra_22: Optional[str] = None


class PredictionResponse(BaseModel):
    risco_defasagem: int
    probabilidade: float
    nivel_risco: str
    explicacao: list
    sugestao_intervencao: str


@router.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@router.get("/metrics")
def metrics():
    if METRICS_PATH.exists():
        with open(str(METRICS_PATH)) as f:
            return json.load(f)
    raise HTTPException(404, "Arquivo de metricas nao encontrado")


def _build_model_input(features: dict) -> pd.DataFrame:
    """Constroi DataFrame com todas as features que o pipeline espera.

    Features nao fornecidas sao preenchidas com NaN para o imputer tratar.
    Features de leakage (idade, fase, ian) nao sao incluidas.
    """
    # Colunas de entrada do feature engineer (antes da transformacao)
    input_cols = [
        "inde", "iaa", "ieg", "ips", "ida", "ipv",
        "matem", "portug", "genero", "pedra_22",
        "ano_ingresso", "cg", "cf", "ct", "n_av",
    ]
    row = {}
    for col in input_cols:
        row[col] = features.get(col, np.nan)
    return pd.DataFrame([row])


@router.post("/predict", response_model=PredictionResponse)
def predict(data: AlunoInput):
    all_features = {k: v for k, v in data.model_dump().items() if v is not None}

    # Constroi input do modelo (sem features de leakage, NaN para ausentes)
    df = _build_model_input(all_features)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    nivel = risk_score(all_features)
    explicacao = explain_prediction(all_features)
    sugestao = intervention_suggestion(nivel)

    log_prediction(logger, all_features, prediction, probability)

    return PredictionResponse(
        risco_defasagem=prediction,
        probabilidade=round(probability, 4),
        nivel_risco=nivel,
        explicacao=explicacao,
        sugestao_intervencao=sugestao,
    )
