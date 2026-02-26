"""Preprocessamento de dados: carregamento, limpeza e criacao do target."""
import pandas as pd
import numpy as np
import logging

from src.config import (
    RENAME_MAP, LEAKAGE_COLUMNS, ID_COLUMNS,
    NON_FEATURE_COLUMNS, DATA_PATH,
)

logger = logging.getLogger("passos_magicos.preprocessing")


def load_data(path: str = None) -> pd.DataFrame:
    """Carrega o dataset Excel."""
    path = path or str(DATA_PATH)
    df = pd.read_excel(path)
    logger.info(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes das colunas usando o mapa de renomeacao."""
    df = df.copy()
    df = df.rename(columns=RENAME_MAP)
    # Normaliza: lowercase, remove acentos de ordinal/grau, espaço -> _
    df.columns = [
        c.lower().strip()
        .replace(" ", "_")
        .replace("°", "")   # degree sign U+00B0
        .replace("\u00ba", "")  # ordinal indicator U+00BA
        for c in df.columns
    ]
    return df


def create_target(df: pd.DataFrame) -> pd.Series:
    """Cria target binario: 1 se aluno tem defasagem (defas < 0).

    No dataset, Defas = Fase_atual - Fase_ideal.
    Valores negativos indicam que o aluno esta atrasado.
    """
    if "defas" not in df.columns:
        raise ValueError("Coluna 'defas' nao encontrada no dataset.")
    target = (df["defas"] < 0).astype(int)
    logger.info(f"Distribuicao do target: {target.value_counts().to_dict()}")
    return target


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas de ID, leakage, e colunas nao-feature."""
    df = df.copy()

    cols_to_drop = []

    # Identificadores
    cols_to_drop.extend([c for c in ID_COLUMNS if c in df.columns])

    # Colunas com leakage
    cols_to_drop.extend([c for c in LEAKAGE_COLUMNS if c in df.columns])

    # Colunas nao-feature
    cols_to_drop.extend([c for c in NON_FEATURE_COLUMNS if c in df.columns])

    # Colunas com >50% de nulos
    high_null = [
        c for c in df.columns
        if df[c].isnull().mean() > 0.50 and c not in cols_to_drop
    ]
    cols_to_drop.extend(high_null)

    cols_to_drop = list(set(cols_to_drop))
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    logger.info(f"Colunas removidas: {sorted(cols_to_drop)}")
    logger.info(f"Colunas restantes: {sorted(df.columns.tolist())}")
    return df


def preprocess(df: pd.DataFrame = None, path: str = None):
    """Pipeline completa de preprocessamento. Retorna (X, y)."""
    if df is None:
        df = load_data(path)
    df = rename_columns(df)
    y = create_target(df)
    df = clean_data(df)

    mask = y.notna()
    X = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    logger.info(f"Dataset final: {X.shape[0]} linhas, {X.shape[1]} features")
    return X, y
