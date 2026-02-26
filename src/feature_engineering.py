"""Feature engineering: criacao de features derivadas e encoding de categoricas."""
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("passos_magicos.feature_engineering")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer sklearn para feature engineering.

    Pode ser colocado dentro de um Pipeline e salvo com joblib,
    garantindo que as mesmas transformacoes sao aplicadas no treinamento e predicao.
    """

    PEDRA_ORDER = {
        "quartzo": 1, "agata": 2, "ágata": 2,
        "ametista": 3, "topazio": 4, "topázio": 4,
    }

    def __init__(self):
        self.feature_names_ = None
        self.column_order_ = None

    def fit(self, X, y=None):
        # Transforma uma vez para capturar a ordem das colunas
        result = self._do_transform(X)
        self.column_order_ = list(result.columns)
        return self

    def _do_transform(self, X):
        """Logica interna de transformacao."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Gap idade-fase (idade esperada = fase + 6)
        if "idade" in df.columns and "fase" in df.columns:
            df["gap_idade_fase"] = df["idade"] - (df["fase"] + 6)
            df["idade_acima_esperado"] = (df["gap_idade_fase"] > 0).astype(int)

        # Media academica (matematica + portugues)
        if "matem" in df.columns and "portug" in df.columns:
            df["media_academica"] = df[["matem", "portug"]].mean(axis=1)

        # INDE baixo (indicador binario)
        if "inde" in df.columns:
            df["inde_baixo"] = (df["inde"] < 6).astype(int)

        # Anos no programa
        if "ano_ingresso" in df.columns:
            df["anos_no_programa"] = 2022 - df["ano_ingresso"]
            df = df.drop(columns=["ano_ingresso"], errors="ignore")

        # Interacao gap x inde
        if "gap_idade_fase" in df.columns and "inde" in df.columns:
            df["gap_x_inde"] = df["gap_idade_fase"] * df["inde"]

        # Encoding ordinal de Pedra
        if "pedra_22" in df.columns:
            df["pedra_22"] = (
                df["pedra_22"]
                .astype(str).str.lower().str.strip()
                .map(self.PEDRA_ORDER)
                .fillna(0)
                .astype(int)
            )

        # Encoding binario de genero
        if "genero" in df.columns:
            df["genero"] = df["genero"].map({"M": 1, "F": 0}).fillna(0).astype(int)

        return df

    def transform(self, X):
        df = self._do_transform(X)

        # Reordena colunas para manter consistencia entre treino e predicao
        if self.column_order_ is not None:
            # Usa apenas colunas presentes, na ordem do treino
            cols = [c for c in self.column_order_ if c in df.columns]
            # Adiciona colunas novas que nao estavam no treino
            extra = [c for c in df.columns if c not in self.column_order_]
            df = df[cols + extra]

        self.feature_names_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper funcional para compatibilidade."""
    eng = FeatureEngineer()
    return eng.fit_transform(df)
