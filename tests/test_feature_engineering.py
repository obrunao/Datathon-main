"""Testes para o modulo de feature engineering."""
import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer, add_features


class TestFeatureEngineer:
    def test_fit_returns_self(self, sample_preprocessed_df):
        fe = FeatureEngineer()
        result = fe.fit(sample_preprocessed_df)
        assert result is fe

    def test_transform_returns_dataframe(self, sample_preprocessed_df):
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_preprocessed_df)
        assert isinstance(result, pd.DataFrame)

    def test_media_academica(self):
        df = pd.DataFrame({"matem": [8.0, 6.0], "portug": [6.0, 4.0]})
        fe = FeatureEngineer()
        result = fe.fit_transform(df)
        assert "media_academica" in result.columns
        np.testing.assert_array_almost_equal(
            result["media_academica"].values, [7.0, 5.0]
        )

    def test_inde_baixo_flag(self):
        df = pd.DataFrame({"inde": [4.0, 7.0, 5.9, 6.0]})
        fe = FeatureEngineer()
        result = fe.fit_transform(df)
        assert "inde_baixo" in result.columns
        assert list(result["inde_baixo"]) == [1, 0, 1, 0]

    def test_pedra_encoding(self):
        df = pd.DataFrame({"pedra_22": ["Quartzo", "Ágata", "Ametista", "Topázio"]})
        fe = FeatureEngineer()
        result = fe.fit_transform(df)
        assert list(result["pedra_22"]) == [1, 2, 3, 4]

    def test_pedra_encoding_null(self):
        df = pd.DataFrame({"pedra_22": ["Quartzo", None, "Ametista"]})
        fe = FeatureEngineer()
        result = fe.fit_transform(df)
        assert result["pedra_22"].iloc[1] == 0

    def test_genero_encoding(self):
        df = pd.DataFrame({"genero": ["M", "F", "M"]})
        fe = FeatureEngineer()
        result = fe.fit_transform(df)
        assert list(result["genero"]) == [1, 0, 1]

    def test_anos_no_programa(self):
        df = pd.DataFrame({"ano_ingresso": [2020, 2018]})
        fe = FeatureEngineer()
        result = fe.fit_transform(df)
        assert "anos_no_programa" in result.columns
        assert list(result["anos_no_programa"]) == [2, 4]
        assert "ano_ingresso" not in result.columns

    def test_missing_columns_graceful(self):
        df = pd.DataFrame({"iaa": [5.0, 6.0], "ieg": [7.0, 8.0]})
        fe = FeatureEngineer()
        result = fe.fit_transform(df)
        assert "gap_idade_fase" not in result.columns
        assert "media_academica" not in result.columns

    def test_get_feature_names_out(self, sample_preprocessed_df):
        fe = FeatureEngineer()
        fe.fit_transform(sample_preprocessed_df)
        names = fe.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) > 0


class TestAddFeatures:
    def test_functional_wrapper(self, sample_preprocessed_df):
        result = add_features(sample_preprocessed_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_preprocessed_df)
