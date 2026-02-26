"""Testes para o modulo de preprocessamento."""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    rename_columns, create_target, clean_data, preprocess,
)
from src.config import ID_COLUMNS, LEAKAGE_COLUMNS


class TestRenameColumns:
    def test_renames_known_columns(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        assert "inde" in df.columns
        assert "genero" in df.columns
        assert "defas" in df.columns

    def test_lowercases_all_columns(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        for col in df.columns:
            assert col == col.lower()

    def test_returns_copy(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        assert df is not sample_raw_dataframe


class TestCreateTarget:
    def test_binary_output(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        target = create_target(df)
        assert set(target.unique()).issubset({0, 1})

    def test_negative_defas_is_positive_class(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        target = create_target(df)
        mask = df["defas"] < 0
        assert (target[mask] == 1).all()

    def test_zero_defas_is_negative_class(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        target = create_target(df)
        mask = df["defas"] >= 0
        assert (target[mask] == 0).all()

    def test_raises_without_defas_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="defas"):
            create_target(df)

    def test_returns_series(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        target = create_target(df)
        assert isinstance(target, pd.Series)


class TestCleanData:
    def test_drops_id_columns(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        cleaned = clean_data(df)
        for col in ID_COLUMNS:
            assert col not in cleaned.columns

    def test_drops_leakage_columns(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        cleaned = clean_data(df)
        for col in LEAKAGE_COLUMNS:
            assert col not in cleaned.columns

    def test_returns_dataframe(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        cleaned = clean_data(df)
        assert isinstance(cleaned, pd.DataFrame)

    def test_preserves_row_count(self, sample_raw_dataframe):
        df = rename_columns(sample_raw_dataframe)
        cleaned = clean_data(df)
        assert len(cleaned) == len(df)


class TestPreprocess:
    def test_returns_tuple(self, sample_raw_dataframe):
        X, y = preprocess(df=sample_raw_dataframe)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_no_target_in_features(self, sample_raw_dataframe):
        X, y = preprocess(df=sample_raw_dataframe)
        assert "defas" not in X.columns

    def test_same_length(self, sample_raw_dataframe):
        X, y = preprocess(df=sample_raw_dataframe)
        assert len(X) == len(y)

    def test_target_is_binary(self, sample_raw_dataframe):
        X, y = preprocess(df=sample_raw_dataframe)
        assert set(y.unique()).issubset({0, 1})
