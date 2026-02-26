"""Testes de integracao para o pipeline de treinamento."""
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.train import build_pipeline, get_candidate_models, select_best_model
from src.preprocessing import preprocess


@pytest.fixture
def training_data(sample_raw_dataframe):
    """Preprocessa o dataframe de exemplo para treino."""
    X, y = preprocess(df=sample_raw_dataframe)
    return X, y


class TestGetCandidateModels:
    def test_returns_dict(self):
        models = get_candidate_models()
        assert isinstance(models, dict)

    def test_has_three_models(self):
        models = get_candidate_models()
        assert len(models) == 3

    def test_model_names(self):
        models = get_candidate_models()
        assert "LogisticRegression" in models
        assert "RandomForest" in models
        assert "GradientBoosting" in models


class TestBuildPipeline:
    def test_returns_pipeline(self):
        models = get_candidate_models()
        pipeline = build_pipeline(models["LogisticRegression"])
        assert isinstance(pipeline, Pipeline)

    def test_has_four_steps(self):
        models = get_candidate_models()
        pipeline = build_pipeline(models["LogisticRegression"])
        assert len(pipeline.steps) == 4

    def test_pipeline_step_names(self):
        models = get_candidate_models()
        pipeline = build_pipeline(models["LogisticRegression"])
        names = [name for name, _ in pipeline.steps]
        assert "feature_engineering" in names
        assert "imputer" in names
        assert "scaler" in names
        assert "model" in names


class TestSelectBestModel:
    def test_returns_tuple(self, training_data):
        X, y = training_data
        result = select_best_model(X, y)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_best_name_is_string(self, training_data):
        X, y = training_data
        best_name, _, _ = select_best_model(X, y)
        assert isinstance(best_name, str)

    def test_cv_results_has_all_models(self, training_data):
        X, y = training_data
        _, _, cv_results = select_best_model(X, y)
        assert "LogisticRegression" in cv_results
        assert "RandomForest" in cv_results
        assert "GradientBoosting" in cv_results


class TestPipelineIntegration:
    def test_fit_predict(self, training_data):
        X, y = training_data
        models = get_candidate_models()
        pipeline = build_pipeline(models["LogisticRegression"])
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, training_data):
        X, y = training_data
        models = get_candidate_models()
        pipeline = build_pipeline(models["LogisticRegression"])
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_single_row(self, training_data):
        X, y = training_data
        models = get_candidate_models()
        pipeline = build_pipeline(models["LogisticRegression"])
        pipeline.fit(X, y)
        single = X.iloc[[0]]
        pred = pipeline.predict(single)
        assert len(pred) == 1
