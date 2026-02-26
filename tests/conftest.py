"""Fixtures compartilhadas para o test suite."""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_raw_dataframe():
    """Cria DataFrame minimo imitando a estrutura do Excel."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "Idade 22": np.random.randint(10, 22, n),
        "Fase": np.random.randint(0, 8, n),
        "INDE 22": np.random.uniform(2.0, 9.5, n).round(3),
        "Gênero": np.random.choice(["M", "F"], n),
        "Ano ingresso": np.random.randint(2016, 2022, n),
        "Matem": np.random.uniform(3.0, 10.0, n).round(1),
        "Portug": np.random.uniform(3.0, 10.0, n).round(1),
        "IAA": np.random.uniform(2.0, 10.0, n).round(2),
        "IEG": np.random.uniform(2.0, 10.0, n).round(2),
        "IPS": np.random.uniform(2.0, 10.0, n).round(2),
        "IDA": np.random.uniform(2.0, 10.0, n).round(2),
        "IPV": np.random.uniform(2.0, 10.0, n).round(2),
        "IAN": np.random.uniform(2.0, 10.0, n).round(2),
        "Pedra 22": np.random.choice(
            ["Quartzo", "Ágata", "Ametista", "Topázio", None], n
        ),
        "Defas": np.random.choice([0, -1, -2, -3, 1], n, p=[0.3, 0.35, 0.2, 0.1, 0.05]),
        "Cg": np.random.randint(1, 100, n),
        "Cf": np.random.randint(1, 50, n),
        "Ct": np.random.randint(1, 18, n),
        "N° Av": np.random.randint(1, 6, n),
        "RA": range(n),
        "Nome": [f"Aluno_{i}" for i in range(n)],
    })


@pytest.fixture
def sample_preprocessed_df():
    """DataFrame ja preprocessado (colunas renomeadas, sem IDs)."""
    np.random.seed(42)
    n = 40
    return pd.DataFrame({
        "inde": np.random.uniform(2.0, 9.5, n).round(3),
        "genero": np.random.choice(["M", "F"], n),
        "ano_ingresso": np.random.randint(2016, 2022, n),
        "matem": np.random.uniform(3.0, 10.0, n).round(1),
        "portug": np.random.uniform(3.0, 10.0, n).round(1),
        "iaa": np.random.uniform(2.0, 10.0, n).round(2),
        "ieg": np.random.uniform(2.0, 10.0, n).round(2),
        "ips": np.random.uniform(2.0, 10.0, n).round(2),
        "ida": np.random.uniform(2.0, 10.0, n).round(2),
        "ipv": np.random.uniform(2.0, 10.0, n).round(2),
        "pedra_22": np.random.choice(
            ["Quartzo", "Ágata", "Ametista", "Topázio"], n
        ),
        "cg": np.random.randint(1, 100, n),
        "cf": np.random.randint(1, 50, n),
        "ct": np.random.randint(1, 18, n),
        "n_av": np.random.randint(1, 6, n),
    })


@pytest.fixture
def sample_target():
    """Target binario de exemplo."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 40, p=[0.3, 0.7]))
