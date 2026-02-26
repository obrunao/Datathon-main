"""Configuracao centralizada do pipeline de ML."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "base.xlsx"
MODEL_DIR = PROJECT_ROOT / "app" / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
LOG_DIR = PROJECT_ROOT / "logs"

# Mapeamento de colunas do Excel para nomes padronizados
RENAME_MAP = {
    "Idade 22": "idade",
    "Fase": "fase",
    "INDE 22": "inde",
    "Gênero": "genero",
    "Ano ingresso": "ano_ingresso",
    "Matem": "matem",
    "Portug": "portug",
    "IAA": "iaa",
    "IEG": "ieg",
    "IPS": "ips",
    "IDA": "ida",
    "IPV": "ipv",
    "IAN": "ian",
    "Pedra 22": "pedra_22",
    "Cg": "cg",
    "Cf": "cf",
    "Ct": "ct",
    "N\u00ba Av": "n_av",
    "N\u00b0 Av": "n_av",
    "Defas": "defas",
}

# Features numericas que serao usadas
NUMERIC_FEATURES = [
    "idade", "fase", "inde", "iaa", "ieg", "ips",
    "ida", "ipv", "ian", "matem", "portug",
    "cg", "cf", "ct", "n_av",
]

# Features categoricas
CATEGORICAL_FEATURES = ["genero", "pedra_22"]

# Colunas que causam data leakage (nao usar como features)
# idade + fase permitem reconstruir defas perfeitamente
# IAN (Indice de Adequacao ao Nivel) tem correlacao -0.983 com defasagem
LEAKAGE_COLUMNS = ["defas", "fase_ideal", "idade", "fase", "ian"]

# Colunas de identificacao (nao usar como features)
ID_COLUMNS = ["ra", "nome"]

# Colunas nao-feature (dropar)
NON_FEATURE_COLUMNS = [
    "turma", "instituição_de_ensino", "instituicao_de_ensino",
    "ano_nasc", "avaliador1", "avaliador2", "avaliador3", "avaliador4",
    "rec_av1", "rec_av2", "rec_av3", "rec_av4",
    "indicado", "atingiu_pv", "rec_psicologia",
    "destaque_ieg", "destaque_ida", "destaque_ipv",
    "pedra_20", "pedra_21",
]

# Configuracao do modelo
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
