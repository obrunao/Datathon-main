import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Gap idade-fase (feature chave)
    df["gap_idade_fase"] = df["idade"] - (df["fase"] + 6)

    # Indicadores binários
    df["inde_baixo"] = (df["inde"] < 6).astype(int)
    df["idade_acima_esperado"] = (df["gap_idade_fase"] > 0).astype(int)

    return df
