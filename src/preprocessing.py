
import pandas as pd

def preprocess(df: pd.DataFrame):
    df = df.copy()
    df = df.rename(columns={
        "Idade 22": "idade",
        "Fase": "fase",
        "INDE 22": "inde"
    })
    df = df[["idade", "fase", "inde"]].dropna()
    df["risco_defasagem"] = ((df["idade"] > df["fase"] + 10) | (df["inde"] < 6)).astype(int)
    X = df[["idade", "fase", "inde"]]
    y = df["risco_defasagem"]
    return X, y
