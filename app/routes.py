from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd

router = APIRouter()

model = joblib.load("app/model/model.joblib")


# 🔹 Definição clara das features esperadas
class AlunoInput(BaseModel):
    idade: int
    fase: int
    inde: float


@router.post("/predict")
def predict(data: AlunoInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"risco_defasagem": int(prediction)}
