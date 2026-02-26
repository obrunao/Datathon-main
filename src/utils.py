import pandas as pd

def predict_risk(model, idade: int, fase: int, inde: float) -> int:
    df = pd.DataFrame([{
        "idade": idade,
        "fase": fase,
        "inde": inde
    }])

    return int(model.predict(df)[0])

def explain_prediction(idade, fase, inde):
    reasons = []

    if idade > fase + 10:
        reasons.append("Idade acima do esperado para a fase")

    if inde < 6:
        reasons.append("Desempenho acadêmico abaixo do ideal (INDE baixo)")

    if not reasons:
        reasons.append("Indicadores dentro do esperado")

    return reasons

def risk_score(idade, fase, inde):
    score = 0

    if idade > fase + 10:
        score += 2
    if inde < 6:
        score += 2
    if inde < 4:
        score += 1

    if score >= 3:
        return "Alto"
    elif score == 2:
        return "Médio"
    else:
        return "Baixo"

def intervention_suggestion(risk_level):
    if risk_level == "Alto":
        return "Encaminhar para acompanhamento pedagógico intensivo"
    elif risk_level == "Médio":
        return "Monitoramento contínuo e reforço escolar"
    else:
        return "Acompanhamento regular"

def prediction_confidence(model, idade, fase, inde):
    df = pd.DataFrame([{
        "idade": idade,
        "fase": fase,
        "inde": inde
    }])

    proba = model.predict_proba(df)[0][1]  # probabilidade da classe risco
    return round(proba * 100, 2)
