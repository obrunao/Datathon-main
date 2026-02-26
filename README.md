# Datathon – Passos Mágicos

Projeto de Machine Learning para previsão de risco de defasagem escolar.

## Como rodar
1. Coloque o arquivo Excel em `data/base.xlsx`
2. Instale dependências:
   pip install -r requirements.txt
3. Treine o modelo:
   python src/train.py
4. Suba a API:
   uvicorn app.main:app --reload
