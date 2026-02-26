# Datathon Passos Magicos - Sistema de Predicao de Defasagem Escolar

Sistema inteligente que identifica alunos em risco de defasagem escolar usando Machine Learning, desenvolvido para o Datathon da Passos Magicos.

**F1-Score: 0.990 (CV) | 0.996 (Holdout) | 88% cobertura de testes | 82 testes unitarios**

---

## 1. Visao Geral

### O Problema
A Passos Magicos acompanha centenas de alunos com mais de 40 indicadores. Identificar manualmente quais alunos estao ficando para tras e inviavel. Nosso sistema automatiza essa deteccao.

### A Solucao
Pipeline completa de ML que:
- Analisa **860 alunos** com **42 variaveis** (demograficas, academicas e indices de desempenho)
- Compara **3 algoritmos** com cross-validation 5-fold estratificada
- Disponibiliza predicoes via **API REST**, **interface web** e **predicao em lote (CSV)**
- Monitora drift e registra logs de predicao

### Variavel Alvo
`Defas < 0` (binario) — alunos cuja fase atual esta abaixo da fase ideal estao em risco. Distribuicao: **70% em risco**, 30% sem risco.

### Metricas do Modelo Selecionado (LogisticRegression)

| Metrica | Cross-Validation (5-fold) | Teste Holdout (20%) |
|---------|:------------------------:|:-------------------:|
| F1-Score | 0.990 (+/-0.009) | 0.996 |
| AUC-ROC | 0.999 (+/-0.001) | 1.000 |
| Precision | — | 0.992 |
| Recall | — | 1.000 |
| Accuracy | 0.985 | 0.994 |

### Por que LogisticRegression?
| Modelo | F1 (CV) | AUC (CV) |
|--------|:-------:|:--------:|
| **LogisticRegression** | **0.990** | **0.999** |
| GradientBoosting | 0.919 | 0.925 |
| RandomForest | 0.908 | 0.902 |

LogReg venceu com F1 significativamente superior. Alem da performance, ela permite **interpretabilidade** — gestores conseguem entender quais fatores aumentam o risco de cada aluno.

### Stack Tecnologica
| Componente | Tecnologia |
|------------|------------|
| Linguagem | Python 3.11+ |
| ML | scikit-learn, pandas, numpy |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| Serializacao | joblib |
| Testes | pytest (88% cobertura, 82 testes) |
| Container | Docker + Docker Compose |
| Monitoramento | logging + drift z-score |

---

## 2. Estrutura do Projeto

```
datathon/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Ponto de entrada FastAPI (CORS, logging, startup)
│   ├── routes.py                  # Endpoints: /predict, /health, /metrics
│   └── model/
│       ├── model.joblib           # Pipeline treinado serializado
│       ├── pipeline.joblib        # Pipeline treinado (copia)
│       └── metrics.json           # Metricas + stats de referencia para drift
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuracao centralizada (paths, features, hiperparametros)
│   ├── preprocessing.py           # Carga do Excel, limpeza, criacao do target
│   ├── feature_engineering.py     # Transformer sklearn (features derivadas + encoding)
│   ├── train.py                   # Pipeline de treino: compara 3 modelos, salva o melhor
│   ├── evaluate.py                # Calculo de metricas (AUC, F1, Precision, Recall, CM)
│   ├── utils.py                   # Predicao, explicacao, score de risco, intervencao
│   └── monitoring.py              # Logging estruturado + deteccao de drift por z-score
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Fixtures compartilhadas (DataFrames de teste)
│   ├── test_preprocessing.py      # 16 testes
│   ├── test_feature_engineering.py # 11 testes
│   ├── test_evaluate.py           # 7 testes
│   ├── test_model.py              # 12 testes de integracao
│   ├── test_utils.py              # 14 testes
│   ├── test_api.py                # 10 testes da API
│   └── test_monitoring.py         # 12 testes
│
├── data/
│   └── base.xlsx                  # Dataset Passos Magicos (860 alunos, 42 colunas)
│
├── logs/                           # Logs de predicao (gerado automaticamente)
├── streamlit_app.py               # Interface web com dashboard completo
├── Dockerfile                     # Empacotamento Docker
├── docker-compose.yml             # Orquestracao API + Streamlit
├── requirements.txt               # Dependencias Python
└── README.md
```

---

## 3. Como Usar

### Pre-requisitos
- Python 3.11 ou superior
- pip (gerenciador de pacotes)
- Docker e Docker Compose (opcional)

---

### Opcao A: Instalacao Local (Passo a Passo)

**1. Clonar o repositorio**
```bash
git clone https://github.com/obrunao/Datathon-main.git
cd Datathon-main
```

**2. Criar ambiente virtual (recomendado)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**3. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**4. Treinar o modelo**
```bash
python -m src.train
```
Saida esperada:
```
INFO - === Iniciando pipeline de treinamento ===
INFO - Shape: (860, 15), Balanco target: 69.88% positivo
INFO - LogisticRegression: F1=0.990 (+/-0.009), AUC=0.999
INFO - RandomForest: F1=0.908 (+/-0.017), AUC=0.902
INFO - GradientBoosting: F1=0.919 (+/-0.016), AUC=0.925
INFO - Melhor modelo: LogisticRegression
INFO - AUC no teste: 1.000
INFO - Pipeline salvo em app/model/pipeline.joblib
INFO - Metricas salvas em app/model/metrics.json
```

**5. Iniciar a API (terminal 1)**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
A API estara disponivel em: **http://localhost:8000**

**6. Iniciar o Streamlit (terminal 2)**
```bash
python -m streamlit run streamlit_app.py --server.port 8501
```
A interface estara disponivel em: **http://localhost:8501**

---

### Opcao B: Docker (Um Comando)

```bash
docker-compose up --build
```

Isso sobe automaticamente:
- **API:** http://localhost:8000
- **Streamlit:** http://localhost:8501

Para parar:
```bash
docker-compose down
```

---

### Opcao C: Apenas Treinar e Avaliar (Sem Servidor)

```bash
# Instalar dependencias
pip install -r requirements.txt

# Treinar o modelo
python -m src.train

# Ver metricas salvas
python -c "import json; m=json.load(open('app/model/metrics.json')); print(f'F1: {m[\"test_metrics\"][\"f1\"]:.3f}, AUC: {m[\"test_metrics\"][\"auc\"]:.3f}')"
```

---

## 4. Executar Testes

```bash
# Todos os testes (82 testes)
python -m pytest tests/ -v

# Com cobertura de codigo
python -m pytest tests/ --cov=src --cov=app --cov-report=term-missing

# Apenas um modulo especifico
python -m pytest tests/test_api.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_model.py -v
```

Resultado esperado:
```
82 passed
Coverage: 88%
```

---

## 5. Exemplos de Uso da API

### 5.1 Health Check
```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy", "model_loaded": true}
```

### 5.2 Predicao com campos minimos (3 campos)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"idade": 16, "fase": 5, "inde": 5.2}'
```
```json
{
  "risco_defasagem": 1,
  "probabilidade": 0.8732,
  "nivel_risco": "Alto",
  "explicacao": [
    "Idade 5 ano(s) acima do esperado para a fase",
    "INDE abaixo de 6 (desempenho geral baixo)"
  ],
  "sugestao_intervencao": "Encaminhar para acompanhamento pedagogico intensivo"
}
```

### 5.3 Predicao com todos os campos (14 campos)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 14,
    "fase": 4,
    "inde": 7.2,
    "iaa": 6.5,
    "ieg": 7.0,
    "ips": 8.0,
    "ida": 6.8,
    "ipv": 7.5,
    "ian": 5.0,
    "matem": 7.5,
    "portug": 6.8,
    "genero": "M",
    "ano_ingresso": 2020,
    "pedra_22": "Ametista"
  }'
```
```json
{
  "risco_defasagem": 0,
  "probabilidade": 0.0312,
  "nivel_risco": "Baixo",
  "explicacao": ["Indicadores dentro do esperado"],
  "sugestao_intervencao": "Acompanhamento regular"
}
```

### 5.4 Consultar metricas do modelo
```bash
curl http://localhost:8000/metrics
```

### 5.5 Usando Python (requests)
```python
import requests

# Predicao individual
response = requests.post("http://localhost:8000/predict", json={
    "idade": 18, "fase": 5, "inde": 4.5,
    "iaa": 3.2, "ieg": 4.0, "matem": 3.5, "portug": 4.0
})
resultado = response.json()
print(f"Risco: {resultado['nivel_risco']}")
print(f"Probabilidade: {resultado['probabilidade']:.1%}")
print(f"Intervencao: {resultado['sugestao_intervencao']}")
```

### 5.6 Usando PowerShell (Windows)
```powershell
$body = @{
    idade = 16
    fase = 5
    inde = 5.2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

---

## 6. Interface Streamlit

A interface web oferece 3 abas:

### Aba 1: Predicao Individual
- Preencha idade, fase e INDE (obrigatorios)
- Opcionalmente adicione indices na sidebar (IAA, IEG, IPS, IDA, IPV, Matem, Portug, etc.)
- Clique "Executar avaliacao" para ver: resultado, nivel de risco, explicacao e sugestao de intervencao
- **Grafico radar** compara o aluno com a media do dataset
- **Simulacao multi-feature** permite ajustar 6 indices e ver o resultado em tempo real

### Aba 2: Predicao em Lote (CSV)
- Baixe o template CSV de exemplo
- Faca upload de um CSV com multiplos alunos
- Receba resultado de todos com tabela colorida + grafico + download dos resultados

### Aba 3: Historico da Sessao
- Registro automatico de todas as predicoes feitas
- Resumo com metricas e grafico de probabilidades

### Dashboard de Monitoramento (expandivel)
- Metricas do modelo (AUC, F1, Precision, Recall)
- Distribuicao do target (donut chart)
- Comparacao de modelos (barras agrupadas com erro)
- Matriz de confusao (heatmap)
- Metricas por classe
- Feature importance (coeficientes da LogReg)
- Distribuicao dos indices (range plot)

---

## 7. Campos da API

### Campos Obrigatorios
| Campo | Tipo | Range | Descricao |
|-------|------|-------|-----------|
| `idade` | int | 6-30 | Idade do aluno |
| `fase` | int | 0-8 | Fase escolar atual |
| `inde` | float | 0-10 | Indice de Desenvolvimento Educacional |

### Campos Opcionais
| Campo | Tipo | Range | Descricao |
|-------|------|-------|-----------|
| `iaa` | float | 0-10 | Indice de Auto-Aprendizagem |
| `ieg` | float | 0-10 | Indice de Engajamento |
| `ips` | float | 0-10 | Indice Psicossocial |
| `ida` | float | 0-10 | Indice de Desenvolvimento Academico |
| `ipv` | float | 0-10 | Indice de Ponto de Virada |
| `ian` | float | 0-10 | Indice de Adequacao ao Nivel |
| `matem` | float | 0-10 | Nota de Matematica |
| `portug` | float | 0-10 | Nota de Portugues |
| `genero` | string | M/F | Genero do aluno |
| `ano_ingresso` | int | 2010-2025 | Ano de ingresso no programa |
| `pedra_22` | string | — | Classificacao: Quartzo, Agata, Ametista ou Topazio |

Campos nao fornecidos sao preenchidos com NaN e tratados pelo imputer (mediana).

---

## 8. Pipeline de ML (Etapas Detalhadas)

### Etapa 1: Carregamento dos Dados
Leitura do arquivo Excel com 860 alunos e 42 variaveis coletadas pela Passos Magicos.

### Etapa 2: Padronizacao de Colunas
Renomeia colunas para lowercase padronizado (ex: "Idade 22" -> "idade", "INDE 22" -> "inde"). Normaliza caracteres especiais (ordinal indicator U+00BA e degree sign U+00B0).

### Etapa 3: Criacao do Target
Target binario: `Defas < 0` indica que o aluno esta atrasado em relacao a fase ideal. Defasagem = Fase_atual - Fase_ideal; valor negativo significa atraso.

### Etapa 4: Limpeza e Remocao de Leakage
Remove:
- **Identificadores:** RA, Nome (sem poder preditivo)
- **Leakage:** idade + fase (reconstroem Defas perfeitamente), IAN (correlacao -0.983 com target)
- **Nao-preditivas:** avaliadores, recomendacoes, destaques
- **Alta nulidade:** colunas com >50% de valores nulos

### Etapa 5: Feature Engineering (Transformer sklearn)
| Feature Criada | Formula | Justificativa |
|----------------|---------|---------------|
| `media_academica` | (matem + portug) / 2 | Resume desempenho academico |
| `inde_baixo` | 1 se INDE < 6 | Captura limiar critico nao-linear |
| `anos_no_programa` | 2022 - ano_ingresso | Tempo no programa pode influenciar risco |
| `pedra_22` (encoding) | Quartzo=1, Agata=2, Ametista=3, Topazio=4 | Preserva ordem natural |
| `genero` (encoding) | M=1, F=0 | Encoding binario para modelo numerico |

### Etapa 6: Imputacao
Valores ausentes preenchidos com **mediana** (robusta a outliers, melhor que media para notas com zeros).

### Etapa 7: Normalizacao
StandardScaler (media=0, std=1) — melhora convergencia da LogisticRegression.

### Etapa 8: Comparacao de Modelos
3 algoritmos avaliados com cross-validation 5-fold estratificada:
- **LogisticRegression** (class_weight="balanced") — compensa desbalanceamento 70/30
- **RandomForestClassifier** (100 arvores, class_weight="balanced")
- **GradientBoostingClassifier** (100 arvores)

Metrica de selecao: **F1-Score** (equilibra Precision e Recall; melhor que Accuracy para dados desbalanceados).

### Etapa 9: Treinamento Final
Melhor modelo treinado no conjunto de treino completo (80% = 688 alunos).

### Etapa 10: Avaliacao Holdout
Metricas calculadas em dados nunca vistos (20% = 172 alunos). Duas etapas de validacao (CV + holdout) eliminam vies de selecao.

### Etapa 11: Salvamento de Artefatos
- `pipeline.joblib`: Pipeline completa (FeatureEngineer + Imputer + Scaler + Modelo)
- `metrics.json`: Resultados de CV, metricas de teste, confusion matrix e stats de referencia

### Etapa 12: Monitoramento Continuo
- **Logging:** cada predicao registrada com timestamp, input, output e probabilidade
- **Drift detection:** z-score compara features novas com distribuicao do treino (alerta se desvio > 2 sigmas)
- **Dashboard:** Streamlit com 10+ graficos de monitoramento

---

## 9. Justificativas Tecnicas

### Por que F1-Score e nao Accuracy?
Accuracy seria 70% prevendo "em risco" para todos (classe majoritaria). F1 equilibra **Precision** (nao alarmar sem necessidade) e **Recall** (nao perder alunos em risco).

### Por que remover idade, fase e IAN?
Idade + fase permitem reconstruir Defas perfeitamente (leakage direto). IAN tem correlacao -0.983 com o target — zero sobreposicao entre classes. Manter essas features daria metricas artificialmente perfeitas sem generalizacao.

### Por que LogisticRegression e nao ensemble?
F1=0.990 vs 0.919 (GB) e 0.908 (RF). Alem de performance superior, LogReg oferece **coeficientes interpretaveis** — essencial em contexto educacional onde gestores precisam entender o "por que" da classificacao.

### Por que mediana no Imputer?
Mediana e robusta a outliers. Notas academicas podem ter valores extremos (0 para alunos que faltaram) que distorceriam a media.

### Por que class_weight="balanced"?
Dataset com 70/30 de desbalanceamento. O parametro ajusta automaticamente os pesos das classes, evitando que o modelo favoreca a classe majoritaria.
