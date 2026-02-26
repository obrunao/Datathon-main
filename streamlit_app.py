import streamlit as st
import joblib
import json
import matplotlib.pyplot as plt
import time
from pathlib import Path

from src.utils import (
    predict_risk,
    explain_prediction,
    risk_score,
    intervention_suggestion,
    prediction_confidence,
)

# ======================================================
# CONFIGURACAO GLOBAL
# ======================================================
st.set_page_config(
    page_title="Sistema de Apoio a Decisao Educacional",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# ESTILO VISUAL
# ======================================================
st.markdown("""
<style>
body { background: linear-gradient(120deg, #0b1220, #0e1628); }
.block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; max-width: 1500px; }
.card {
    background: linear-gradient(180deg, #111a2e, #0d1526);
    padding: 1.2rem; border-radius: 16px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.04);
}
h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
.small { color: #9aa4b2; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# MODELO
# ======================================================
model = joblib.load("app/model/model.joblib")

# ======================================================
# CABECALHO
# ======================================================
st.markdown("""
<h1 style="text-align:center;">Sistema de Apoio a Decisao Educacional</h1>
<p style="text-align:center; color:#9aa4b2;">
Avaliacao preditiva de risco de defasagem escolar — Passos Magicos
</p>
""", unsafe_allow_html=True)

st.divider()

# ======================================================
# SIDEBAR: INPUTS OPCIONAIS
# ======================================================
with st.sidebar:
    st.header("Dados complementares")
    st.caption("Preencha para melhorar a precisao da predicao")

    iaa = st.number_input("IAA (Auto-aprendizagem)", 0.0, 10.0, value=None, step=0.1, key="iaa")
    ieg = st.number_input("IEG (Engajamento)", 0.0, 10.0, value=None, step=0.1, key="ieg")
    ips = st.number_input("IPS (Psicossocial)", 0.0, 10.0, value=None, step=0.1, key="ips")
    ida = st.number_input("IDA (Desenvolvimento Academico)", 0.0, 10.0, value=None, step=0.1, key="ida")
    ipv = st.number_input("IPV (Ponto de Virada)", 0.0, 10.0, value=None, step=0.1, key="ipv")
    ian = st.number_input("IAN (Adequacao ao Nivel)", 0.0, 10.0, value=None, step=0.1, key="ian")
    matem = st.number_input("Matematica", 0.0, 10.0, value=None, step=0.1, key="matem")
    portug = st.number_input("Portugues", 0.0, 10.0, value=None, step=0.1, key="portug")
    genero = st.selectbox("Genero", [None, "M", "F"], key="genero")
    pedra_22 = st.selectbox("Pedra 2022", [None, "Quartzo", "Agata", "Ametista", "Topazio"], key="pedra")

# ======================================================
# LAYOUT PRINCIPAL
# ======================================================
col_input, col_result, col_analysis, col_profile = st.columns([1.2, 1.3, 1.8, 2.2])

# INPUTS OBRIGATORIOS
with col_input:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dados do aluno")
    idade = st.number_input("Idade", 6, 30, 14)
    fase = st.number_input("Fase escolar", 0, 8, 4)
    inde = st.number_input("INDE", 0.0, 10.0, 6.0, step=0.1)
    avaliar = st.button("Executar avaliacao", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# EXECUCAO
# ======================================================
if avaliar:
    # Monta dict de features (so inclui opcionais se preenchidos)
    features = {"idade": idade, "fase": fase, "inde": inde}
    for key, val in [("iaa", iaa), ("ieg", ieg), ("ips", ips), ("ida", ida),
                     ("ipv", ipv), ("ian", ian), ("matem", matem), ("portug", portug)]:
        if val is not None:
            features[key] = val
    if genero:
        features["genero"] = genero
    if pedra_22:
        features["pedra_22"] = pedra_22

    with st.status("Executando analise preditiva...", expanded=False) as status:
        time.sleep(0.4)
        resultado = predict_risk(model, features)
        confianca = prediction_confidence(model, features)
        nivel = risk_score(features)
        explicacao = explain_prediction(features)
        acao = intervention_suggestion(nivel)
        status.update(label="Analise concluida", state="complete")

    score = int(confianca)

    # RESULTADO
    with col_result:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Resultado")
        st.markdown(f"### {score} / 100")
        st.progress(min(score / 100, 1.0))
        if resultado == 1:
            st.error("Em risco de defasagem")
        else:
            st.success("Sem risco de defasagem")
        st.write(f"Nivel de risco: **{nivel}**")
        st.write(f"Confianca do modelo: **{confianca}%**")
        st.markdown("</div>", unsafe_allow_html=True)

    # ANALISE
    with col_analysis:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Analise")
        tab1, tab2, tab3 = st.tabs(["Resumo", "Justificativa", "Simulacao"])

        with tab1:
            st.write(f"Nivel estimado: **{nivel}**")
            st.progress(min(score / 100, 1.0))

        with tab2:
            for r in explicacao:
                st.write(f"- {r}")
            st.divider()
            st.write("**Acao recomendada:**")
            st.write(acao)

        with tab3:
            inde_sim = st.slider("Novo INDE", 0.0, 10.0, inde, 0.1)
            features_sim = {**features, "inde": inde_sim}
            resultado_sim = predict_risk(model, features_sim)
            st.write("Cenario simulado:", "**Em risco**" if resultado_sim else "**Sem risco**")

        st.markdown("</div>", unsafe_allow_html=True)

    # PERFIL DO ALUNO
    with col_profile:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Perfil do aluno")

        labels = ["Idade", "Fase", "INDE"]
        aluno_vals = [idade, fase, inde]
        ref_vals = [fase + 6, fase, 7.0]

        # Adiciona indices se preenchidos
        for label, key in [("IAA", "iaa"), ("IEG", "ieg"), ("IPS", "ips"),
                           ("IDA", "ida"), ("IPV", "ipv"), ("IAN", "ian")]:
            if key in features:
                labels.append(label)
                aluno_vals.append(features[key])
                ref_vals.append(7.0)

        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ["#238636", "#1f6feb", "#da3633"] + ["#8b5cf6"] * (len(labels) - 3)
        ax.bar(labels, aluno_vals, color=colors)

        for i, ref in enumerate(ref_vals):
            ax.plot([i - 0.4, i + 0.4], [ref, ref], linestyle="dashed", color="#9aa4b2", linewidth=1)

        ax.set_facecolor("#0b1220")
        fig.patch.set_facecolor("#0b1220")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        st.caption("Linha tracejada indica valor esperado.")
        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# ABA DE MONITORAMENTO
# ======================================================
st.divider()
with st.expander("Dashboard de Monitoramento", expanded=False):
    metrics_path = Path("app/model/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        col_m1, col_m2, col_m3 = st.columns(3)
        test = metrics.get("test_metrics", {})
        col_m1.metric("AUC-ROC", f"{test.get('auc', 0):.3f}")
        col_m2.metric("F1 Score", f"{test.get('f1', 0):.3f}")
        col_m3.metric("Recall", f"{test.get('recall', 0):.3f}")

        st.write(f"**Melhor modelo:** {metrics.get('best_model', 'N/A')}")
        st.write(f"**Dataset:** {metrics.get('dataset_shape', 'N/A')} (linhas, features)")
        st.write(f"**Balanco do target:** {metrics.get('target_balance', 0):.2%} positivo")

        # Resultados de cross-validation
        cv = metrics.get("cv_results", {})
        if cv:
            st.subheader("Comparacao de modelos (Cross-Validation)")
            for name, scores in cv.items():
                st.write(
                    f"- **{name}**: F1={scores['f1_mean']:.3f} "
                    f"(+/-{scores['f1_std']:.3f}), "
                    f"AUC={scores['auc_mean']:.3f}"
                )
    else:
        st.warning("Metricas nao encontradas. Execute o treinamento primeiro.")
