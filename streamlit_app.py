import streamlit as st
import joblib
import matplotlib.pyplot as plt
import time

from src.utils import (
    predict_risk,
    explain_prediction,
    risk_score,
    intervention_suggestion,
    prediction_confidence
)

# ======================================================
# CONFIGURAÇÃO GLOBAL
# ======================================================
st.set_page_config(
    page_title="Sistema de Apoio à Decisão Educacional",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# ESTILO VISUAL MODERNO
# ======================================================
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #0b1220, #0e1628);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    max-width: 1500px;
}

.card {
    background: linear-gradient(180deg, #111a2e, #0d1526);
    padding: 1.2rem;
    border-radius: 16px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.04);
}

h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.02em;
}

.small {
    color: #9aa4b2;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# MODELO
# ======================================================
model = joblib.load("app/model/model.joblib")

# ======================================================
# CABEÇALHO
# ======================================================
st.markdown("""
<h1 style="text-align:center;">Sistema de Apoio à Decisão Educacional</h1>
<p style="text-align:center; color:#9aa4b2;">
Avaliação preditiva de risco de defasagem escolar
</p>
""", unsafe_allow_html=True)

st.divider()

# ======================================================
# LAYOUT HORIZONTAL
# ======================================================
col_input, col_result, col_analysis, col_profile = st.columns([1.2, 1.3, 1.8, 2.2])

# ------------------------------------------------------
# INPUTS
# ------------------------------------------------------
with col_input:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dados do aluno")

    idade = st.number_input("Idade", 0, 100, 18)
    fase = st.number_input("Fase escolar", 0, 20, 7)
    inde = st.number_input("INDE", 0.0, 10.0, 6.0, step=0.1)

    avaliar = st.button("Executar avaliação", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# EXECUÇÃO
# ======================================================
if avaliar:
    with st.status("Executando análise preditiva...", expanded=False) as status:
        time.sleep(0.6)

        resultado = predict_risk(model, idade, fase, inde)
        confianca = prediction_confidence(model, idade, fase, inde)
        nivel = risk_score(idade, fase, inde)
        explicacao = explain_prediction(idade, fase, inde)
        acao = intervention_suggestion(nivel)

        time.sleep(0.4)
        status.update(label="Análise concluída", state="complete")

    score = int(confianca)

    # --------------------------------------------------
    # RESULTADO
    # --------------------------------------------------
    with col_result:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Resultado")

        st.markdown(f"### {score} / 100")
        st.progress(score / 100)

        if resultado == 1:
            st.error("Em risco de defasagem")
        else:
            st.success("Sem risco de defasagem")

        st.write(f"Nível de risco: {nivel}")
        st.write(f"Confiança do modelo: {confianca}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_analysis:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Análise")

        tab1, tab2, tab3 = st.tabs(["Resumo", "Justificativa", "Simulação"])

        with tab1:
            st.write(f"Nível estimado: {nivel}")
            st.progress(score / 100)

        with tab2:
            for r in explicacao:
                st.write(f"- {r}")
            st.divider()
            st.write("Ação recomendada:")
            st.write(acao)

        with tab3:
            inde_sim = st.slider("Novo INDE", 0.0, 10.0, inde, 0.1)
            resultado_sim = predict_risk(model, idade, fase, inde_sim)
            st.write(
                "Cenário simulado:",
                "Em risco" if resultado_sim else "Sem risco"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # PERFIL DO ALUNO (GRÁFICO VERTICAL)
    # --------------------------------------------------
    with col_profile:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Perfil do aluno")

        labels = ["Idade", "Fase", "INDE"]
        aluno_vals = [idade, fase, inde]
        ref_vals = [fase + 6, fase, 7.0]

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(labels, aluno_vals, color=["#238636", "#1f6feb", "#da3633"])

        for ref in ref_vals:
            ax.axhline(
        y=ref,
        linestyle="dashed",
        color="#9aa4b2",
        linewidth=1
    )

        ax.set_facecolor("#0b1220")
        fig.patch.set_facecolor("#0b1220")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(colors="#c9d1d9")

        st.pyplot(fig, use_container_width=True)
        st.caption("Linha tracejada indica valor esperado para o perfil escolar.")

        st.markdown("</div>", unsafe_allow_html=True)
