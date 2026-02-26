import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
from pathlib import Path

from src.utils import (
    predict_risk,
    explain_prediction,
    risk_score,
    intervention_suggestion,
    prediction_confidence,
    _build_model_input,
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
# CONSTANTES DE TEMA
# ======================================================
BG_COLOR = "#0b1220"
CARD_BG = "#111a2e"
TEXT_COLOR = "#c9d1d9"
MUTED_COLOR = "#9aa4b2"
GREEN = "#238636"
BLUE = "#1f6feb"
RED = "#da3633"
PURPLE = "#8b5cf6"
ORANGE = "#f0883e"
CYAN = "#58a6ff"
YELLOW = "#d29922"


def _style_ax(fig, ax):
    """Aplica estilo escuro padrao a um eixo matplotlib."""
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)


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
# MODELO E METRICAS
# ======================================================
model = joblib.load("app/model/model.joblib")

metrics_path = Path("app/model/metrics.json")
_metrics_data = {}
if metrics_path.exists():
    with open(metrics_path) as f:
        _metrics_data = json.load(f)

# ======================================================
# SESSION STATE — historico de predicoes
# ======================================================
if "historico" not in st.session_state:
    st.session_state.historico = []

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
# TABS PRINCIPAIS
# ======================================================
tab_pred, tab_lote, tab_hist = st.tabs([
    "Predicao Individual", "Predicao em Lote (CSV)", "Historico da Sessao"
])

# ======================================================
# TAB 1: PREDICAO INDIVIDUAL
# ======================================================
with tab_pred:
    col_input, col_result, col_analysis, col_profile = st.columns([1.2, 1.3, 1.8, 2.2])

    with col_input:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dados do aluno")
        idade = st.number_input("Idade", 6, 30, 14, key="idade_input")
        fase = st.number_input("Fase escolar", 0, 8, 4, key="fase_input")
        inde = st.number_input("INDE", 0.0, 10.0, 6.0, step=0.1, key="inde_input")
        avaliar = st.button("Executar avaliacao", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if avaliar:
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
            time.sleep(0.3)
            resultado = predict_risk(model, features)
            confianca = prediction_confidence(model, features)
            nivel = risk_score(features)
            explicacao = explain_prediction(features)
            acao = intervention_suggestion(nivel)
            status.update(label="Analise concluida", state="complete")

        # Salva no historico
        st.session_state.historico.append({
            "idade": idade, "fase": fase, "inde": inde,
            "risco": resultado, "probabilidade": confianca,
            "nivel": nivel,
        })

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
            tab_r, tab_j, tab_s = st.tabs(["Resumo", "Justificativa", "Simulacao"])

            with tab_r:
                st.write(f"Nivel estimado: **{nivel}**")
                st.progress(min(score / 100, 1.0))

            with tab_j:
                for r in explicacao:
                    st.write(f"- {r}")
                st.divider()
                st.write("**Acao recomendada:**")
                st.write(acao)

            with tab_s:
                st.caption("Ajuste os indices para simular cenarios diferentes")
                sc1, sc2 = st.columns(2)
                with sc1:
                    sim_inde = st.slider("INDE", 0.0, 10.0, inde, 0.1, key="sim_inde")
                    sim_iaa = st.slider("IAA", 0.0, 10.0, float(iaa or 5.0), 0.1, key="sim_iaa")
                    sim_ieg = st.slider("IEG", 0.0, 10.0, float(ieg or 5.0), 0.1, key="sim_ieg")
                with sc2:
                    sim_ips = st.slider("IPS", 0.0, 10.0, float(ips or 5.0), 0.1, key="sim_ips")
                    sim_ida = st.slider("IDA", 0.0, 10.0, float(ida or 5.0), 0.1, key="sim_ida")
                    sim_ipv = st.slider("IPV", 0.0, 10.0, float(ipv or 5.0), 0.1, key="sim_ipv")

                features_sim = {
                    **features,
                    "inde": sim_inde, "iaa": sim_iaa, "ieg": sim_ieg,
                    "ips": sim_ips, "ida": sim_ida, "ipv": sim_ipv,
                }
                resultado_sim = predict_risk(model, features_sim)
                conf_sim = prediction_confidence(model, features_sim)

                if resultado_sim == 1:
                    st.error(f"Cenario simulado: **Em risco** ({conf_sim}%)")
                else:
                    st.success(f"Cenario simulado: **Sem risco** ({100 - conf_sim:.1f}% seguro)")

            st.markdown("</div>", unsafe_allow_html=True)

        # PERFIL DO ALUNO — RADAR CHART
        with col_profile:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Perfil do aluno")

            ref_stats = _metrics_data.get("reference_stats", {})

            radar_labels = ["INDE"]
            aluno_vals = [inde]
            ref_means = [ref_stats.get("inde", {}).get("mean", 7.0)]

            for label, key in [("IAA", "iaa"), ("IEG", "ieg"), ("IPS", "ips"),
                               ("IDA", "ida"), ("IPV", "ipv")]:
                if key in features:
                    radar_labels.append(label)
                    aluno_vals.append(features[key])
                    ref_means.append(ref_stats.get(key, {}).get("mean", 7.0))

            for label, key in [("Matem", "matem"), ("Portug", "portug")]:
                if key in features:
                    radar_labels.append(label)
                    aluno_vals.append(features[key])
                    ref_means.append(ref_stats.get(key, {}).get("mean", 6.0))

            if len(radar_labels) >= 3:
                angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
                aluno_plot = aluno_vals + [aluno_vals[0]]
                ref_plot = ref_means + [ref_means[0]]
                angles += [angles[0]]

                fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                ax.fill(angles, aluno_plot, color=CYAN, alpha=0.25)
                ax.plot(angles, aluno_plot, color=CYAN, linewidth=2, label="Aluno")
                ax.fill(angles, ref_plot, color=ORANGE, alpha=0.1)
                ax.plot(angles, ref_plot, color=ORANGE, linewidth=1.5, linestyle="--", label="Media ref.")

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(radar_labels, color=TEXT_COLOR, fontsize=8)
                ax.set_ylim(0, 10)
                ax.set_yticks([2, 4, 6, 8, 10])
                ax.set_yticklabels(["2", "4", "6", "8", "10"], color=MUTED_COLOR, fontsize=7)
                ax.set_facecolor(BG_COLOR)
                fig.patch.set_facecolor(BG_COLOR)
                ax.spines["polar"].set_color(MUTED_COLOR)
                ax.grid(color=MUTED_COLOR, alpha=0.3)
                ax.legend(loc="upper right", fontsize=7, facecolor=CARD_BG, edgecolor=MUTED_COLOR,
                          labelcolor=TEXT_COLOR, bbox_to_anchor=(1.3, 1.1))
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.caption("Azul = aluno | Laranja tracejado = media do dataset")
            else:
                labels = ["Idade", "Fase", "INDE"]
                vals = [idade, fase, inde]
                ref_v = [fase + 6, fase, 7.0]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(labels, vals, color=[GREEN, BLUE, RED])
                for i, ref in enumerate(ref_v):
                    ax.plot([i - 0.4, i + 0.4], [ref, ref], linestyle="dashed", color=MUTED_COLOR, linewidth=1)
                _style_ax(fig, ax)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.caption("Preencha mais indices na sidebar para ver o grafico radar.")

            st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# TAB 2: PREDICAO EM LOTE (CSV)
# ======================================================
with tab_lote:
    st.subheader("Predicao em lote via CSV")
    st.markdown("""
    Faca upload de um arquivo CSV com os dados dos alunos. Colunas aceitas:
    `inde`, `iaa`, `ieg`, `ips`, `ida`, `ipv`, `matem`, `portug`, `genero`, `pedra_22`, `ano_ingresso`

    Colunas `idade` e `fase` sao opcionais (usadas apenas para explicacao, nao para o modelo).
    """)

    col_upload, col_template = st.columns([3, 1])
    with col_template:
        # Gera CSV template para download
        template_df = pd.DataFrame({
            "idade": [14, 18, 10],
            "fase": [4, 5, 4],
            "inde": [7.2, 4.5, 8.5],
            "iaa": [6.5, 3.2, 8.0],
            "ieg": [7.0, 4.0, 7.5],
            "ips": [8.0, "", 8.0],
            "ida": [6.8, "", 7.8],
            "ipv": [7.5, "", 8.2],
            "matem": [7.5, 3.5, 8.0],
            "portug": [6.8, 4.0, 7.5],
            "genero": ["M", "F", "M"],
            "pedra_22": ["Ametista", "Quartzo", "Topazio"],
        })
        st.download_button(
            "Baixar template CSV",
            template_df.to_csv(index=False),
            "template_alunos.csv",
            "text/csv",
            use_container_width=True,
        )

    with col_upload:
        uploaded = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.write(f"**{len(df_upload)} alunos carregados**")

            # Processa cada aluno
            results = []
            progress = st.progress(0)
            for i, row in df_upload.iterrows():
                features_row = row.dropna().to_dict()
                # Converte tipos
                for k in features_row:
                    if k in ["idade", "fase", "ano_ingresso"]:
                        features_row[k] = int(features_row[k])
                    elif k not in ["genero", "pedra_22"]:
                        features_row[k] = float(features_row[k])

                pred = predict_risk(model, features_row)
                conf = prediction_confidence(model, features_row)
                niv = risk_score(features_row)
                sug = intervention_suggestion(niv)

                results.append({
                    "Aluno": i + 1,
                    "Idade": features_row.get("idade", "-"),
                    "Fase": features_row.get("fase", "-"),
                    "INDE": features_row.get("inde", "-"),
                    "Risco": "Em risco" if pred == 1 else "Sem risco",
                    "Probabilidade (%)": conf,
                    "Nivel": niv,
                    "Intervencao": sug,
                })
                progress.progress((i + 1) / len(df_upload))

            df_results = pd.DataFrame(results)
            progress.empty()

            # Resumo visual
            n_risco = sum(1 for r in results if r["Risco"] == "Em risco")
            n_sem = len(results) - n_risco

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Total de alunos", len(results))
            col_s2.metric("Em risco", n_risco)
            col_s3.metric("Sem risco", n_sem)
            col_s4.metric("Taxa de risco", f"{n_risco / len(results):.1%}")

            # Grafico donut do lote
            col_table, col_chart = st.columns([3, 1.5])

            with col_chart:
                fig, ax = plt.subplots(figsize=(3, 3))
                sizes = [n_risco, n_sem]
                colors_pie = [RED, GREEN]
                labels_pie = [f"Em risco\n{n_risco}", f"Sem risco\n{n_sem}"]
                ax.pie(sizes, labels=labels_pie, colors=colors_pie,
                       startangle=90, wedgeprops=dict(width=0.4, edgecolor=BG_COLOR),
                       textprops=dict(color=TEXT_COLOR, fontsize=9))
                ax.set_facecolor(BG_COLOR)
                fig.patch.set_facecolor(BG_COLOR)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            with col_table:
                # Colore a coluna de risco
                def _color_risco(val):
                    if val == "Em risco":
                        return "background-color: #3d1f1f; color: #ff6b6b"
                    return "background-color: #1a3d1f; color: #6bff6b"

                styled = df_results.style.applymap(_color_risco, subset=["Risco"])
                st.dataframe(styled, use_container_width=True, height=400)

            # Download dos resultados
            csv_out = df_results.to_csv(index=False)
            st.download_button(
                "Baixar resultados CSV",
                csv_out,
                "resultados_predicao.csv",
                "text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Erro ao processar CSV: {e}")

# ======================================================
# TAB 3: HISTORICO DA SESSAO
# ======================================================
with tab_hist:
    st.subheader("Historico de predicoes desta sessao")

    if not st.session_state.historico:
        st.info("Nenhuma predicao realizada ainda. Use a aba 'Predicao Individual' para comecar.")
    else:
        hist = st.session_state.historico
        df_hist = pd.DataFrame(hist)
        df_hist.index = range(1, len(df_hist) + 1)
        df_hist.index.name = "#"

        # Renomeia colunas para display
        df_display = df_hist.rename(columns={
            "idade": "Idade", "fase": "Fase", "inde": "INDE",
            "risco": "Risco (1=sim)", "probabilidade": "Prob (%)", "nivel": "Nivel",
        })

        # Resumo
        n_total = len(hist)
        n_risco = sum(1 for h in hist if h["risco"] == 1)
        n_sem = n_total - n_risco

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total predicoes", n_total)
        c2.metric("Em risco", n_risco)
        c3.metric("Sem risco", n_sem)
        c4.metric("Taxa de risco", f"{n_risco / n_total:.1%}" if n_total > 0 else "0%")

        col_hist_table, col_hist_chart = st.columns([3, 1.5])

        with col_hist_table:
            st.dataframe(df_display, use_container_width=True)

        with col_hist_chart:
            if n_total >= 2:
                fig, ax = plt.subplots(figsize=(3, 3))
                probs = [h["probabilidade"] for h in hist]
                colors_bar = [RED if h["risco"] == 1 else GREEN for h in hist]
                ax.bar(range(1, n_total + 1), probs, color=colors_bar, width=0.6)
                ax.set_xlabel("Predicao #", fontsize=9)
                ax.set_ylabel("Prob. risco (%)", fontsize=9)
                ax.set_ylim(0, 105)
                ax.axhline(y=50, color=YELLOW, linestyle="--", linewidth=1, alpha=0.5)
                _style_ax(fig, ax)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(3, 3))
                sizes = [n_risco, n_sem] if n_risco > 0 or n_sem > 0 else [1]
                colors_p = [RED, GREEN] if len(sizes) == 2 else [MUTED_COLOR]
                ax.pie(sizes, colors=colors_p,
                       startangle=90, wedgeprops=dict(width=0.4, edgecolor=BG_COLOR))
                ax.set_facecolor(BG_COLOR)
                fig.patch.set_facecolor(BG_COLOR)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

        # Botao limpar historico
        if st.button("Limpar historico"):
            st.session_state.historico = []
            st.rerun()

# ======================================================
# DASHBOARD DE MONITORAMENTO
# ======================================================
st.divider()
with st.expander("Dashboard de Monitoramento", expanded=False):
    if _metrics_data:
        metrics = _metrics_data
        test = metrics.get("test_metrics", {})
        cv = metrics.get("cv_results", {})
        ref_stats = metrics.get("reference_stats", {})
        target_bal = metrics.get("target_balance", 0.7)

        # ---- ROW 1: Metricas + Donut ----
        col_metrics, col_donut = st.columns([3, 1.5])

        with col_metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AUC-ROC", f"{test.get('auc', 0):.3f}")
            c2.metric("F1 Score", f"{test.get('f1', 0):.3f}")
            c3.metric("Precision", f"{test.get('precision', 0):.3f}")
            c4.metric("Recall", f"{test.get('recall', 0):.3f}")

            st.write(f"**Melhor modelo:** {metrics.get('best_model', 'N/A')}")
            st.write(f"**Dataset:** {metrics.get('dataset_shape', 'N/A')} (linhas, features)")

        with col_donut:
            st.markdown("**Distribuicao do Target**")
            fig, ax = plt.subplots(figsize=(3, 3))
            sizes = [target_bal, 1 - target_bal]
            colors_pie = [RED, GREEN]
            labels_pie = [f"Em risco\n{target_bal:.1%}", f"Sem risco\n{1 - target_bal:.1%}"]
            ax.pie(sizes, labels=labels_pie, colors=colors_pie,
                   startangle=90, wedgeprops=dict(width=0.4, edgecolor=BG_COLOR),
                   textprops=dict(color=TEXT_COLOR, fontsize=8))
            ax.set_facecolor(BG_COLOR)
            fig.patch.set_facecolor(BG_COLOR)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        st.divider()

        # ---- ROW 2: Comparacao de Modelos + Matriz de Confusao ----
        col_cv, col_cm = st.columns(2)

        with col_cv:
            st.markdown("**Comparacao de Modelos (Cross-Validation 5-fold)**")
            if cv:
                model_names = list(cv.keys())
                short_names = [n.replace("Regression", "Reg").replace("Classifier", "") for n in model_names]
                f1_means = [cv[m]["f1_mean"] for m in model_names]
                f1_stds = [cv[m]["f1_std"] for m in model_names]
                auc_means = [cv[m]["auc_mean"] for m in model_names]
                auc_stds = [cv[m]["auc_std"] for m in model_names]
                acc_means = [cv[m]["accuracy_mean"] for m in model_names]

                x = np.arange(len(model_names))
                width = 0.25

                fig, ax = plt.subplots(figsize=(6, 3.5))
                bars1 = ax.bar(x - width, f1_means, width, yerr=f1_stds, label="F1",
                               color=CYAN, capsize=3, error_kw=dict(ecolor=MUTED_COLOR, lw=1))
                bars2 = ax.bar(x, auc_means, width, yerr=auc_stds, label="AUC",
                               color=PURPLE, capsize=3, error_kw=dict(ecolor=MUTED_COLOR, lw=1))
                bars3 = ax.bar(x + width, acc_means, width, label="Accuracy",
                               color=ORANGE)

                ax.set_xticks(x)
                ax.set_xticklabels(short_names, fontsize=9)
                ax.set_ylim(0.8, 1.02)
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
                ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=MUTED_COLOR, labelcolor=TEXT_COLOR)

                for bars in [bars1, bars2, bars3]:
                    for bar in bars:
                        h = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                                f"{h:.3f}", ha="center", va="bottom", fontsize=6, color=TEXT_COLOR)

                _style_ax(fig, ax)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

        with col_cm:
            st.markdown("**Matriz de Confusao (Teste Holdout)**")
            cm = test.get("confusion_matrix", [[0, 0], [0, 0]])
            cm = np.array(cm)
            class_labels = ["Sem risco", "Em risco"]

            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.imshow(cm, cmap="Blues", aspect="auto")

            for i in range(2):
                for j in range(2):
                    val = cm[i][j]
                    color = "white" if val > cm.max() / 2 else TEXT_COLOR
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=18, fontweight="bold", color=color)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(class_labels, fontsize=9)
            ax.set_yticklabels(class_labels, fontsize=9)
            ax.set_xlabel("Predito", fontsize=10)
            ax.set_ylabel("Real", fontsize=10)
            _style_ax(fig, ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        st.divider()

        # ---- ROW 3: Metricas por Classe + Feature Importance ----
        col_class, col_importance = st.columns(2)

        with col_class:
            st.markdown("**Metricas por Classe (Teste Holdout)**")
            report = test.get("classification_report", {})
            class_0 = report.get("0", {})
            class_1 = report.get("1", {})

            metric_names = ["Precision", "Recall", "F1-Score"]
            vals_0 = [class_0.get("precision", 0), class_0.get("recall", 0), class_0.get("f1-score", 0)]
            vals_1 = [class_1.get("precision", 0), class_1.get("recall", 0), class_1.get("f1-score", 0)]

            x = np.arange(len(metric_names))
            width = 0.3

            fig, ax = plt.subplots(figsize=(5, 3.5))
            b1 = ax.bar(x - width / 2, vals_0, width, label=f"Sem risco (n={int(class_0.get('support', 0))})",
                        color=GREEN)
            b2 = ax.bar(x + width / 2, vals_1, width, label=f"Em risco (n={int(class_1.get('support', 0))})",
                        color=RED)

            for bars in [b1, b2]:
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

            ax.set_xticks(x)
            ax.set_xticklabels(metric_names, fontsize=9)
            ax.set_ylim(0.95, 1.02)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
            ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=MUTED_COLOR, labelcolor=TEXT_COLOR)
            _style_ax(fig, ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col_importance:
            st.markdown("**Feature Importance (Coeficientes do Modelo)**")
            try:
                # Extrai coeficientes da LogisticRegression dentro do pipeline
                lr_model = model.named_steps["model"]
                fe_step = model.named_steps["feature_engineering"]
                feature_names = fe_step.feature_names_ or fe_step.column_order_

                if hasattr(lr_model, "coef_"):
                    coefs = lr_model.coef_[0]

                    # Mapeia nomes legiveis
                    name_map = {
                        "inde": "INDE", "iaa": "IAA", "ieg": "IEG", "ips": "IPS",
                        "ida": "IDA", "ipv": "IPV", "matem": "Matem", "portug": "Portug",
                        "genero": "Genero", "pedra_22": "Pedra", "cg": "CG", "cf": "CF",
                        "ct": "CT", "n_av": "N. Av", "media_academica": "Media Acad.",
                        "inde_baixo": "INDE Baixo", "anos_no_programa": "Anos Prog.",
                    }

                    labels_fi = [name_map.get(f, f) for f in feature_names]

                    # Ordena por valor absoluto
                    sorted_idx = np.argsort(np.abs(coefs))
                    top_n = min(12, len(sorted_idx))
                    top_idx = sorted_idx[-top_n:]

                    fi_labels = [labels_fi[i] for i in top_idx]
                    fi_vals = [coefs[i] for i in top_idx]
                    fi_colors = [RED if v > 0 else GREEN for v in fi_vals]

                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.barh(range(len(fi_labels)), fi_vals, color=fi_colors, height=0.6)
                    ax.set_yticks(range(len(fi_labels)))
                    ax.set_yticklabels(fi_labels, fontsize=8)
                    ax.axvline(x=0, color=MUTED_COLOR, linewidth=0.8)
                    ax.set_xlabel("Coeficiente", fontsize=9)
                    _style_ax(fig, ax)

                    # Anotacao
                    ax.text(0.98, 0.02, "Vermelho = aumenta risco\nVerde = reduz risco",
                            transform=ax.transAxes, fontsize=6, color=MUTED_COLOR,
                            ha="right", va="bottom")

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Modelo nao possui coeficientes (nao e linear).")
            except Exception as e:
                st.warning(f"Nao foi possivel extrair feature importance: {e}")

        st.divider()

        # ---- ROW 4: Distribuicao dos Indices ----
        st.markdown("**Distribuicao dos Indices (Referencia do Treino)**")
        idx_keys = [
            ("INDE", "inde"), ("IAA", "iaa"), ("IEG", "ieg"),
            ("IPS", "ips"), ("IDA", "ida"), ("IPV", "ipv"),
            ("Matem", "matem"), ("Portug", "portug"),
        ]
        idx_labels = []
        idx_mins = []
        idx_q25s = []
        idx_means = []
        idx_q75s = []
        idx_maxs = []

        for label, key in idx_keys:
            if key in ref_stats:
                s = ref_stats[key]
                idx_labels.append(label)
                idx_mins.append(s["min"])
                idx_q25s.append(s["q25"])
                idx_means.append(s["mean"])
                idx_q75s.append(s["q75"])
                idx_maxs.append(s["max"])

        if idx_labels:
            fig, ax = plt.subplots(figsize=(10, 3))
            y = np.arange(len(idx_labels))

            for i in range(len(idx_labels)):
                ax.plot([idx_mins[i], idx_maxs[i]], [i, i], color=MUTED_COLOR, linewidth=1, solid_capstyle="round")

            for i in range(len(idx_labels)):
                ax.barh(i, idx_q75s[i] - idx_q25s[i], left=idx_q25s[i], height=0.4,
                        color=PURPLE, alpha=0.7, edgecolor="none")

            ax.scatter(idx_means, y, color=CYAN, zorder=5, s=30, marker="D")

            ax.set_yticks(y)
            ax.set_yticklabels(idx_labels, fontsize=9)
            ax.set_xlim(0, 10.5)
            ax.set_xlabel("Valor", fontsize=9)
            _style_ax(fig, ax)

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=MUTED_COLOR, lw=1, label="Min-Max"),
                plt.Rectangle((0, 0), 1, 1, fc=PURPLE, alpha=0.7, label="Q25-Q75"),
                Line2D([0], [0], marker="D", color="w", markerfacecolor=CYAN, markersize=6, label="Media", linestyle="None"),
            ]
            ax.legend(handles=legend_elements, fontsize=7, facecolor=CARD_BG,
                      edgecolor=MUTED_COLOR, labelcolor=TEXT_COLOR, loc="lower right")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    else:
        st.warning("Metricas nao encontradas. Execute o treinamento primeiro.")
