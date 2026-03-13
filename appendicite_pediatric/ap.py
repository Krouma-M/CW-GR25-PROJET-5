import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AppendIA — Triage Pédiatrique",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main { background: #0d1117; color: #e6edf3; }
  .block-container { padding: 2rem 3rem; }
  h1, h2, h3 { font-family: 'Space Mono', monospace; }
  .hero-title { font-family: 'Space Mono', monospace; font-size: 2.6rem; font-weight: 700; color: #f0f6fc; letter-spacing: -1px; line-height: 1.1; }
  .hero-sub { color: #8b949e; font-size: 1rem; margin-top: 0.4rem; font-weight: 300; }
  .metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center; }
  .metric-val { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #58a6ff; }
  .metric-label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }
  .risk-high { background: linear-gradient(135deg, #3d0000, #1a0000); border: 1px solid #f85149; border-radius: 12px; padding: 1.5rem 2rem; text-align: center; }
  .risk-low { background: linear-gradient(135deg, #003320, #001a0f); border: 1px solid #3fb950; border-radius: 12px; padding: 1.5rem 2rem; text-align: center; }
  .risk-label { font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700; letter-spacing: 2px; }
  .disclaimer { background: #161b22; border-left: 3px solid #d29922; border-radius: 6px; padding: 0.8rem 1.2rem; font-size: 0.82rem; color: #8b949e; margin-top: 1.5rem; }
  .stButton > button { background: #238636; color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-family: 'Space Mono', monospace; font-size: 0.9rem; font-weight: 700; letter-spacing: 1px; width: 100%; transition: background 0.2s; }
  .stButton > button:hover { background: #2ea043; }
  div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ── Chargement du modèle ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Colonnes du modèle — SANS les colonnes de fuite ──────────────────────────
# Supprimées : Management, Severity, Length_of_Stay, Perforation, Appendicular_Abscess
ALL_COLUMNS = [
    'Age', 'BMI', 'Sex', 'Height', 'Weight',
    'Alvarado_Score', 'Paedriatic_Appendicitis_Score', 'Appendix_on_US',
    'Appendix_Diameter', 'Migratory_Pain', 'Lower_Right_Abd_Pain',
    'Contralateral_Rebound_Tenderness', 'Coughing_Pain', 'Nausea',
    'Loss_of_Appetite', 'Body_Temperature', 'WBC_Count',
    'Neutrophil_Percentage', 'Segmented_Neutrophils', 'Neutrophilia',
    'RBC_Count', 'Hemoglobin', 'RDW', 'Thrombocyte_Count',
    'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine', 'CRP',
    'Dysuria', 'Stool', 'Peritonitis', 'Psoas_Sign',
    'Ipsilateral_Rebound_Tenderness', 'US_Performed', 'Free_Fluids',
    'Appendix_Wall_Layers', 'Target_Sign', 'Appendicolith', 'Perfusion',
    'Surrounding_Tissue_Reaction',
    'Abscess_Location', 'Pathological_Lymph_Nodes', 'Lymph_Nodes_Location',
    'Bowel_Wall_Thickening', 'Conglomerate_of_Bowel_Loops', 'Ileus',
    'Coprostasis', 'Meteorism', 'Enteritis', 'Gynecological_Findings',
]

# ── Valeurs par défaut ────────────────────────────────────────────────────────
DEFAULTS = {col: 0 for col in ALL_COLUMNS}
DEFAULTS.update({
    'Age': 8, 'BMI': 18.0, 'Height': 130.0, 'Weight': 30.0,
    'Body_Temperature': 37.5, 'WBC_Count': 8.0,
    'Neutrophil_Percentage': 55.0, 'Segmented_Neutrophils': 50.0,
    'RBC_Count': 4.5, 'Hemoglobin': 12.0, 'RDW': 13.0,
    'Thrombocyte_Count': 250.0, 'CRP': 5.0, 'Appendix_Diameter': 6.0,
    'Alvarado_Score': 5, 'Paedriatic_Appendicitis_Score': 5,
})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 AppendIA")
    st.markdown("<span style='color:#8b949e;font-size:0.82rem'>Modèle CatBoost — Recall 95.2%</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**🧒 Patient**")
    age    = st.slider("Âge (années)", 1, 18, 8)
    sex    = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Fille" if x == 0 else "Garçon")
    weight = st.slider("Poids (kg)", 5.0, 100.0, 30.0, step=0.5)
    height = st.slider("Taille (cm)", 60.0, 190.0, 130.0, step=1.0)
    bmi    = round(weight / ((height / 100) ** 2), 1)
    st.markdown(f"<span style='color:#58a6ff;font-size:0.85rem'>IMC calculé : {bmi}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("**🌡️ Signes vitaux & Bio**")
    temp   = st.slider("Température (°C)", 36.0, 41.0, 37.5, step=0.1)
    wbc    = st.slider("Leucocytes GB (×10³/µL)", 1.0, 30.0, 8.0, step=0.1)
    neutro = st.slider("Neutrophiles (%)", 10.0, 95.0, 55.0, step=0.5)
    crp    = st.slider("CRP (mg/L)", 0.0, 300.0, 5.0, step=1.0)

    st.divider()
    st.markdown("**🩺 Signes cliniques**")
    migr    = st.selectbox("Douleur migratoire FID",       [0,1], format_func=lambda x: "Oui" if x else "Non")
    lrap    = st.selectbox("Douleur fosse iliaque droite", [0,1], format_func=lambda x: "Oui" if x else "Non")
    rebound = st.selectbox("Rebond controlatéral",         [0,1], format_func=lambda x: "Oui" if x else "Non")
    ipsi    = st.selectbox("Rebond ipsilatéral",           [0,1], format_func=lambda x: "Oui" if x else "Non")
    cough   = st.selectbox("Douleur à la toux",            [0,1], format_func=lambda x: "Oui" if x else "Non")
    nausea  = st.selectbox("Nausées/vomissements",         [0,1], format_func=lambda x: "Oui" if x else "Non")
    anorex  = st.selectbox("Perte d'appétit",              [0,1], format_func=lambda x: "Oui" if x else "Non")
    psoas   = st.selectbox("Signe du psoas",               [0,1], format_func=lambda x: "Oui" if x else "Non")
    periton = st.selectbox("Péritonite",                   [0,1], format_func=lambda x: "Oui" if x else "Non")

    st.divider()
    st.markdown("**🔊 Échographie**")
    us_done  = st.selectbox("Échographie réalisée",  [0,1], format_func=lambda x: "Oui" if x else "Non")
    app_us   = st.selectbox("Appendice visualisé",   [0,1], format_func=lambda x: "Oui" if x else "Non")
    app_diam = st.slider("Diamètre appendice (mm)", 0.0, 20.0, 6.0, step=0.5)
    free_fl  = st.selectbox("Épanchement libre",     [0,1], format_func=lambda x: "Oui" if x else "Non")

    st.divider()
    st.markdown("**📊 Scores cliniques**")
    alvarado = st.slider("Score d'Alvarado", 0, 10, 5)
    pas      = st.slider("Pediatric Appendicitis Score", 0, 10, 5)

    st.divider()
    predict_btn = st.button("⚡ ANALYSER")

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_metrics = st.columns([2, 3])
with col_title:
    st.markdown('<div class="hero-title">Append<span style="color:#58a6ff">IA</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Aide au triage — Appendicite pédiatrique<br>CatBoost · Recall 95.2% · AUC 0.979</div>', unsafe_allow_html=True)

with col_metrics:
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, "95.2%",    "Recall"),
        (m2, "97.9%",    "AUC"),
        (m3, "88.2%",    "Précision"),
        (m4, "CatBoost", "Modèle"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{val}</div>
          <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Prédiction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not model_loaded:
        st.error("⚠️ `best_model.pkl` introuvable. Lance d'abord `evaluate.py`.")
    else:
        row = DEFAULTS.copy()
        row.update({
            'Age': age, 'BMI': bmi, 'Sex': sex, 'Height': height, 'Weight': weight,
            'Body_Temperature': temp, 'WBC_Count': wbc,
            'Neutrophil_Percentage': neutro, 'Segmented_Neutrophils': neutro * 0.9,
            'CRP': crp, 'Migratory_Pain': migr, 'Lower_Right_Abd_Pain': lrap,
            'Contralateral_Rebound_Tenderness': rebound,
            'Ipsilateral_Rebound_Tenderness': ipsi,
            'Coughing_Pain': cough, 'Nausea': nausea, 'Loss_of_Appetite': anorex,
            'Psoas_Sign': psoas, 'Peritonitis': periton,
            'US_Performed': us_done, 'Appendix_on_US': app_us,
            'Appendix_Diameter': app_diam, 'Free_Fluids': free_fl,
            'Alvarado_Score': alvarado, 'Paedriatic_Appendicitis_Score': pas,
        })

        input_df = pd.DataFrame([row])[ALL_COLUMNS]

        proba = model.predict_proba(input_df)[0][1]
        prediction = int(proba >= 0.5)

        col_res, col_shap = st.columns([1, 2])

        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                  <div class="risk-label" style="color:#f85149">⚠ RISQUE ÉLEVÉ</div>
                  <div style="font-size:2.8rem;font-family:'Space Mono',monospace;color:#f85149;margin:0.5rem 0">{proba:.1%}</div>
                  <div style="color:#8b949e;font-size:0.82rem">probabilité d'appendicite</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                  <div class="risk-label" style="color:#3fb950">✓ RISQUE FAIBLE</div>
                  <div style="font-size:2.8rem;font-family:'Space Mono',monospace;color:#3fb950;margin:0.5rem 0">{proba:.1%}</div>
                  <div style="color:#8b949e;font-size:0.82rem">probabilité d'appendicite</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge, ax = plt.subplots(figsize=(4, 0.5))
            fig_gauge.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")
            ax.barh(0, 1, color="#30363d", height=0.5)
            color = "#f85149" if proba >= 0.5 else "#3fb950"
            ax.barh(0, proba, color=color, height=0.5)
            ax.set_xlim(0, 1); ax.axis("off")
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)

        with col_shap:
            st.markdown("### Facteurs explicatifs (SHAP)")
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(input_df)
                sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

                shap_df = pd.DataFrame({
                    "feature": ALL_COLUMNS,
                    "shap": sv,
                }).sort_values("shap", key=abs, ascending=True).tail(10)

                fig_shap, ax2 = plt.subplots(figsize=(6, 4))
                fig_shap.patch.set_facecolor("#0d1117")
                ax2.set_facecolor("#161b22")
                colors = ["#f85149" if v > 0 else "#58a6ff" for v in shap_df["shap"]]
                ax2.barh(shap_df["feature"], shap_df["shap"], color=colors)
                ax2.axvline(0, color="#30363d", linewidth=1)
                ax2.tick_params(colors="#8b949e", labelsize=8)
                ax2.spines[:].set_color("#30363d")
                ax2.set_xlabel("Impact SHAP", color="#8b949e", fontsize=8)
                fig_shap.tight_layout()
                st.pyplot(fig_shap, use_container_width=True)
                plt.close(fig_shap)
            except Exception as e:
                st.info(f"SHAP non disponible : {e}")

        st.markdown("""
        <div class="disclaimer">
          ⚠️ <strong>Outil d'aide à la décision uniquement.</strong>
          Ce score ne remplace pas le jugement clinique. Toute décision thérapeutique reste sous la responsabilité du médecin.
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#30363d">
      <div style="font-size:4rem">🔬</div>
      <div style="font-family:'Space Mono',monospace;font-size:1rem;margin-top:1rem">
        Renseigne les paramètres cliniques<br>dans la barre latérale, puis clique sur ANALYSER
      </div>
    </div>""", unsafe_allow_html=True)