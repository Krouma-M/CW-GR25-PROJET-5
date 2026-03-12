import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -----------------------------
# CONFIGURATION PAGE
# -----------------------------

st.set_page_config(
    page_title="Appendix — Diagnostic Pédiatrique",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CSS PREMIUM
# -----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050d1a !important;
    color: #e8edf5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,168,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,255,180,0.05) 0%, transparent 60%),
        #050d1a !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: transparent !important;
    border: none !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 1200px !important;
    padding: 2rem 2.5rem 4rem !important;
}

.hero {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 2.5rem 3rem;
    background: linear-gradient(135deg, rgba(0,168,255,0.08) 0%, rgba(0,255,180,0.04) 100%);
    border: 1px solid rgba(0,168,255,0.15);
    border-radius: 20px;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,168,255,0.12), transparent 70%);
    border-radius: 50%;
}

.hero-icon { font-size: 3.5rem; line-height: 1; filter: drop-shadow(0 0 20px rgba(0,168,255,0.5)); }

.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

.hero-sub {
    font-size: 0.95rem;
    color: rgba(232,237,245,0.6);
    margin-top: 0.4rem;
    font-weight: 300;
}

.hero-badge {
    margin-left: auto;
    background: rgba(0,255,180,0.08);
    border: 1px solid rgba(0,255,180,0.25);
    color: #00ffb4;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    white-space: nowrap;
}

.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00a8ff;
    margin-bottom: 0.4rem;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1.5rem;
}

.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem 1.8rem 1.8rem;
    backdrop-filter: blur(10px);
    height: 100%;
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: rgba(232,237,245,0.5);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.demo-banner {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(255,190,0,0.06);
    border: 1px solid rgba(255,190,0,0.2);
    border-radius: 12px;
    padding: 0.85rem 1.25rem;
    margin-bottom: 2rem;
    font-size: 0.85rem;
    color: rgba(255,220,100,0.9);
}

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8edf5 !important;
}

[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8edf5 !important;
}

label[data-testid="stWidgetLabel"] p {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: rgba(232,237,245,0.55) !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #00a8ff 0%, #0055cc 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 24px rgba(0,168,255,0.3) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,168,255,0.45) !important;
}

.result-high {
    background: linear-gradient(135deg, rgba(255,60,60,0.1), rgba(200,40,40,0.05));
    border: 1px solid rgba(255,60,60,0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
}

.result-low {
    background: linear-gradient(135deg, rgba(0,255,140,0.08), rgba(0,180,90,0.04));
    border: 1px solid rgba(0,255,140,0.25);
    border-radius: 16px;
    padding: 1.5rem 2rem;
}

.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.result-sub { font-size: 0.85rem; color: rgba(232,237,245,0.55); }

.stat-row { display: flex; gap: 1rem; margin: 1.5rem 0; }

.stat-box {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #00a8ff;
}

.stat-label {
    font-size: 0.7rem;
    color: rgba(232,237,245,0.45);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2.5rem 0;
}

.shap-note {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    font-size: 0.82rem;
    color: rgba(232,237,245,0.5);
    line-height: 1.8;
}

.footer {
    text-align: center;
    font-size: 0.72rem;
    color: rgba(232,237,245,0.2);
    margin-top: 3rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HERO
# -----------------------------

st.markdown("""
<div class="hero">
    <div class="hero-icon">🩺</div>
    <div>
        <div class="hero-title">Appendix</div>
        <div class="hero-sub">Système d'aide au diagnostic · Appendicite pédiatrique · Modèle CatBoost</div>
    </div>
    <div class="hero-badge">v1.0 · Démo</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="demo-banner">
    &nbsp;&nbsp;<strong>Mode démonstration</strong> — Prédictions simulées. Le modèle réel sera activé une fois l'entraînement finalisé.
</div>
""", unsafe_allow_html=True)

# -----------------------------
# FORMULAIRE
# -----------------------------

st.markdown('<div class="section-label">Étape 1 — Saisie</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Données cliniques du patient</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="card"><div class="card-title"> Profil patient</div>', unsafe_allow_html=True)
    age = st.slider("Âge (années)", 1, 18, 8)
    sexe = st.selectbox("Sexe biologique", ["Garçon", "Fille"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title"> Symptômes cliniques</div>', unsafe_allow_html=True)
    douleur = st.selectbox("Douleur fosse iliaque droite", ["Oui", "Non"])
    fievre = st.selectbox("Fièvre > 38 °C", ["Oui", "Non"])
    vomissements = st.selectbox("Nausées / Vomissements", ["Oui", "Non"])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><div class="card-title">🔬 Résultats biologiques</div>', unsafe_allow_html=True)
    wbc = st.number_input("Leucocytes WBC (G/L)", 0.0, 30.0, 8.5, step=0.1,
                          help="Normale : 4.5–11.0 G/L")
    crp = st.number_input("CRP (mg/L)", 0.0, 300.0, 10.0, step=1.0,
                          help="Normale : < 5 mg/L")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1.5, 2, 1.5])
with btn_col:
    analyser = st.button("🔍  Lancer l'analyse diagnostique")

# -----------------------------
# RÉSULTATS
# -----------------------------

if analyser:

    sexe_val = 1 if sexe == "Garçon" else 0
    douleur_val = 1 if douleur == "Oui" else 0
    fievre_val = 1 if fievre == "Oui" else 0
    vomissements_val = 1 if vomissements == "Oui" else 0

    score = (douleur_val * 0.35 + fievre_val * 0.20 + vomissements_val * 0.15
             + min(wbc / 20.0, 1) * 0.18 + min(crp / 150.0, 1) * 0.12)
    probabilite = float(np.clip(score + np.random.uniform(-0.03, 0.05), 0.02, 0.97))
    prediction = 1 if probabilite >= 0.5 else 0
    pct = round(probabilite * 100, 1)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Étape 2 — Résultats</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analyse diagnostique</div>', unsafe_allow_html=True)

    res_col, gauge_col = st.columns([1.1, 1], gap="large")

    with res_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-high">
                <div style="font-size:2rem;margin-bottom:0.5rem"></div>
                <div class="result-title">Risque Élevé d'Appendicite</div>
                <div class="result-sub">Probabilité estimée : <strong style="color:#ff6060">{pct}%</strong><br>
                Consultation chirurgicale pédiatrique recommandée</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <div style="font-size:2rem;margin-bottom:0.5rem"></div>
                <div class="result-title">Risque Faible d'Appendicite</div>
                <div class="result-sub">Probabilité estimée : <strong style="color:#00ff8c">{pct}%</strong><br>
                Surveillance clinique et réévaluation conseillées</div>
            </div>""", unsafe_allow_html=True)

        risk_label = "ÉLEVÉ" if pct >= 70 else ("MODÉRÉ" if pct >= 30 else "FAIBLE")
        confiance = round(abs(probabilite - 0.5) * 200, 0)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-value">{pct}%</div>
                <div class="stat-label">Probabilité</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{risk_label}</div>
                <div class="stat-label">Niveau risque</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{confiance:.0f}%</div>
                <div class="stat-label">Confiance</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix': '%', 'font': {'size': 40, 'color': '#ffffff', 'family': 'Syne'}},
            title={'text': "Score de risque estimé",
                   'font': {'size': 13, 'color': 'rgba(232,237,245,0.45)', 'family': 'DM Sans'}},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': 'rgba(255,255,255,0.15)',
                    'tickfont': {'color': 'rgba(255,255,255,0.35)', 'size': 10}
                },
                'bar': {'color': '#00a8ff', 'thickness': 0.22},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0,255,140,0.1)'},
                    {'range': [30, 70], 'color': 'rgba(255,190,0,0.1)'},
                    {'range': [70, 100], 'color': 'rgba(255,60,60,0.12)'}
                ],
                'threshold': {
                    'line': {'color': '#00ffb4', 'width': 2},
                    'thickness': 0.75,
                    'value': pct
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=270,
            margin=dict(t=40, b=10, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── SHAP RÉEL ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Étape 3 — Interprétabilité</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explication de la décision — Analyse SHAP</div>', unsafe_allow_html=True)

    import os

    SHAP_DIR = "/workspaces/CW-GR25-PROJET-5/figures/shap"

    shap_options = {
        " CatBoost — Résumé (Beeswarm)":  f"{SHAP_DIR}/catboost_summary.png",
        " CatBoost — Importance globale": f"{SHAP_DIR}/catboost_importance.png",
    }

    shap_col, note_col = st.columns([2.2, 1], gap="large")

    with shap_col:
        choix = st.selectbox("Sélectionner le graphique SHAP", list(shap_options.keys()))
        img_path = shap_options[choix]
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Image non trouvée : {img_path}")

    with note_col:
        st.markdown("""
        <div class="shap-note">
            <strong style="color:#00a8ff;font-family:Syne;font-size:0.88rem"> Légende</strong><br><br>
             <strong>Résumé (Beeswarm)</strong><br>
            Chaque point = un patient. La couleur indique la valeur de la variable
            (rouge = élevée, bleue = faible). La position horizontale montre
            l'impact sur la prédiction.<br><br>
             <strong>Importance globale</strong><br>
            Longueur de barre = importance moyenne de chaque variable sur
            l'ensemble des patients.
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="margin-top:2rem;padding:1rem 1.5rem;
         background:rgba(255,255,255,0.015);
         border-left:3px solid rgba(0,168,255,0.35);
         border-radius:0 10px 10px 0;
         font-size:0.8rem;color:rgba(232,237,245,0.38);line-height:1.7">
         <strong style="color:rgba(232,237,245,0.55)">Avertissement médical</strong> —
        Cet outil constitue une aide au diagnostic et ne se substitue en aucun cas
        au jugement clinique d'un médecin qualifié. Toute décision thérapeutique
        doit être prise par un professionnel de santé habilité.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Appendix &nbsp;·&nbsp; Projet CW-GR25 &nbsp;·&nbsp; 2025 &nbsp;·&nbsp; Usage médical supervisé uniquement
</div>
""", unsafe_allow_html=True)