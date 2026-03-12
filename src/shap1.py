"""
SHAP analysis module for model explainability.
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, preprocess_data

# Créer le dossier pour les figures
os.makedirs("figures/shap", exist_ok=True)


def load_best_model():
    """Charge le meilleur modèle sauvegardé par evaluate.py"""
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "best_model.pkl introuvable ! Lance d'abord evaluate.py"
        )
    return joblib.load(model_path)


def generate_global_shap(model, X_train, feature_names):
    """
    Génère les graphiques SHAP globaux sur toutes les données.

    Args:
        model: Modèle entraîné
        X_train: Données d'entraînement
        feature_names: Noms des colonnes

    Returns:
        shap_values: Valeurs SHAP calculées
        explainer: Explainer SHAP
    """
    print("\n--- Génération SHAP global ---")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Gérer les formats différents
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values

    # 1. Graphique résumé beeswarm
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values_plot,
        X_train,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title(
        "Graphique SHAP Résumé\n(Impact sur la prédiction d'appendicite)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("figures/shap/best_model_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Résumé sauvegardé: figures/shap/best_model_summary.png")

    # 2. Graphique importance globale
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_plot,
        X_train,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(
        "Importance des caractéristiques (SHAP)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("figures/shap/best_model_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Importance sauvegardée: figures/shap/best_model_importance.png")

    # 3. Top 10 features importantes
    mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()
    top_10_idx = np.argsort(mean_abs_shap)[-10:][::-1]

    print("\nTop 10 caractéristiques les plus importantes :")
    for i, idx in enumerate(top_10_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:40s} (Score: {mean_abs_shap[idx]:.4f})")

    return shap_values, explainer


def explain_patient(model, patient_data, feature_names):
    """
    Explique la prédiction pour UN patient spécifique.
    Utilisée par app.py pour afficher le graphique SHAP individuel.

    Args:
        model: Modèle entraîné
        patient_data: Données du patient (array 1D ou 2D)
        feature_names: Noms des colonnes

    Returns:
        fig: Figure matplotlib avec le graphique waterfall
        probability: Probabilité d'appendicite
        prediction: Prédiction (0 ou 1)
    """
    explainer = shap.TreeExplainer(model)

    # S'assurer que les données sont en 2D
    if len(patient_data.shape) == 1:
        patient_data = patient_data.reshape(1, -1)

    shap_values = explainer.shap_values(patient_data)

    # Gérer les formats différents
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values[0]
        base_value = explainer.expected_value

    # Prédiction et probabilité
    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0][1]

    # Graphique waterfall pour ce patient
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv,
            base_values=base_value,
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()

    return fig, probability, prediction


if __name__ == "__main__":
    print("Chargement du meilleur modèle...")
    model = load_best_model()

    print("Chargement des données...")
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    feature_names = X_train.columns.tolist()

    print("Génération des graphiques SHAP globaux...")
    shap_values, explainer = generate_global_shap(model, X_train, feature_names)

    print(" Analyse SHAP terminée !")
    print("Fichiers générés :")
    print("  - figures/shap/best_model_summary.png")
    print("  - figures/shap/best_model_importance.png")
