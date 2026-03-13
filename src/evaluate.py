"""
Evaluation module for model performance assessment.
"""

import os
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd

sys.path.append(os.path.dirname(__file__))


def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    return metrics


def print_evaluation(metrics):
    print(f"\n--- {metrics['model']} ---")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)


def compare_models(results):
    df = pd.DataFrame(results)
    df = df.set_index("model")
    return df


def print_comparison(df):
    print("\n" + "=" * 60)
    print("COMPARAISON DES MODÈLES")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)
    print("\nMeilleur modèle par métrique:")
    for col in df.columns:
        best_idx = df[col].idxmax()
        best_val = df[col].max()
        print(f"  {col}: {best_idx} ({best_val:.3f})")


def select_best_model(df):
    """
    Sélectionne le meilleur modèle selon la règle :
    1. Meilleur Recall
    2. En cas d'égalité → meilleur ROC-AUC
    3. En cas d'égalité → meilleur F1

    Args:
        df: DataFrame avec les métriques (index = nom du modèle)

    Returns:
        str: Nom du meilleur modèle
    """
    # Arrondi à 4 décimales pour détecter les égalités
    recall_max = df["recall"].max()
    best_by_recall = df[df["recall"].round(4) == round(recall_max, 4)]

    print(f"\n  → Meilleur Recall : {recall_max:.4f}")
    print(f"  → Modèles ex-aequo sur Recall : {list(best_by_recall.index)}")

    if len(best_by_recall) == 1:
        # Un seul gagnant sur le Recall
        best_name = best_by_recall.index[0]
    else:
        # Égalité sur Recall → départager par ROC-AUC
        auc_max = best_by_recall["roc_auc"].max()
        best_by_auc = best_by_recall[best_by_recall["roc_auc"].round(4) == round(auc_max, 4)]

        print(f"  → Égalité ! Départage par ROC-AUC : {auc_max:.4f}")
        print(f"  → Modèles ex-aequo sur AUC : {list(best_by_auc.index)}")

        if len(best_by_auc) == 1:
            best_name = best_by_auc.index[0]
        else:
            # Égalité sur AUC aussi → départager par F1
            best_name = best_by_auc["f1"].idxmax()
            print(f"  → Égalité encore ! Départage par F1 → {best_name}")

    return best_name


if __name__ == "__main__":
    import joblib
    from data_processing import load_data, preprocess_data

    # Charger les données
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)

    # Charger les modèles sauvegardés
    models = {
        "Random Forest": joblib.load("models/Random Forest.pkl"),
        "LightGBM"     : joblib.load("models/LightGBM.pkl"),
        "CatBoost"     : joblib.load("models/CatBoost.pkl"),
    }

    # Évaluer chaque modèle
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba, name)
        print_evaluation(metrics)
        results.append(metrics)

    # Comparer tous les modèles
    df_results = compare_models(results)
    print_comparison(df_results)

    # ── Sélection du meilleur modèle ──────────────────────────────────────────
    # Règle : 1) Meilleur Recall  2) Ex-aequo → Meilleur ROC-AUC  3) Ex-aequo → Meilleur F1
    print("\n" + "=" * 60)
    print("SÉLECTION DU MEILLEUR MODÈLE")
    print("=" * 60)

    best_model_name = select_best_model(df_results)
    best_model = joblib.load(f"models/{best_model_name}.pkl")

    joblib.dump(best_model, "models/best_model.pkl")
    print(f" Meilleur modèle : {best_model_name}")
    print(f" Sauvegardé : models/best_model.pkl")