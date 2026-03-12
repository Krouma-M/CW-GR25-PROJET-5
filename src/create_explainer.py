import joblib
import shap
import os

# Chemins (à adapter selon votre structure)
MODEL_PATH = "appendicite_pediatric/models/model_appendicite.pkl"
EXPLAINER_PATH = "appendicite_pediatric/explications/explainer_appendicite.pkl"

def main():
    print("Chargement du modèle...")
    model = joblib.load(MODEL_PATH)

    # Si le modèle est un pipeline scikit-learn, on extrait le classifieur
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
        print("Modèle pipeline détecté, classifieur extrait.")
    else:
        classifier = model
        print("Modèle simple détecté.")

    # Choix de l'explainer selon le type de modèle
    # Pour les modèles arborescents (RandomForest, LightGBM, CatBoost, XGBoost)
    if 'tree' in str(type(classifier)).lower() or hasattr(classifier, 'feature_importances_'):
        explainer = shap.TreeExplainer(classifier)
        print("TreeExplainer créé avec succès.")
    else:
        # Pour les modèles non arborescents (SVM, régression logistique, etc.)
        # On a besoin d'un échantillon de données d'entraînement pour KernelExplainer
        print("Modèle non arborescent. KernelExplainer nécessite un échantillon.")
        print("Veuillez charger un échantillon de X_train (par exemple depuis un fichier).")
        # Exemple si vous avez sauvegardé X_train_sample (à adapter)
        # X_train_sample = joblib.load("data/processed/X_train_sample.pkl")
        # explainer = shap.KernelExplainer(classifier.predict_proba, X_train_sample)
        raise NotImplementedError("Pour les modèles non arborescents, vous devez fournir un échantillon.")

    # Sauvegarde
    joblib.dump(explainer, EXPLAINER_PATH)
    print(f"Explainer sauvegardé dans {EXPLAINER_PATH}")

if __name__ == "__main__":
    main()