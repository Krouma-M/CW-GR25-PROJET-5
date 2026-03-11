import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from data_processing import load_data, clean_data, split_data, normalize_data, optimize_memory
def evaluate_models(trained_models, X_test, y_test):
    results = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            "Accuracy"  : accuracy_score(y_test, y_pred),
            "Precision" : precision_score(y_test, y_pred),
            "Recall"    : recall_score(y_test, y_pred),
            "F1"        : f1_score(y_test, y_pred),
            "ROC-AUC"   : roc_auc_score(y_test, y_prob)
        }
    
    return results
def print_results(results):
    print(" RÉSULTATS DES MODÈLES ")
    
    for name, metrics in results.items():
        print(f"\n {name} :")
        print(f"   Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"   Precision : {metrics['Precision']:.4f}")
        print(f"   Recall    : {metrics['Recall']:.4f}")
        print(f"   F1        : {metrics['F1']:.4f}")
        print(f"   ROC-AUC   : {metrics['ROC-AUC']:.4f}")
    
    print(" MEILLEUR MODÈLE ")
    best = max(results, key=lambda x: results[x]["Recall"])
    print(f" {best} avec Recall = {results[best]['Recall']:.4f}")
    
    return best
if __name__ == "__main__":
    # Préparer les données
    X, y = load_data()
    X, y = clean_data(X, y)
    X = optimize_memory(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, scaler = normalize_data(X_train, X_test)
    
    # Charger les modèles sauvegardés
    trained_models = {
        "Random Forest": joblib.load("models/Random Forest.pkl"),
        "LightGBM"     : joblib.load("models/LightGBM.pkl"),
        "CatBoost"     : joblib.load("models/CatBoost.pkl")
    }
    
    # Évaluer et afficher
    results = evaluate_models(trained_models, X_test, y_test)
    best = print_results(results)