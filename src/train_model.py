import joblib
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from data_processing import load_data, clean_data, split_data, normalize_data, optimize_memory
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            random_state=42
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100,
            random_state=42,
            verbose=0
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Entraînement de {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f" {name} terminé !")
    
    return trained_models 
def save_model(model, model_name):
    os.makedirs("models", exist_ok=True)
    
    filename = f"models/{model_name}.pkl"
    joblib.dump(model, filename)
    
    print(f" Modèle sauvegardé : {filename}")
if __name__ == "__main__":
    # Charger et préparer les données
    X, y = load_data()
    X, y = clean_data(X, y)
    X = optimize_memory(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, scaler = normalize_data(X_train, X_test)
    
    # Entraîner les 3 modèles
    trained_models = train_models(X_train, y_train)
    
    # Sauvegarder tous les modèles
    for name, model in trained_models.items():
        save_model(model, name)
    
    print(" Tous les modèles sont entraînés et sauvegardés !")