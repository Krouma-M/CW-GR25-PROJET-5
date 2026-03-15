import joblib
import os
import sys
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import catboost as cb

sys.path.append(os.path.dirname(__file__))
from data_processing import load_data, preprocess_data, optimize_memory

print("Script lancé...")

# Charger et préparer les données
df = load_data()
X_train, X_test, y_train, y_test, le_target = preprocess_data(df)

# Créer le dossier models/
os.makedirs("models", exist_ok=True)

# 1. Random Forest avec GridSearchCV
print("\nEntraînement Random Forest...")
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring="f1", n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
joblib.dump(best_rf, "models/Random Forest.pkl")
print(f" Random Forest terminé ! Params : {grid_rf.best_params_}")

# 2. LightGBM avec GridSearchCV
print("\nEntraînement LightGBM...")
lgbm = lgb.LGBMClassifier(random_state=42, verbose=-1)
param_grid_lgb = {
    "n_estimators": [100, 200, 300],
    "max_depth": [-1, 5, 10],
    "learning_rate": [0.05, 0.1, 0.2],
}
grid_lgb = GridSearchCV(lgbm, param_grid_lgb, cv=5, scoring="f1", n_jobs=-1)
grid_lgb.fit(X_train, y_train)
best_lgb = grid_lgb.best_estimator_
joblib.dump(best_lgb, "models/LightGBM.pkl")
print(f" LightGBM terminé ! Params : {grid_lgb.best_params_}")

# 3. CatBoost avec GridSearchCV
print("\nEntraînement CatBoost...")
cat = cb.CatBoostClassifier(verbose=0, random_state=42)
param_grid_cat = {
    "iterations": [100, 200, 300],
    "depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
}
grid_cat = GridSearchCV(cat, param_grid_cat, cv=5, scoring="f1", n_jobs=-1)
grid_cat.fit(X_train, y_train)
best_cat = grid_cat.best_estimator_
joblib.dump(best_cat, "models/CatBoost.pkl")
print(f" CatBoost terminé ! Params : {grid_cat.best_params_}")

print(" Tous les modèles entraînés et sauvegardés !")
