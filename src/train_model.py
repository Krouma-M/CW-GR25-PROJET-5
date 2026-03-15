import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import catboost as cb
import joblib
import shap
import os

print("Script lancé...")

# Charger les données brutes
df = pd.read_csv("data/raw/appendicitis.csv")

# Supprimer les lignes où la cible est manquante
df = df.dropna(subset=["Diagnosis"])

# Séparer X et y
target_col = "Diagnosis"
y = df[target_col].astype(str)
X = df.drop(columns=[target_col])

# Encoder la cible
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Classes: {le_target.classes_}")

# Inverser les labels pour que 1 = appendicite, 0 = non appendicite
y_encoded = 1 - y_encoded   # car les classes originales sont ['appendicitis', 'no appendicitis'] → 0 et 1

# Définir les colonnes numériques et catégorielles (d'après la sortie de votre script)
numeric_features = [
    'Age', 'BMI', 'Height', 'Weight', 'Length_of_Stay', 'Alvarado_Score',
    'Paedriatic_Appendicitis_Score', 'Appendix_Diameter', 'Body_Temperature',
    'WBC_Count', 'Neutrophil_Percentage', 'Segmented_Neutrophils', 'RBC_Count',
    'Hemoglobin', 'RDW', 'Thrombocyte_Count', 'CRP'
]
categorical_features = [
    'Sex', 'Appendix_on_US', 'Migratory_Pain', 'Lower_Right_Abd_Pain',
    'Contralateral_Rebound_Tenderness', 'Coughing_Pain', 'Nausea',
    'Loss_of_Appetite', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine',
    'WBC_in_Urine', 'Dysuria', 'Stool', 'Peritonitis', 'Psoas_Sign',
    'Ipsilateral_Rebound_Tenderness', 'US_Performed', 'Free_Fluids',
    'Appendix_Wall_Layers', 'Target_Sign', 'Appendicolith', 'Perfusion',
    'Perforation', 'Surrounding_Tissue_Reaction', 'Appendicular_Abscess',
    'Abscess_Location', 'Pathological_Lymph_Nodes', 'Lymph_Nodes_Location',
    'Bowel_Wall_Thickening', 'Conglomerate_of_Bowel_Loops', 'Ileus',
    'Coprostasis', 'Meteorism', 'Enteritis', 'Gynecological_Findings',
    'Management', 'Severity'
]

# Vérifier que toutes les colonnes existent
for col in numeric_features + categorical_features:
    if col not in X.columns:
        print(f"Attention : colonne {col} non trouvée dans les données.")

# Créer le préprocesseur
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_features)
])

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------
# Modèles avec GridSearchCV dans un pipeline
# -------------------------------

# 1. Random Forest
print("\nRecherche d'hyperparamètres pour RandomForest...")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("Meilleurs paramètres pour RandomForest:", grid_rf.best_params_)
print("F1 score sur test:", f1_score(y_test, y_pred_rf))

# 2. LightGBM
print("\nRecherche d'hyperparamètres pour LightGBM...")
lgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1))
])
param_grid_lgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [-1, 5, 10],
    'classifier__learning_rate': [0.05, 0.1, 0.2]
}
grid_lgb = GridSearchCV(lgb_pipeline, param_grid_lgb, cv=5, scoring='f1', n_jobs=-1)
grid_lgb.fit(X_train, y_train)
best_lgb = grid_lgb.best_estimator_
y_pred_lgb = best_lgb.predict(X_test)
print("Meilleurs paramètres pour LightGBM:", grid_lgb.best_params_)
print("F1 score sur test:", f1_score(y_test, y_pred_lgb))

# 3. CatBoost
print("\nRecherche d'hyperparamètres pour CatBoost...")
cat_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', cb.CatBoostClassifier(verbose=0, random_state=42,early_stopping_rounds=50))
])
param_grid_cat = {
    'classifier__iterations': [100, 200, 300],
    'classifier__depth': [4, 6, 8],
    'classifier__learning_rate': [0.05, 0.1, 0.2]
}
grid_cat = GridSearchCV(cat_pipeline, param_grid_cat, cv=5, scoring='f1', n_jobs=-1)
grid_cat.fit(X_train, y_train)
best_cat = grid_cat.best_estimator_
y_pred_cat = best_cat.predict(X_test)
print("Meilleurs paramètres pour CatBoost:", grid_cat.best_params_)
print("F1 score sur test:", f1_score(y_test, y_pred_cat))

# -------------------------------
# Sélection du meilleur modèle (CatBoost semble meilleur)
# -------------------------------
best_model = best_cat  # ou best_rf, best_lgb selon les scores

print("\nMeilleur modèle: CatBoost")
print("Évaluation finale sur test:")
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

# Sauvegarder le pipeline complet
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/pipeline.pkl")
print("Pipeline sauvegardé dans models/pipeline.pkl")

# Sauvegarder la liste des colonnes (ordre attendu par le préprocesseur)
feature_columns = numeric_features + categorical_features
joblib.dump(feature_columns, "models/columns.pkl")
print("Liste des colonnes sauvegardée dans models/columns.pkl")

# Extraire le classifieur du pipeline
#classifier = best_model.named_steps['classifier']

# Créer et sauvegarder l'explainer SHAP
#explainer = shap.TreeExplainer(classifier)
#joblib.dump(explainer, "models/explainer.pkl")

# Extraire les composants du pipeline
preprocessor = best_model.named_steps["preprocessor"]
classifier = best_model.named_steps["classifier"]

# Transformer les données
X_background = preprocessor.transform(X_train)

# Créer l'explainer SHAP
explainer = shap.Explainer(classifier, X_background)

joblib.dump(explainer, "models/explainer.pkl")

print("Explainer SHAP sauvegardé dans models/explainer.pkl")

print("Fin du script")