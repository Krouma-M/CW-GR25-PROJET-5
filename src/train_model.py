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

print("Script lancé...")

# Charger les données brutes (sans aucun preprocessing)
df = pd.read_csv("data/raw/appendicitis.csv")

# Supprimer les lignes où la cible est manquante
df = df.dropna(subset=["Diagnosis"])

# Séparer X et y (y est la cible)
target_col = "Diagnosis"
y = df[target_col].values
X = df.drop(columns=[target_col])

# Encoder la cible en 0/1 (appendicitis / no appendicitis)
le_target = LabelEncoder()
y = le_target.fit_transform(y)
print("Classes:", le_target.classes_)

# Identifier les colonnes numériques et catégorielles dans les données brutes
# On utilise les types d'origine avant tout encodage
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("Colonnes numériques :", numeric_features)
print("Colonnes catégorielles :", categorical_features)

# Créer le préprocesseur
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),   # imputation des valeurs manquantes numériques
        ('scaler', StandardScaler())                     # normalisation
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # remplacer NaN par 'missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # one-hot encoding
    ]), categorical_features)
])

# Division train/test (avant tout traitement pour éviter le data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Définir les modèles à tester
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
    'CatBoost': cb.CatBoostClassifier(verbose=0, random_state=42)
}

# Grilles d'hyperparamètres (simplifiées pour l'exemple)
param_grids = {
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10],
    },
    'LightGBM': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
    },
    'CatBoost': {
        'classifier__iterations': [100, 200],
        'classifier__depth': [4, 6],
    }
}
joblib.dump(X.columns.tolist(), "models/columns.pkl")
best_estimators = {}

for name in models:
    print(f"\nRecherche d'hyperparamètres pour {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', models[name])
    ])
    
    grid = GridSearchCV(
        pipeline, 
        param_grids[name], 
        cv=5, 
        scoring='f1', 
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    
    best_estimators[name] = grid.best_estimator_
    y_pred = grid.predict(X_test)
    print(f"Meilleurs paramètres pour {name}: {grid.best_params_}")
    print(f"F1 score sur test: {f1_score(y_test, y_pred):.3f}")

# Choisir le meilleur modèle (par exemple celui avec le meilleur F1 sur validation)
# Pour simplifier, on prend CatBoost (car vos précédents résultats montraient qu'il était bon)
best_model_name = 'CatBoost'  # ou vous pouvez comparer les scores
best_pipeline = best_estimators[best_model_name]

print(f"\nMeilleur modèle: {best_model_name}")
print("Évaluation finale sur test:")
y_pred = best_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, best_pipeline.predict_proba(X_test)[:,1]))

# Sauvegarder le pipeline complet (préprocesseur + modèle)
joblib.dump(best_pipeline, "models/pipeline.pkl")
print("Pipeline sauvegardé dans models/pipeline.pkl")

# Sauvegarder aussi l'encodeur de la cible (pour décoder les prédictions si besoin)
joblib.dump(le_target, "models/label_encoder_target.pkl")
print("Ordre des colonnes:", X.columns.tolist())