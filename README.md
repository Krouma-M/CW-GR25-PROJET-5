# CW-GR25-PROJET-5
Développer une application d'aide à la décision clinique visant à aider les pédiatres à diagnostiquer avec précision l'appendicite chez l'enfant, en utilisant les symptômes et les résultats des tests cliniques. Compte tenu du contexte médical critique, le modèle doit fournir des prédictions transparentes, fiables et explicables.
# Pediatric Appendicitis Diagnosis

Projet de classification pour le diagnostic de l'appendicite chez l'enfant avec ML explicable (SHAP).

## Structure
- `data/` : données brutes et traitées (non versionnées)
- `notebooks/` : analyses exploratoires
- `src/` : code source du pipeline ML
- `app/` : application web (Streamlit/Flask)
- `tests/` : tests unitaires
- `.github/workflows/` : CI/CD

## Installation
télécharger les données:



# 🏥 Diagnostic d'Appendicite Pédiatrique — ML Pipeline

Projet 5 — Semaine de codage Mars 2026, Centrale Casablanca  
Dataset : Regensburg Pediatric Appendicitis (UCI id=938)  
782 patients | 53 features | Cible : Diagnosis (appendicitis / no appendicitis)

---

## 📁 Structure du projet
```
CW-GR25-PROJET-5/
├── data/
│   └── raw/
│       └── appendicitis.csv
├── models/
│   ├── Random Forest.pkl
│   ├── LightGBM.pkl
│   ├── CatBoost.pkl
│   └── best_model.pkl
├── src/
│   ├── data_processing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── shap1.py
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── figures/
│   └── shap/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation
```bash
git clone https://github.com/CW-GR25-PROJET-5.git
cd CW-GR25-PROJET-5
python -m venv .env
.env\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## 🔄 Pipeline ML

### 1. data_processing.py

Prépare les données brutes pour l'entraînement.

**Fonctions :**
- `load_data(filepath)` — charge le CSV
- `preprocess_data(df)` — nettoie, encode, impute, split
- `optimize_memory(df)` — réduit la mémoire (float64→float32, int64→int32)
- `get_feature_names(X)` — retourne les noms des colonnes
- `get_data_summary(df)` — retourne les statistiques du dataset

**Détails du prétraitement :**
- Suppression des lignes sans diagnostic
- Encodage des variables catégorielles (LabelEncoder)
- Imputation des valeurs manquantes (médiane)
- Optimisation mémoire : 0.33 MB → 0.16 MB (gain 52%)
- Split train/test : 80/20 avec stratify
```bash
python src/data_processing.py
```

**Résultat :**
```
Training set : 624 patients
Test set     : 156 patients
Features     : 55 colonnes
Missing      : 0 valeurs manquantes
```

---

### 2. train_model.py

Entraîne 3 modèles avec optimisation GridSearchCV.

**Modèles :**
- Random Forest
- LightGBM  
- CatBoost

**GridSearchCV :** cv=5, scoring="f1", n_jobs=-1

**Meilleurs paramètres trouvés :**

| Modèle | Paramètres |
|--------|-----------|
| Random Forest | n_estimators=300, max_depth=None, min_samples_split=5 |
| LightGBM | n_estimators=200, max_depth=5, learning_rate=0.1 |
| CatBoost | iterations=100, depth=6, learning_rate=0.2 |
```bash
python src/train_model.py
```

**Résultat :** 3 fichiers `.pkl` sauvegardés dans `models/`

---

### 3. evaluate.py

Évalue les modèles et sélectionne le meilleur.

**Métriques calculées :** Accuracy, Precision, Recall, F1, ROC-AUC
```bash
python src/evaluate.py
```

**Résultats :**

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Random Forest | 97.44% | 98.36% | 95.24% | 96.77% | **99.40%** |
| LightGBM | 98.08% | 98.39% | 96.83% | 97.60% | 98.81% |
| **CatBoost** | **98.72%** | **98.41%** | **98.41%** | **98.41%** | 99.08% |

**Meilleur modèle : CatBoost** — gagne sur 4/5 métriques  
Sauvegardé automatiquement dans `models/best_model.pkl`

> ⚠️ Le **Recall** est la métrique prioritaire en médecine :  
> rater une appendicite = danger de mort pour l'enfant.

---

## 🧪 Tests

### test_data_processing.py — 6 tests

| Test | Description |
|------|-------------|
| test_load_data | Vérifie le chargement des données |
| test_no_missing_values | Vérifie 0 valeurs manquantes après imputation |
| test_split_ratio | Vérifie le split 80/20 |
| test_data_summary | Vérifie get_data_summary() |
| test_optimize_memory | Vérifie la réduction mémoire |
| test_load_best_model | Vérifie le chargement de best_model.pkl |

### test_model.py — 5 tests

| Test | Description |
|------|-------------|
| test_random_forest_gridsearch | RF avec GridSearch, Recall ≥ 80% |
| test_lightgbm_gridsearch | LightGBM avec GridSearch, Recall ≥ 80% |
| test_catboost_gridsearch | CatBoost avec GridSearch, F1 ≥ 80% |
| test_data_loaded | Vérifie les données chargées |
| test_best_model_exists | Vérifie best_model.pkl |
```bash
python -m pytest tests/test_data_processing.py -v
python -m pytest tests/test_model.py -v
```

---

##  Explicabilité SHAP
```bash
python src/shap1.py
```

**Top 10 features importantes :**

| Rang | Feature | Score SHAP |
|------|---------|-----------|
| 1 | Appendix_on_US | 2.5888 |
| 2 | Management | 2.1934 |
| 3 | Appendix_Diameter | 1.5836 |
| 4 | WBC_Count | 0.4042 |
| 5 | Alvarado_Score | 0.3680 |

Graphiques sauvegardés dans `figures/shap/`

---

##  Équipe

Projet réalisé dans le cadre de la semaine de codage  
Centrale Casablanca — Mars 2026