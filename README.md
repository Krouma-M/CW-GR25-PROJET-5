# 🏥 Diagnostic d'Appendicite Pédiatrique — ML Pipeline

![CI](https://github.com/Krouma-M/CW-GR25-PROJET-5/actions/workflows/ci.yml/badge.svg?branch=develop)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Projet de classification pour le diagnostic de l'appendicite chez l'enfant avec ML explicable (SHAP).

**Coding Week — 09 au 15 Mars 2026 | École Centrale Casablanca**  
Dataset : Regensburg Pediatric Appendicitis (UCI id=938) — 782 patients | 53 features | Cible : `Diagnosis`

---

## 📋 Description

Ce projet développe un modèle de Machine Learning entraîné sur le dataset **Regensburg Pediatric Appendicitis** (UCI) pour prédire la probabilité d'appendicite chez un enfant à partir de ses symptômes et résultats cliniques.

Compte tenu du contexte médical critique, chaque prédiction est accompagnée d'une **explication SHAP** indiquant quels facteurs ont influencé la décision du modèle, garantissant des prédictions transparentes, fiables et interprétables.

**Objectifs :**
- Comprendre les données grâce à une analyse exploratoire approfondie (EDA)
- Développer un modèle ML robuste et explicable
- Assurer la transparence des prédictions via SHAP
- Créer une interface intuitive avec Streamlit
- Suivre les bonnes pratiques de développement logiciel (GitHub, CI/CD)

---

## 📁 Structure du projet

```
CW-GR25-PROJET-5/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Pipeline CI/CD (GitHub Actions)
├── app/
│   ├── explications/                 # Explications SHAP pour l'interface
│   ├── images/                       # Assets visuels de l'application
│   └── app.py                        # Interface Streamlit
├── data/
│   └── raw/
│       └── appendicitis.csv          # Dataset brut
├── models/
│   ├── best_model.pkl                # Meilleur modèle sélectionné (CatBoost)
│   ├── CatBoost.pkl                  # Modèle CatBoost
│   ├── LightGBM.pkl                  # Modèle LightGBM
│   ├── Random Forest.pkl             # Modèle Random Forest
│   ├── columns.pkl                   # Colonnes du dataset
│   ├── pipeline.pkl                  # Pipeline de prétraitement
│   └── explainer.pkl                 # Explainer SHAP
├── notebooks/
│   └── eda.ipynb                     # Analyse exploratoire des données
├── reports/
│   └── figures/
│       ├── eda/                      # Graphiques EDA
│       └── shap/                     # Graphiques SHAP
├── src/
│   ├── data_processing.py            # Prétraitement et optimisation mémoire
│   ├── download_data.py              # Téléchargement du dataset UCI
│   ├── evaluate.py                   # Évaluation et métriques
│   ├── shap1.py                      # Explicabilité SHAP
│   └── train_model.py                # Entraînement des modèles ML
├── tests/
│   ├── test_data_processing.py       # Tests du prétraitement
│   └── test_model.py                 # Tests du modèle
├── create_explainer.py               # Script de création de l'explainer SHAP
├── Tableau_Jira_du_projet.csv        # Suivi des tâches Jira
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prérequis

#### 🐍 Python 3.10+

**Windows :**
```powershell
winget install Python.Python.3.10
```
**Mac :**
```bash
brew install python@3.10
```
**Linux (Ubuntu/Debian) :**
```bash
sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip
```
Vérifier : `python --version`

---

#### 🔧 Git

**Windows :** `winget install Git.Git`  
**Mac :** `brew install git`  
**Linux :** `sudo apt install git`  

Vérifier : `git --version`

---

#### 💻 VS Code *(recommandé)*

**Windows :** `winget install Microsoft.VisualStudioCode`  
**Mac :** `brew install --cask visual-studio-code`  
**Linux :** `sudo snap install code --classic`

---

### Mise en place

**1. Cloner le repository**
```bash
git clone https://github.com/CW-GR25-PROJET-5.git
cd CW-GR25-PROJET-5
```

**2. Créer et activer un environnement virtuel**
```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

---

## 🚀 Utilisation

**Entraîner le modèle :**
```bash
python src/train_model.py
```

**Lancer l'application :**
```bash
streamlit run app/app.py
    ou
avec python directement : python3 -m streamlit run app.py
```
L'application s'ouvre automatiquement sur `http://localhost:8501`

**Lancer les tests :**
```bash
python -m pytest tests/test_data_processing.py -v
python -m pytest tests/test_model.py -v
```

---

## 🔄 Pipeline ML

### 1. `data_processing.py` — Prétraitement

Prépare les données brutes pour l'entraînement.

**Fonctions :**
- `load_data(filepath)` — charge le CSV
- `preprocess_data(df)` — nettoie, encode, impute, split
- `optimize_memory(df)` — réduit la mémoire (float64→float32, int64→int32)
- `get_feature_names(X)` — retourne les noms des colonnes
- `get_data_summary(df)` — retourne les statistiques du dataset

**Détails :**
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

### 2. `train_model.py` — Entraînement

Entraîne 3 modèles avec optimisation **GridSearchCV** (cv=5, scoring="f1", n_jobs=-1).

**Modèles :** Random Forest, LightGBM, CatBoost

**Meilleurs hyperparamètres trouvés :**

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

### 3. `evaluate.py` — Évaluation

Évalue les modèles et sélectionne le meilleur automatiquement.

```bash
python src/evaluate.py
```

**Résultats :**

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Random Forest | 93.59% | 92.06% | 92.06% | 92.06% | **98.66%** |
| LightGBM | **96.15%** | **95.24%** | **95.24%** | **95.24%** | 96.96% |
| **CatBoost** ✅ | 92.95% | 88.24% | 95.24% | 91.60% | 97.92% |

**Meilleur modèle : CatBoost** — sélectionné par départage sur le Recall puis le ROC-AUC  
Sauvegardé automatiquement dans `models/best_model.pkl`

**Logique de sélection :**
- Meilleur Recall global : **0.9524** — ex-aequo entre LightGBM et CatBoost
- Départage par ROC-AUC : CatBoost (**0.9792**) > LightGBM (0.9696)
- ✅ **CatBoost sélectionné**

---

## 🤖 Choix du modèle — Justification CatBoost

> ⚠️ Le **Recall** est la métrique prioritaire en médecine : rater une appendicite expose l'enfant à des complications graves (perforation, péritonite).

**1. Recall maximal (95.24%)** — CatBoost et LightGBM sont ex-aequo sur le Recall, la métrique la plus critique cliniquement. Les deux modèles détectent 95.24% de tous les vrais cas d'appendicite.

**2. Départage par ROC-AUC** — En cas d'égalité sur le Recall, on utilise la ROC-AUC comme critère secondaire. CatBoost obtient **97.92%** contre 96.96% pour LightGBM, ce qui traduit une meilleure capacité de discrimination globale entre les deux classes.

**3. Avantages techniques** — CatBoost est naturellement résistant à l'overfitting, gère efficacement les variables mixtes sans encodage manuel, et nécessite peu de réglages d'hyperparamètres.

> **Conclusion :** Bien que LightGBM domine sur accuracy, precision et F1, CatBoost est retenu car il égale le meilleur Recall tout en offrant une meilleure discrimination globale (ROC-AUC). Dans un contexte médical où manquer un diagnostic est inacceptable, ce critère prime sur les autres.

---

## 📊 Métriques — Rappel

| Métrique | Définition | Importance |
|----------|-----------|-----------|
| **Accuracy** | % de toutes les prédictions correctes | Globale |
| **Precision** | Parmi les cas prédits positifs, combien sont vrais ? | Éviter les fausses alarmes |
| **Recall** | Parmi tous les vrais cas, combien sont détectés ? | ⭐ Critique en médecine |
| **F1-Score** | Moyenne harmonique entre Precision et Recall | Équilibre |
| **ROC-AUC** | Capacité à distinguer les classes (1.0 = parfait) | Discrimination |

---

## 🧠 Explicabilité SHAP

```bash
python src/shap1.py
```

**Top 10 features les plus importantes :**

| Rang | Feature | Score SHAP |
|------|---------|-----------|
| 1 | Appendix_on_US | 1.9749 |
| 2 | Appendix_Diameter | 1.2907 |
| 3 | Ipsilateral_Rebound_Tenderness | 0.5618 |
| 4 | WBC_Count | 0.3521 |
| 5 | CRP | 0.3305 |
| 6 | Peritonitis | 0.3014 |
| 7 | Alvarado_Score | 0.2840 |
| 8 | Neutrophilia | 0.2793 |
| 9 | Surrounding_Tissue_Reaction | 0.1885 |
| 10 | Free_Fluids | 0.1795 |

Graphiques sauvegardés dans `reports/figures/shap/`

---

## 🧪 Tests

### `test_data_processing.py` — 6 tests

| Test | Description |
|------|-------------|
| test_load_data | Vérifie le chargement des données |
| test_no_missing_values | Vérifie 0 valeurs manquantes après imputation |
| test_split_ratio | Vérifie le split 80/20 |
| test_data_summary | Vérifie get_data_summary() |
| test_optimize_memory | Vérifie la réduction mémoire |
| test_load_best_model | Vérifie le chargement de best_model.pkl |

### `test_model.py` — 5 tests

| Test | Description |
|------|-------------|
| test_random_forest_gridsearch | RF avec GridSearch, Recall ≥ 80% |
| test_lightgbm_gridsearch | LightGBM avec GridSearch, Recall ≥ 80% |
| test_catboost_gridsearch | CatBoost avec GridSearch, F1 ≥ 80% |
| test_data_loaded | Vérifie les données chargées |
| test_best_model_exists | Vérifie best_model.pkl |

---

## ❓ Questions importantes

**Le dataset était-il équilibré ?**  
Oui, le dataset est approximativement équilibré (~50% appendicite, ~50% pas d'appendicite). Aucune technique de rééquilibrage (SMOTE, undersampling) n'a été nécessaire.

**Quel modèle a le mieux performé ?**  
LightGBM domine sur accuracy (96.15%), precision, F1 et ROC-AUC. Cependant, LightGBM et CatBoost sont ex-aequo sur le Recall (95.24%), métrique prioritaire en médecine. CatBoost est finalement sélectionné car il remporte le départage par ROC-AUC (97.92% vs 96.96%).

**Quelles features ont le plus influencé les prédictions (SHAP) ?**  
D'après l'analyse SHAP, les features les plus importantes sont : Appendix_on_US, Appendix_Diameter, Ipsilateral_Rebound_Tenderness, WBC_Count et CRP.

---

## 🧠 Prompt Engineering

**Tâche choisie :** Génération de la fonction `optimize_memory(df)`

**Prompt utilisé :**
> *"Écris une fonction Python appelée optimize_memory(df) qui prend un DataFrame pandas en entrée et réduit sa consommation mémoire en convertissant les colonnes float64 en float32 et int64 en int32. Affiche la mémoire avant et après optimisation en MB ainsi que le pourcentage de réduction."*

**Efficacité :** Très efficace car le prompt était précis sur le nom de la fonction, les types à convertir et le format d'affichage attendu. Un prompt plus vague comme *"optimise la mémoire"* aurait donné un résultat moins ciblé.

---

## 👥 Équipe

| Membres | Rôles |
|--------|------|
| Mamady KOUROUMA | Interface Streamlit & Tests automatisés |
| Appolinaire YAMEOGO | EDA, CI/CD & Documentation |
| Mamoudou SISSOKO | Interface Streamlit & Gestion Tableau Jira |
| Islam Mohamed OUJBAIR | Prétraitement des données & Modélisation ML |
| Mamadou BERTHE | Prétraitement des données & Modélisation ML |

---

## 📊 Dataset

[Regensburg Pediatric Appendicitis — UCI ML Repository](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis)
