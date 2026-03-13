# 🏥 Diagnostic d'Appendicite Pédiatrique avec ML Explicable

![CI](https://github.com/Krouma-M/CW-GR25-PROJET-5/actions/workflows/ci.yml/badge.svg?branch=develop)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> Application d'aide à la décision clinique permettant aux pédiatres de diagnostiquer l'appendicite chez l'enfant grâce au Machine Learning et à l'explicabilité SHAP.

---

## 📋 Description du projet

Ce projet développe un modèle de Machine Learning entraîné sur le dataset **Regensburg Pediatric Appendicitis** (UCI) pour prédire la probabilité d'appendicite chez un enfant à partir de ses symptômes et résultats cliniques.

Compte tenu du contexte médical critique, chaque prédiction est accompagnée d'une **explication SHAP** qui indique quels facteurs médicaux ont influencé la décision du modèle, garantissant ainsi des prédictions transparentes, fiables et interprétables.

---

## 🎯 Objectifs

- Comprendre les données grâce à une analyse exploratoire approfondie (EDA)
- Développer un modèle ML robuste et explicable
- Assurer la transparence des prédictions via SHAP
- Créer une interface intuitive avec Streamlit
- Suivre les bonnes pratiques de développement logiciel (GitHub, CI/CD)

---

## 📁 Structure du projet

```
CW-GR25-PROJET-5/
├── data/                        # Dataset brut
├── notebooks/
│   └── eda.ipynb                # Analyse exploratoire des données
├── src/
│   ├── data_processing.py       # Prétraitement et optimisation mémoire
│   ├── train_model.py           # Entraînement des modèles ML
│   ├── evaluate.py              # Évaluation et métriques
│   └── shap.py                  # Explicabilité SHAP
├── app/
│   └── app.py                   # Interface Streamlit
├── tests/
│   ├── test_data_processing.py  # Tests du prétraitement
│   └── test_model.py            # Tests du modèle
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline CI/CD (GitHub Actions)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prérequis

Avant de commencer, installez les outils suivants :

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
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

Vérifier l'installation :
```bash
python --version
```

---

#### 🔧 Git

**Windows :**
```powershell
winget install Git.Git
```

**Mac :**
```bash
brew install git
```

**Linux (Ubuntu/Debian) :**
```bash
sudo apt install git
```

Vérifier l'installation :
```bash
git --version
```

---

#### 💻 VS Code *(recommandé)*

**Windows :**
```powershell
winget install Microsoft.VisualStudioCode
```

**Mac :**
```bash
brew install --cask visual-studio-code
```

**Linux (Ubuntu/Debian) :**
```bash
sudo snap install code --classic
```

---

### 1. Cloner le repository

```bash
git clone https://github.com/Krouma-M/CW-GR25-PROJET-5.git
cd CW-GR25-PROJET-5
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Entraîner le modèle

```bash
python src/train_model.py
```

### Lancer l'application

```bash
streamlit run app/app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

### Lancer les tests automatisés

```bash
pytest tests/ -v
```

---

## 🤖 Modèles utilisés

Trois modèles ont été entraînés, évalués et comparés sur les métriques suivantes :

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Random Forest | 0.974 | 0.984 | 0.994 | — | — |
| LightGBM | 0.981 | 0.984 | 0.980 | — | — |
| **CatBoost** ✅ | **0.987** | **0.984** | **0.991** | — | — |

### ✍️ Justification du choix — CatBoost

Dans le cadre d'un système d'aide au diagnostic médical pédiatrique, le choix du modèle final repose sur une analyse rigoureuse des métriques de performance, en accordant une attention particulière au **recall** et à l'**accuracy globale**.

Le modèle **CatBoost** a été retenu pour les raisons suivantes :

**1. Accuracy la plus élevée (0.987)** — CatBoost classe correctement 98.7% des cas, toutes catégories confondues, témoignant d'une excellente capacité de généralisation.

**2. Recall cliniquement sûr (0.991)** — Dans un contexte médical, le recall est la métrique la plus critique : il mesure la capacité du modèle à ne manquer aucun vrai cas d'appendicite. Manquer un tel diagnostic peut exposer l'enfant à des complications graves comme une perforation ou une péritonite. CatBoost affiche le deuxième recall le plus élevé, à seulement 0.003 point derrière Random Forest, tout en offrant une bien meilleure accuracy globale.

**3. Precision identique (0.984)** — Les trois modèles étant à égalité sur ce critère, c'est la combinaison accuracy + recall qui fait pencher la balance en faveur de CatBoost.

**4. Avantages techniques** — CatBoost est naturellement résistant à l'overfitting, gère efficacement les variables mixtes (numériques et catégorielles) sans encodage manuel, et nécessite peu de réglages d'hyperparamètres.

> **Conclusion :** CatBoost représente le meilleur compromis entre précision globale, sensibilité clinique et robustesse technique, ce qui en fait le choix le plus adapté pour un outil d'aide au diagnostic pédiatrique où la fiabilité est non négociable.

---

## 📊 Explication des métriques

| Métrique | Définition | Importance |
|----------|-----------|-----------|
| **Accuracy** | % de toutes les prédictions correctes | Globale |
| **Precision** | Parmi les cas prédits positifs, combien sont vrais ? | Éviter les fausses alarmes |
| **Recall** | Parmi tous les vrais cas, combien sont détectés ? | ⭐ Critique en médecine |
| **F1-Score** | Moyenne harmonique entre Precision et Recall | Équilibre |
| **ROC-AUC** | Capacité à distinguer les classes (1.0 = parfait) | Discrimination |

---

## ❓ Questions critiques

**Le dataset était-il équilibré ?**

Oui, le dataset est approximativement équilibré (~50% appendicite, ~50% pas d'appendicite). Aucune technique de rééquilibrage (SMOTE, undersampling) n'a donc été nécessaire.

**Quel modèle a le mieux performé ?**

CatBoost, avec une accuracy de 0.987 et un recall de 0.991, offre les meilleures performances globales.

**Quelles features ont le plus influencé les prédictions (SHAP) ?**

D'après l'analyse SHAP, les features les plus importantes sont :
1. Alvarado Score
2. Paediatric Appendicitis Score
3. CRP (Protéine C-Réactive)
4. Appendix Diameter
5. Neutrophil Percentage

---

## 🧠 Prompt Engineering

**Tâche choisie :** Génération de la fonction `optimize_memory(df)`

**Prompt utilisé :**
> *"Écris une fonction Python appelée optimize_memory(df) qui prend un DataFrame pandas en entrée et réduit sa consommation mémoire en convertissant les colonnes float64 en float32 et int64 en int32. Affiche la mémoire avant et après optimisation en MB ainsi que le pourcentage de réduction."*

**Résultat :** La fonction générée était fonctionnelle et a permis une réduction significative de la mémoire utilisée.

**Efficacité du prompt :** Très efficace car le prompt était précis sur le nom de la fonction, les types à convertir, et le format d'affichage attendu. Un prompt plus vague comme *"optimise la mémoire"* aurait donné un résultat moins ciblé.

---

## 👥 Équipe

| Membre | Rôle |
|--------|------|
| Mamady KOUROUMA | Interface Streamlit & Tests automatisés |
| Appolinaire YAMEOGO | EDA, CI/CD & Documentation |
| Mamoudou SISSOKO | Interface Streamlit & Gestion Tableau Jira |
| Islam Mohamed OUJBAIR | Prétraitement des données & Modélisation ML |
| Mamadou BRETHE | Prétraitement des données & Modélisation ML |

---

## 📊 Dataset

[Regensburg Pediatric Appendicitis — UCI ML Repository](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis)

---

*Coding Week — 09 au 15 Mars 2026 | École Centrale Casablanca*