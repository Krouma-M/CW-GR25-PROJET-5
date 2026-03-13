# 🏥 Diagnostic d'Appendicite Pédiatrique avec ML Explicable

CW-GR25-PROJET-5

PROJET : Pediatric Appendicitis Diagnosis

Développement  d'une application d'aide à la décision clinique visant à aider les pédiatres à diagnostiquer avec précision l'appendicite chez l'enfant, en utilisant les symptômes et les résultats des tests cliniques. Compte tenu du contexte médical critique, le modèle doit fournir des prédictions transparentes, fiables et explicables.C'est pourquoi nous utiliserons le  Machine Learning à l'explicabilité SHAP.


---

## 📋 DESCRIPTION DU PROJET

Ce projet développe un modèle de Machine Learning entraîné sur le dataset Regensburg Pediatric Appendicitis (UCI) pour prédire la probabilité d'appendicite chez un enfant à partir de ses symptômes et résultats cliniques.

Chaque prédiction est accompagnée d'une explication SHAP qui indique quels facteurs médicaux ont influencé la décision du modèle.

---

## 🎯 OBJECTIFS

-Comprendre la corrélation entre les données à l'aide de      l'analyse exploratoire
- Développer un modèle ML robuste et explicable
- Assurer la transparence via SHAP
- Créer une interface intuitive avec Streamlit
- Suivre les bonnes pratiques de développement (GitHub, CI/CD)

---

## 📁 STRUCTURE DU PROJET
```
CW-GR25-PROJET-5/
├── data/                    # Dataset
├── notebooks/
│   └── eda.ipynb            # Analyse exploratoire
├── src/
│   ├── data_processing.py   # Prétraitement et optimisation mémoire
│   ├── train_model.py       # Entraînement des modèles
│   ├── evaluate.py          # Évaluation et métriques
│   └── shap.py              # Explicabilité SHAP
├── app/
│   └── app.py               # Interface Streamlit
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── .github/
│   └── workflows/
│       └── ci.yml           # Pipeline CI/CD
├── requirements.txt
└── README.md
```

---

## ⚙️ INSTALLATION

### 1. Cloner le repo
```bash
git clone https://github.com/Krouma-M/CW-GR25-PROJET-5.git
cd CW-GR25-PROJET-5
```

### 2. CREER UN ENVIRONNEMENT VIRTUEL
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. INSTALLER LES DEPENDANCES
```bash
pip install -r requirements.txt
```

---

## 🚀 UTILISATION

### ENTRAINER LE MODELE
```bash
python src/train_model.py
```

### LANCER L'APPLICATION
```bash
streamlit run app/app.py
```

L'application s'ouvre sur `http://localhost:8501`

### LANCER LES TESTS
```bash
pytest tests/ -v
```

---

## 🤖 MODELES UTILISES

Trois modèles ont été entraînés et comparés :

| Modèle | Accuracy |Precision  | Recall | F1 |ROC-AUC |
|--------|---------|----------|----------|
| Random Forest | 0.974 | 0.984 | 0.9940 |
| LightGBM | 0.981 | 0.984 | 0.98 |
| CatBoost | 0.987 | 0.984 | 0.9908 |

## ✍️ Justification du choix de CatBoost

---

> ### Modèle retenu : CatBoost
>
> Dans le cadre d'un système d'aide au diagnostic médical pédiatrique, le choix du modèle final repose sur une analyse rigoureuse de l'ensemble des métriques de performance, en accordant une attention particulière au **recall** et à l'**accuracy globale**.
>
> Trois modèles ont été entraînés et évalués : Random Forest, LightGBM et CatBoost. Les résultats obtenus sont les suivants :
>
> | Modèle | Accuracy | Precision | Recall |
> |--------|----------|-----------|--------|
> | Random Forest | 0.974 | 0.984 | 0.9940 |
> | LightGBM | 0.981 | 0.984 | 0.9800 |
> | **CatBoost** | **0.987** | **0.984** | **0.9908** |
>
> Le modèle **CatBoost** a été retenu comme modèle final pour les raisons suivantes.
>
> Premièrement, il affiche l'**accuracy la plus élevée** avec **0.987**, ce qui signifie qu'il classe correctement 98.7% des cas, toutes catégories confondues. Cette performance globale supérieure témoigne d'une meilleure capacité de généralisation sur l'ensemble du dataset.
>
> Deuxièmement, son **recall de 0.9908** est le deuxième plus élevé, à seulement 0.003 point derrière Random Forest. Dans un contexte médical, le recall est crucial car il mesure la capacité du modèle à ne manquer aucun vrai cas d'appendicite — manquer un tel cas pouvant exposer l'enfant à des complications graves comme une perforation ou une péritonite. CatBoost garantit ainsi un niveau de sécurité clinique très élevé.
>
> Troisièmement, les trois modèles affichent une **precision identique de 0.984**, ce critère ne permet donc pas de les différencier. C'est donc la combinaison **accuracy + recall** qui fait pencher la balance en faveur de CatBoost.
>
> Enfin, CatBoost présente des avantages techniques notables : il est naturellement résistant à l'overfitting, gère efficacement les variables mixtes (numériques et catégorielles), et nécessite peu de réglages d'hyperparamètres, ce qui renforce la fiabilité et la robustesse du modèle en production.
>
> **En conclusion, CatBoost représente le meilleur compromis entre précision globale, sensibilité clinique et robustesse technique**, ce qui en fait le choix le plus adapté pour un outil d'aide au diagnostic pédiatrique où la fiabilité est non négociable.

---

> 💡 **Conseil :** Place cette rédaction dans la section **"🤖 Modèles utilisés"** de ton README, juste en dessous du tableau comparatif !

ccuracy (Précision globale)
C'est le pourcentage de toutes les prédictions correctes (appendicites ET non-appendicites bien classées) sur l'ensemble du dataset.

Ex : CatBoost à 0.987 signifie qu'il se trompe seulement 1.3% du temps au total.


Precision (Précision positive)
Parmi tous les cas que le modèle a dit "c'est une appendicite", combien étaient vraiment des appendicites ?

C'est la question : "Quand tu sonnes l'alarme, as-tu raison ?"
Les 3 modèles sont à 0.984 — quasi identiques.


Recall (Rappel / Sensibilité)
Parmi tous les vrais cas d'appendicite, combien le modèle en a-t-il détectés ?

C'est la question : "Est-ce que tu rates des malades ?"
C'est la métrique LA PLUS IMPORTANTE en médecine.


F1-Score
C'est la moyenne harmonique entre Precision et Recall. Il équilibre les deux.

Utile quand on veut un seul chiffre résumant les deux.


ROC-AUC
Mesure la capacité du modèle à distinguer les cas positifs des négatifs, quelle que soit la décision seuil.

Un score de 1.0 = parfait. Un score de 0.5 = aléatoire.

**Modèle retenu : CatBoost** — car il offre le meilleur compromis, surtout son pourcentage de recall  
entre performance et vitesse de prédiction.

---

## ❓ QUESTIONS IMPORTANTES

**Le dataset était-il équilibré ?**  
Oui, le dataset est approximativement équilibré (~50% appendicite, 
~50% pas d'appendicite). Aucune technique de rééquilibrage n'a été 
nécessaire.

**Quel modèle a le mieux performé ?**  
CatBoost avec un ROC-AUC de 0.XX. (à compléter après les résultats)

**Quelles features ont le plus influencé les prédictions ?**  
D'après les résultats SHAP, les features les plus importantes sont :
1. CRP (Protéine C-Réactive)
2. Leucocytes
3. Température
4. Durée de la douleur

(à compléter après l'analyse SHAP)

---

## 🧠 PROMPT ENGINEERING

### Tâche choisie : Fonction optimize_memory()

**Prompt utilisé avec Claude/ChatGPT :**
> "Écris une fonction Python appelée optimize_memory(df) qui prend 
> un DataFrame pandas en entrée et réduit sa consommation mémoire 
> en convertissant les colonnes float64 en float32 et int64 en int32. 
> Affiche la mémoire avant et après."

**Résultat obtenu :** La fonction générée était fonctionnelle et 
réduisait la mémoire de X%. 

**Efficacité du prompt :** Très efficace car le prompt était précis 
sur le nom de la fonction, les types à convertir et l'affichage attendu.

---

## 👥 ÉQUIPE

| Membre | Rôle |
|--------|------|
| Prénom NOM | Data preprocessing & EDA |
| Prénom NOM | Modélisation ML |
| Prénom NOM | Interface Streamlit |
| Prénom NOM | ConfigurTableau  |
| Prénom NOM | Tests & CI/CD |


---

## 📊 DATASET

[Regensburg Pediatric Appendicitis — UCI ML Repository](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis)