# Documentation du dossier src/

Ce dossier contient le code source du projet de diagnostic d'appendicite.

## Fichiers

### 1. [download_data.py](download_data.py)
**Téléchargement des données depuis UCI Repository**

- `download_appendicitis_data()` : Télécharge le dataset depuis UCI (id=938) et le sauvegarde en CSV

**Utilisation :**
```bash
python src/download_data.py
```

---

### 2. [data_processing.py](data_processing.py)
**Prétraitement des données**

Fonctions principales :

- `load_data(filepath)` : Charge les données depuis un fichier CSV
- `preprocess_data(df, target_col, test_size, random_state)` : Prétraite les données
  - Supprime les lignes avec cible manquante
  - Encode la variable cible (LabelEncoder)
  - Encode les variables catégorielles
  - Impute les valeurs manquantes (médiane)
  - Split train/test (80/20)
- `get_feature_names(X)` : Retourne les noms des features
- `get_data_summary(df)` : Affiche les statistiques du dataset

**Utilisation :**
```python
from src.data_processing import load_data, preprocess_data

df = load_data("data/raw/appendicitis.csv")
X_train, X_test, y_train, y_test, le_target = preprocess_data(df)
```

---

### 3. [train_model.py](train_model.py)
**Entraînement des modèles ML**

Ce script entraîne 3 modèles avec GridSearchCV :
- Random Forest
- LightGBM
- CatBoost (meilleur modèle)

**Résultats :**
- CatBoost : Accuracy 0.987, F1 0.984
- LightGBM : Accuracy 0.981, F1 0.976
- Random Forest : Accuracy 0.974, F1 0.968

**Les modèles sont sauvegardés dans `models/` :**
- `random_forest_model.pkl`
- `lightgbm_model.pkl`
- `catboost_model.cbm`
- `label_encoder.pkl`

**Utilisation :**
```bash
python src/train_model.py
```

---

### 4. [shap_analysis.py](shap_analysis.py)
**Analyse SHAP avec le meilleur modèle (CatBoost)**

Ce script :
1. Entraîne CatBoost (le meilleur modèle)
2. Génère des graphiques SHAP pour expliquer les prédictions

**Graphiques générés :**
- `figures/shap/catboost_summary.png` - Graphique résumé SHAP
- `figures/shap/catboost_importance.png` - Importance des features

**Top 5 symptômes/tests les plus importants :**
1. Appendix_on_US
2. Management
3. Appendix_Diameter
4. WBC_Count
5. Alvarado_Score

**Utilisation :**
```bash
python src/shap_analysis.py
```

---

### 5. [evaluate.py](evaluate.py)
**Évaluation des modèles**

Fonctions :

- `evaluate_model(y_true, y_pred, y_pred_proba, model_name)` : Calcule les métriques
- `print_evaluation(metrics)` : Affiche les métriques
- `get_confusion_matrix(y_true, y_pred)` : Matrice de confusion
- `get_classification_report(y_true, y_pred)` : Rapport de classification
- `compare_models(results)` : Compare plusieurs modèles
- `print_comparison(df)` : Affiche la comparaison

**Utilisation :**
```python
from src.evaluate import evaluate_model, print_evaluation

metrics = evaluate_model(y_test, y_pred, model_name="CatBoost")
print_evaluation(metrics)
```

---

## Ordre d'exécution recommandé

1. **Télécharger les données** (si pas encore fait) :
   ```bash
   python src/download_data.py
   ```

2. **Entraîner les modèles** :
   ```bash
   python src/train_model.py
   ```

3. **Analyse SHAP** (optionnel) :
   ```bash
   python src/shap_analysis.py
   ```

---

## Dépendances

Voir `requirements.txt` :
- pandas
- numpy
- scikit-learn
- lightgbm
- catboost
- shap
- matplotlib

---

## Structure des données

Le dataset contient 780 patients avec 55 features :
- Symptômes (douleur, fièvre, etc.)
- Résultats d'examens (échographie, analyse de sang, etc.)
- Données démographiques (âge, sexe, BMI, etc.)

Variable cible : `Diagnosis` (appendicitis / no appendicitis)
