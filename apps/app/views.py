from django.shortcuts import render
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from .utils import get_model, get_explainer

def predict(request):
    context = {}
    if request.method == 'POST':
        # Récupération des données du formulaire
        age = float(request.POST.get('age'))
        sex = request.POST.get('sex')
        # ... récupérez tous les champs

        # Construction du DataFrame avec les mêmes noms de colonnes qu'à l'entraînement
        input_df = pd.DataFrame([[age, sex, ...]],
                                columns=['age', 'sex', ...])  # adaptez les noms

        # Chargement du modèle et de l'explicateur
        model = get_model()
        explainer = get_explainer()

        # Prétraitement si le modèle est un pipeline
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            input_processed = preprocessor.transform(input_df)
            feature_names = preprocessor.get_feature_names_out()
        else:
            input_processed = input_df.values
            feature_names = input_df.columns.tolist()

        # Prédiction
        proba = model.predict_proba(input_processed)[0][1]
        pred_class = 1 if proba >= 0.5 else 0

        # Calcul des valeurs SHAP
        shap_values = explainer.shap_values(input_processed)

        # Création d'un objet Explanation pour un waterfall plot
        exp = shap.Explanation(values=shap_values[0],
                               base_values=explainer.expected_value,
                               data=input_processed[0],
                               feature_names=feature_names)

        # Génération du graphique
        plt.figure()
        shap.waterfall_plot(exp, show=False)
        plt.tight_layout()

        # Conversion en image base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Ajout au contexte
        context['prediction'] = 'Appendicite' if pred_class else 'Pas d\'appendicite'
        context['probability'] = round(proba * 100, 2)
        context['shap_plot'] = img_str

    return render(request, 'app/predict.html', context)