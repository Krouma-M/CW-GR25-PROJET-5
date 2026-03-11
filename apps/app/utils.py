import joblib # pyright: ignore[reportMissingImports]
import os
from django.conf import settings

# Chemins (ajustez si nécessaire)
MODEL_PATH = os.path.join(settings.BASE_DIR, '../models/best_model.pkl')
EXPLAINER_PATH = os.path.join(settings.BASE_DIR, '../models/explainer.pkl')

_model = None
_explainer = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = joblib.load(EXPLAINER_PATH)
    return _explainer