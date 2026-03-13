from fastapi import FastAPI
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
import numpy as np
from typing import Any
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# CHANGE UNIQUEMENT CES 2 LIGNES POUR CHAQUE NOUVEAU PROJET
MODEL_PATH = "mon_modele_gradient_final.pkl"
API_TITLE  = "API Prédiction Immobilière"
# ============================================================

# Charger le modèle
model = joblib.load(MODEL_PATH)

# Récupérer automatiquement les features du modèle
def get_features():
    # Récupère les noms de colonnes depuis le preprocesseur
    try:
        feature_names = (
            model.named_steps['preprocessor']
            .transformers_[0][2].tolist() +
            model.named_steps['preprocessor']
            .transformers_[1][2].tolist()
        )
    except:
        feature_names = []
    return feature_names

features = get_features()

# Générer dynamiquement la classe de données
fields = {f: (Any, 0) for f in features}
DynamicInput = create_model("DynamicInput", **fields)

# Créer l'app
app = FastAPI(title=API_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def accueil():
    return {
        "message": f"{API_TITLE} opérationnelle !",
        "features_attendues": features,
        "nombre_features": len(features)
    }

@app.get("/features")
def voir_features():
    return {
        "features": features,
        "total": len(features)
    }

@app.post("/predict")
def predire(data: DynamicInput):
    # Convertir en DataFrame
    df = pd.DataFrame([data.dict()])

    # Prédire
    prediction = model.predict(df)[0]

    # Adapter le résultat selon le type de modèle
    if hasattr(model, 'classes_'):
        # Classification
        proba = model.predict_proba(df)[0].max()
        return {
            "prediction": str(prediction),
            "confiance": round(float(proba), 4),
            "type": "classification"
        }
    else:
        # Régression
        return {
            "prediction": round(float(prediction), 2),
            "type": "regression"
        }