from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, create_model
from config import MODEL_PATH, PROJECT_NAME
import joblib
import pandas as pd
import psycopg2
import json
from config import MODEL_PATH, PROJECT_NAME, DATABASE_URL
import numpy as np
from typing import Any

# Charger le modele
#MODEL_PATH = "mon_modele_gradient_final.pkl"
API_TITLE  = "API Prédiction Immobilière"


# Charger le modèle
model = joblib.load(MODEL_PATH)

# Récupérer automatiquement les features du modèle
def get_features():
    try:
        # ColumnTransformer → columntransformer dans le pipeline
        ct = model.named_steps['columntransformer']
        num_cols = ct.transformers_[0][2]
        cat_cols = ct.transformers_[1][2]
        return list(num_cols) + list(cat_cols)
    except Exception as e:
        print(f"get_features error: {e}")
        return []

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
def get_db():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"DB connection error: {e}")
        return None

def sauvegarder_prediction(features_dict, prix):
    conn = get_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO predictions_immo (features, prix_predit) VALUES (%s, %s)",
                (json.dumps(features_dict), prix)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"DB insert error: {e}")

@app.get("/", response_class=HTMLResponse)
def accueil():
    with open("interface.html", "r", encoding="utf-8") as f:
        return f.read()

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
        prix = round(float(prediction), 2)
        sauvegarder_prediction(data.dict(), round(float(prediction), 2))
        return {
            "prediction": round(float(prediction), 2),
            "type": "regression"
        }
    
@app.get("/health")
def health():
    conn = get_db()
    db_ok = conn is not None
    if conn:
        conn.close()
    return {
        "status"      : "ok",
        "model_loaded": model is not None,
        "db_connected": db_ok
    }  