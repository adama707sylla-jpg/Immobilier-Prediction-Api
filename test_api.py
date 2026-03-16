from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

SAMPLE_INPUT = {
    # features
    "MSSubClass": 20, "LotArea": 8000, "OverallQual": 6,
    "OverallCond": 5, "YearBuilt": 2000, "YearRemodAdd": 2000,
    "BsmtFinSF1": 0, "BsmtFinSF2": 0, "BsmtUnfSF": 0,
    "TotalBsmtSF": 800, "1stFlrSF": 800, "2ndFlrSF": 0,
    "LowQualFinSF": 0, "GrLivArea": 800, "BsmtFullBath": 0,
    "BsmtHalfBath": 0, "FullBath": 1, "HalfBath": 0,
    "BedroomAbvGr": 3, "KitchenAbvGr": 1, "TotRmsAbvGrd": 6,
    "Fireplaces": 0, "GarageCars": 1, "GarageArea": 400,
    "WoodDeckSF": 0, "OpenPorchSF": 0, "EnclosedPorch": 0,
    "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0,
    "MiscVal": 0, "MoSold": 6, "YrSold": 2010,
    "MSZoning": 0, "LotFrontage": 0, "Street": 0,
    "LotShape": 0, "LandContour": 0, "Utilities": 0,
    "LotConfig": 0, "LandSlope": 0, "Neighborhood": 0
    "Condition1": 0, "Condition2": 0,"BldgType": 0,
  "HouseStyle": 0,"RoofStyle": 0,"RoofMatl": 0,
  "Exterior1st": 0,"Exterior2nd": 0,"MasVnrArea": 0,
  "ExterQual": 0,"ExterCond": 0,"Foundation": 0,
  "BsmtQual": 0,"BsmtCond": 0,"BsmtExposure": 0,
  "BsmtFinType1": 0,"BsmtFinType2": 0,"Heating": 0,
  "HeatingQC": 0,"CentralAir": 0,"Electrical": 0,
  "KitchenQual": 0,"Functional": 0,"GarageType": 0,
  "GarageYrBlt": 0,"GarageFinish": 0,"GarageQual": 0,
  "GarageCond": 0,"PavedDrive": 0,"SaleType": 0,
  "SaleCondition": 0
}

# ── Test 1 : API vivante 
def test_root():
    r = client.get("/")
    assert r.status_code == 200

# ── Test 2 : Prédiction fonctionne 
def test_predict_status():
    r = client.post("/predict", json=SAMPLE_INPUT)
    assert r.status_code == 200

# ── Test 3 : Structure de la réponse 
def test_predict_structure():
    r = client.post("/predict", json=SAMPLE_INPUT)
    data = r.json()
    # Clés obligatoires dans TOUS les projets
    assert "prediction" in data
    assert "type" in data
    assert data["type"] in ["classification", "regression"]

# ── Test 4 : Logique selon le type 
def test_predict_logique():
    r = client.post("/predict", json=SAMPLE_INPUT)
    data = r.json()

    if data["type"] == "classification":
        # Vérifier que confiance est entre 0 et 1
        assert "confiance" in data
        assert 0.0 <= data["confiance"] <= 1.0
        # Vérifier que prediction est "0" ou "1"
        assert data["prediction"] in ["0", "1"]

    elif data["type"] == "regression":
        # Pas de confiance en régression
        assert "confiance" not in data
        # Vérifier que la valeur est un nombre positif
        assert isinstance(data["prediction"], (int, float))
        assert data["prediction"] > 0

# ── Test 5 : Mauvaise requête → erreur propre 
def test_bad_request():
    r = client.post("/predict", json={})
    # Doit retourner 422 (données invalides)
    assert r.status_code == 422