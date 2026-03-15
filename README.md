# 🏠 ProjetData — AI Automated Real Estate Intelligence

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerisé-blue?style=flat-square&logo=docker)
![R2 Score](https://img.shields.io/badge/R²-92.5%25-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-En%20production-brightgreen?style=flat-square)

> Pipeline Data Science complet pour la prédiction de prix immobiliers — du nettoyage SQL jusqu'au déploiement cloud.

🌍 **API en production** → [immobilier-prediction-api.onrender.com](https://immobilier-prediction-api.onrender.com)  
📖 **Documentation** → [immobilier-prediction-api.onrender.com/docs](https://immobilier-prediction-api.onrender.com/docs)

---

## Résultats

| Métrique | Valeur |
|---|---|
| Algorithme | Gradient Boosting |
| R² Score | **92.5%** |
| Type | Régression (prédiction de prix) |
| Features | 73 paramètres immobiliers |

---

## Pipeline ML

```
Chargement CSV + Connexion SQLite
        ↓
Nettoyage (IQR outliers + valeurs manquantes)
        ↓
Tournoi de modèles (Ridge, Lasso, RandomForest, GradientBoosting)
        ↓
Gradient Boosting sélectionné (R² = 92.5%)
        ↓
API FastAPI + Docker
        ↓
Déploiement Cloud (Render)
```

---

## Stack technique

| Catégorie | Outils |
|---|---|
| **Langage** | Python 3.11 |
| **Machine Learning** | Scikit-Learn, Gradient Boosting, Pipeline automatique |
| **Data Engineering** | Pandas, NumPy, SQLite, IQR outlier detection |
| **API** | FastAPI, Uvicorn |
| **Containerisation** | Docker |
| **Déploiement** | Render (Cloud) |

---

## Utilisation de l'API

### Tester en ligne
```bash
curl -X POST https://immobilier-prediction-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "GrLivArea": 1500,
    "OverallQual": 7,
    "YearBuilt": 2005,
    "TotalBsmtSF": 800,
    "GarageCars": 2
  }'
```

### Réponse
```json
{
  "prediction": 185000.00,
  "type": "regression"
}
```

---

## Lancer en local

```bash
# Cloner le repo
git clone https://github.com/adama707sylla-jpg/Immobilier-Prediction-Api.git
cd Immobilier-Prediction-Api

# Avec Docker (recommandé)
docker build -t immobilier-api .
docker run -p 8000:10000 immobilier-api

# Sans Docker
pip install -r requirements.txt
uvicorn app:app --reload
```

Ouvre ensuite : `http://127.0.0.1:8000/docs`

---

## Structure du projet

```
Immobilier-Prediction-Api/
├── app.py                          # API FastAPI universelle
├── mon_modele_gradient_final.pkl   # Modèle Gradient Boosting
├── mon_outillage.py                # Pipeline nettoyage + comparaison modèles
├── interface.html                  # Interface web utilisateur
├── requirements.txt                # Librairies Python
├── Dockerfile                      # Configuration Docker
└── README.md
```

---

## Points forts

- **Data Engineering** : Connexion SQL et extraction automatisée
- **Smart Cleaning** : Détection des outliers par méthode IQR
- **Auto-Pipeline** : Traitement dynamique colonnes numériques et catégorielles
- **Model Tournament** : Comparaison Ridge, Lasso, RandomForest, GradientBoosting
- **API Universelle** : Template réutilisable pour tout projet ML

---

## Auteur

**Adama Sylla** — Étudiant Data Science (MIAGE L3)  
📧 Adama101sylla@gmail.com  
🌐 [Portfolio](http://adama707.pythonanywhere.com)  
💼 [GitHub](https://github.com/adama707sylla-jpg)

---

*Projet réalisé dans le cadre du IBM Data Science Professional Certificate*

