# train.py — UNIVERSEL classification ET régression
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import MODEL_PATH, PROJECT_NAME, MLFLOW_URI
from queries import get_data_ml
from sklearn.linear_model import Ridge

import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")


#   CONFIG 

CONFIG = {
    #  Données 
    "table"        : "maisons",
    "target"       : "SalePrice",
    "drop_cols"    : ["Id"],
    "outlier_cols" : ["SalePrice", "GrLivArea", "LotArea"],

    # ── MODE : "classification" ou "regression"
    "mode"         : "regression",

    #  Split 
    "test_size"    : 0.2,
    "random_state" : 42,

    #  Modèle 
    "params"       : {
        "alpha": 1.0
}
}
# ══════════════════════════════════════════════
#   PIPELINE UNIVERSEL
# ══════════════════════════════════════════════

def build_preprocessor(X):
    """Construit le préprocesseur adapté aux colonnes."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"   Colonnes numériques    : {len(num_cols)}")
    print(f"   Colonnes catégorielles : {len(cat_cols)}")

    num_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )
    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def get_modeles(mode):
    """Retourne les modèles selon le mode."""
    if mode == "classification":
        return {
            "RandomForest"     : RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting" : GradientBoostingClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVM"              : SVC(),
            "KNN"              : KNeighborsClassifier(),
            "NaiveBayes"       : GaussianNB(),
        }
    else:
        return {
            "RandomForest"     : RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting" : GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge"            : Ridge(),
            "LinearRegression" : LinearRegression(),
            "KNN"              : KNeighborsRegressor(),
        }


def get_meilleur_modele(mode, params):
    """Retourne le meilleur modèle selon le mode."""
    if mode == "classification":
        return RandomForestClassifier(**params)
    else:
        return Ridge(**params)


def compare_modeles(X_train, X_test, y_train, y_test, preprocessor, mode):
    """Compare les modèles — fonctionne en classif ET régression."""
    modeles   = get_modeles(mode)
    resultats = []

    for nom, modele in modeles.items():
        pipe   = make_pipeline(preprocessor, modele)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        if mode == "classification":
            resultats.append({
                "modele"  : nom,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "F1"      : round(f1_score(y_test, y_pred, average="weighted"), 4),
            })
        else:
            resultats.append({
                "modele": nom,
                "R2"    : round(r2_score(y_test, y_pred), 4),
                "RMSE"  : round(np.sqrt(mean_squared_error(y_test, y_pred)), 0),
                "MAE"   : round(mean_absolute_error(y_test, y_pred), 0),
            })

    df = pd.DataFrame(resultats)
    col_tri = "Accuracy" if mode == "classification" else "R2"
    df = df.sort_values(col_tri, ascending=False)
    print(df.to_string(index=False))
    return df


def evaluer_modele(y_test, y_pred, mode, nom="Modele"):
    """Évaluation — fonctionne en classif ET régression."""
    print(f"\n----Evaluation du: {nom}----")

    if mode == "classification":
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        print(f" Accuracy : {acc*100:.2f}%")
        print(f" F1 Score : {f1*100:.2f}%")
        metriques = {"accuracy": acc, "f1": f1}
    else:
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        print(f" R2 Score : {r2:.4f}   (1.0 = parfait)")
        print(f" RMSE     : {rmse:,.0f} $")
        print(f" MAE      : {mae:,.0f} $")
        metriques = {"r2": r2, "rmse": rmse, "mae": mae}

    print("-" * 35)
    return metriques


def cleaner_outlier_df(df, col):
    """Supprime les outliers IQR sur une colonne."""
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    return df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]


# ══════════════════════════════════════════════
#   PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

def run_training(config=CONFIG):

    mode = config["mode"]

    print("\n" + "="*50)
    print(f"  {PROJECT_NAME}  [{mode.upper()}]")
    print("="*50)

    # ── 1. Chargement PostgreSQL ──────────────
    print("\n [1/7] Chargement des données...")
    X, y = get_data_ml(
        table     = config["table"],
        target    = config["target"],
        drop_cols = config["drop_cols"],
        dropna    = False
    )

    # ── 2. Nettoyage outliers ─────────────────
    print("\n [2/7] Nettoyage outliers...")
    df_work = X.copy()
    df_work[config["target"]] = y.values

    for col in config["outlier_cols"]:
        if col in df_work.columns:
            avant       = len(df_work)
            df_work     = cleaner_outlier_df(df_work, col)
            supprimes   = avant - len(df_work)
            print(f"   {col} : {supprimes} outliers supprimés")

    X = df_work.drop(columns=[config["target"]])
    y = df_work[config["target"]]

    # ── 3. Split train/test ───────────────────
    print("\n [3/7] Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config["test_size"],
        random_state = config["random_state"]
    )
    print(f"   Train : {len(X_train)} | Test : {len(X_test)}")

    # ── 4. Comparaison modèles ────────────────
    print("\n [4/7] Comparaison des modèles...")
    preprocessor = build_preprocessor(X_train)
    compare_modeles(X_train, X_test, y_train, y_test, preprocessor, mode)

    # ── 5. Entraînement modèle final ─────────
    print(f"\n [5/7] Entraînement Ridge [{mode}]...")
    modele_final = make_pipeline(
        build_preprocessor(X_train),
        get_meilleur_modele(mode, config["params"])
    )
    modele_final.fit(X_train, y_train)
    y_pred = modele_final.predict(X_test)

    # ── 6. MLflow tracking ────────────────────
    print("\n[6/7] Tracking MLflow...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(PROJECT_NAME)

    metriques = evaluer_modele(y_test, y_pred, mode, "Ridge")

    with mlflow.start_run(run_name=f"RF_{mode}_v1"):
        mlflow.log_param("mode",  mode)
        mlflow.log_param("table", config["table"])
        mlflow.log_params(config["params"])
        for k, v in metriques.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(modele_final, artifact_path="model")
        print("   Expérience enregistrée dans MLflow !")

    # ── 7. Sauvegarde ─────────────────────────
    print(f"\n [7/7] Sauvegarde → {MODEL_PATH}")
    joblib.dump(modele_final, MODEL_PATH)

    print("\n" + "="*50)
    print("  TRAINING TERMINÉ !")
    print("="*50 + "\n")

    return modele_final


if __name__ == "__main__":
    run_training()