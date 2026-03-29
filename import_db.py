import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)

# Charger le CSV brut — TOUTES les colonnes
df = pd.read_csv("train.csv")

print(f"Shape : {df.shape}")
print(f"Colonnes : {list(df.columns)}")

# Importer TOUT dans PostgreSQL
df.to_sql("maisons", engine, if_exists="replace", index=False)

print(f"Table 'maisons' créée avec {len(df)} lignes !")
