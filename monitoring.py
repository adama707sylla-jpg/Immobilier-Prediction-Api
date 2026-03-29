import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset



CONFIG = {
    "reference_path": "train.csv",
    "current_path":   "test.csv",
    "features": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"],
    "encoding": {
        "Sex": {"male": 0, "female": 1}
    },
    "output": "rapport_monitoring.html",
    "project_name": "Titanic"
}



# 1. Charger les données
def charger(path, features, encoding):
    df = pd.read_csv(path)
    cols = [c for c in features if c in df.columns]
    df = df[cols].dropna()
    for col, mapping in encoding.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    print(f"✅ {path} — {len(df)} lignes")
    return df

# 2. Résumé terminal
def resume(reference, current):
    print(f"\n{'Feature':<20} {'Moy.Ref':>10} {'Moy.Prod':>10} {'Diff%':>8}")
    print("-" * 52)
    for col in reference.columns:
        if col in current.columns:
            r = reference[col].mean()
            c = current[col].mean()
            diff = abs((c - r) / r * 100) if r != 0 else 0
            alerte = "⚠️" if diff > 20 else "✅"
            print(f"{col:<20} {r:>10.2f} {c:>10.2f} {diff:>7.1f}% {alerte}")

# 3. Générer rapport HTML
def generer(reference, current, output, project_name):
    print(f"\n📊 Génération rapport {project_name}...")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Monitoring {project_name}</title>
    <style>
        body {{ font-family: Arial; margin: 40px; background: #f8f9fa; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; }}
        .card {{ background: white; padding: 20px; margin: 10px 0;
                 border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .ok {{ color: green; font-weight: bold; }}
        .warn {{ color: orange; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #3498db; color: white; padding: 10px; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; text-align: center; }}
    </style>
</head>
<body>
    <h1>📊 Rapport Monitoring — {project_name}</h1>

    <div class="card">
        <h2>📁 Données</h2>
        <p>Référence (train) : <b>{len(reference)} lignes</b></p>
        <p>Production (test) : <b>{len(current)} lignes</b></p>
    </div>

    <div class="card">
        <h2>📈 Comparaison par feature</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Moyenne Référence</th>
                <th>Moyenne Production</th>
                <th>Différence %</th>
                <th>Statut</th>
            </tr>
            {''.join([
                f"<tr>"
                f"<td>{col}</td>"
                f"<td>{reference[col].mean():.2f}</td>"
                f"<td>{current[col].mean():.2f}</td>"
                f"<td>{abs((current[col].mean() - reference[col].mean()) / reference[col].mean() * 100) if reference[col].mean() != 0 else 0:.1f}%</td>"
                f"<td class='{'warn' if abs((current[col].mean() - reference[col].mean()) / reference[col].mean() * 100) > 20 else 'ok'}'>"
                f"{'⚠️ Drift' if abs((current[col].mean() - reference[col].mean()) / reference[col].mean() * 100) > 20 else '✅ OK'}"
                f"</td></tr>"
                for col in reference.columns if col in current.columns
            ])}
        </table>
    </div>
</body>
</html>"""

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Rapport sauvegardé : {output}")
    print(f"   Ouvre {output} dans ton navigateur !")

# ══════════════════════════════════════════════
#   MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    reference = charger(CONFIG["reference_path"], CONFIG["features"], CONFIG["encoding"])
    current   = charger(CONFIG["current_path"],   CONFIG["features"], CONFIG["encoding"])
    resume(reference, current)
    generer(reference, current, CONFIG["output"], CONFIG["project_name"])