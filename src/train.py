import json
import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from src.common import build_preprocessor

def main():
    print("Chargement des données...")
    df = pd.read_csv("data/adult.csv")
    
    df.columns = df.columns.str.strip().str.replace('-', '_')
    
    target_col = [col for col in df.columns if 'income' in col or 'class' in col]
    target_name = target_col[0] if target_col else df.columns[-1]
    
    df = df.dropna(subset=[target_name])
    if df[target_name].dtype == 'object':
        df[target_name] = df[target_name].astype(str).str.strip()
        df = df[~df[target_name].isin(['nan', '?', '', 'None'])]
    
    X = df.drop(columns=[target_name])
    y = df[target_name]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = build_preprocessor(X_train)
    
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Entraînement
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    duration = time.time() - start_time
    
    print(f"Temps d'entraînement : {duration:.4f} secondes")
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {acc:.4f}")
    
    # Création du dossier artifacts s'il n'existe pas
    os.makedirs("artifacts", exist_ok=True)
    
    # Sauvegarde du modèle
    joblib.dump(pipeline, "artifacts/model.joblib")
    print("Modèle sauvegardé dans artifacts/model.joblib")
    
    # Sauvegarde des métriques au format JSON
    metrics = {
        "accuracy": round(float(acc), 4),
        "training_time_seconds": round(duration, 4),
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test)
    }
    
    metrics_path = "artifacts/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Métriques sauvegardées dans {metrics_path}")

if __name__ == "__main__":
    main()
