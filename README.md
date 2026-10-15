.
├── .dockerignore              # Fichiers exclus du contexte de build Docker
├── .gitignore                 # Exclusion des venv, données et artefacts
├── Dockerfile.train           # Image Docker naïve pour l'entraînement
├── Dockerfile.serve           # Image Docker naïve pour le service d'inférence
├── Dockerfile.serve.opt       # Image Docker multi-étapes optimisée et non-root
├── README.md                  # Documentation du projet
├── benchmark.sh               # Script d'évaluation des performances des builds
├── docker-compose.yml         # Orchestration du service d'inférence avec volume
├── requirements.txt           # Dépendances Python fixées
├── artifacts/                 # Modèles et métriques générés
│   ├── .gitkeep
│   ├── metrics.json
│   └── model.joblib
├── data/                      # Données du projet (ignorées par Git)
│   ├── adult.csv
│   └── sample.csv
└── src/                       # Code source Python
    ├── __init__.py
    ├── common.py              # Pipeline de prétraitement partagé
    ├── download.py            # Téléchargement et jeu synthétique de secours
    ├── predict.py             # API REST d'inférence (FastAPI)
    └── train.py               # Entraînement et évaluation du modèle
```[cite: 2]

---

## ⚙️ Dépendances (`requirements.txt`)
Toutes les dépendances sont strictement épinglées afin de garantir la reproductibilité lors du rechargement du modèle sérialisé[cite: 2] :
* `scikit-learn==1.5.2`[cite: 2]
* `pandas==2.2.3`[cite: 2]
* `numpy==2.1.3`[cite: 2]
* `joblib==1.4.2`[cite: 2]
* `fastapi==0.115.5`[cite: 2]
* `uvicorn[standard]==0.32.1`[cite: 2]
* `pydantic==2.10.3`[cite: 2]

---

## 🚀 Pipeline & Architecture

### 1. Prétraitement Partagé (`src/common.py`)
* Détection automatique des colonnes numériques et catégorielles[cite: 3].
* **Colonnes numériques** : Imputation par la médiane (`SimpleImputer`) + Normalisation (`StandardScaler`)[cite: 3].
* **Colonnes catégorielles** : Imputation par la valeur la plus fréquente (`SimpleImputer`) + Encodage One-Hot (`OneHotEncoder`)[cite: 3].
* Intégré au `Pipeline` scikit-learn pour garantir que le même prétraitement est appliqué à l'entraînement et en production[cite: 3, 4].

### 2. Entraînement du Modèle (`src/train.py`)
* Nettoyage des données et découpage stratifié (80% Train / 20% Test)[cite: 4].
* Algorithme : `RandomForestClassifier` (10 arbres, profondeur max de 10, `n_jobs=-1` pour la parallélisation)[cite: 4, 15].
* Sauvegarde des artefacts dans `artifacts/` :
  * `model.joblib` : Modèle et pipeline de prétraitement sérialisés[cite: 4].
  * `metrics.json` : Métriques de performance (Accuracy, durée d'entraînement, effectifs)[cite: 4, 15].

### 3. API REST d'Inférence (`src/predict.py`)
* Développée avec **FastAPI** et validation des données d'entrée via **Pydantic**[cite: 4, 5].
* Charge le modèle sérialisé depuis le dossier `/artifacts/model.joblib`[cite: 4, 5].
* Expose un point d'accès pour l'inférence ainsi qu'une route de santé `/health`[cite: 5, 10, 19].

---

## 🛠️ Optimisations Docker (`Dockerfile.serve.opt`)

L'image d'inférence optimisée cumule **6 optimisations clés**[cite: 10] :

1. **Image de base minimale (`python:3.11-slim`)** : Suppression des compilateurs et dépendances inutiles de la version complète[cite: 10].
2. **Gestion du cache Docker** : `COPY requirements.txt` placé avant la copie des sources `COPY src/`[cite: 10].
3. **`.dockerignore` complet** : Exclusion de `data/`, `artifacts/`, `__pycache__/` et fichiers de configuration locaux[cite: 10].
4. **`pip install --no-cache-dir`** : Suppression des archives temporaires de paquets[cite: 10, 15].
5. **Build multi-étapes (*Multi-Stage Build*)** : Isolation de l'étape de construction (`builder`) et copie des paquets installés dans l'image finale[cite: 10].
6. **Sécurité et Supervision** : Exécution sous un utilisateur non-root (`appuser`, UID 1000) et ajout d'un `HEALTHCHECK`[cite: 10].

### 📊 Tableau Comparatif des Performances

| Métrique / Indicateur | Version Naïve (`Dockerfile.serve`) | Version Optimisée (`Dockerfile.serve.opt`) | Gain / Impact |
| :--- | :--- | :--- | :--- |
| **Taille de l'image** | **1 597 Mo** (~1,6 Go)[cite: 8] | **552 Mo**[cite: 10, 11] | **-65,4 %** (~1,04 Go en moins)[cite: 11] |
| **Build à froid** | ~58 s - 60 s[cite: 8] | ~86,9 s (téléchargement initial de la base slim)[cite: 11, 18] | Réduction forte de l'empreinte disque |
| **Rebuild (après modif. code)** | **52 s - 67 s** (cache pip invalidé)[cite: 8, 9] | **1,6 s** (dépendances servies depuis le cache)[cite: 11, 18] | **Gains de temps x30 à x40**[cite: 11, 18] |
| **Temps d'entraînement ML** | 0,66 s[cite: 15, 17] | 0,2287 s (hôte) / 0,267 s (conteneur)[cite: 16, 17] | **-60 % à -65 %**[cite: 16, 17] |
| **Sécurité Utilisateur** | `root` (privilégié) | `appuser` (non-root, UID 1000)[cite: 10] | Principe du moindre privilège[cite: 10] |

---

## 🐳 Guide d'Exécution

### 1. Entraîner le modèle localement ou sous Docker
```bash
# Générer/Télécharger les données
python3 -m src.download

# Entraîner le modèle localement
python3 -m src.train

# Ou lancer l'entraînement dans le conteneur naïf
docker build -f Dockerfile.train -t adult-train:naif .
docker run --rm -v $(pwd)/artifacts:/artifacts adult-train:naif
```[cite: 5, 15]

### 2. Démarrer le service avec Docker Compose
Le fichier `docker-compose.yml` monte le dossier `./artifacts` sous forme de volume local[cite: 17] :

```bash
# Lancer l'orchestration en arrière-plan
docker compose up -d --build

# Tester le point d'accès de santé (Healthcheck)
curl http://localhost:8000/health
```[cite: 18, 19]

*Réponse attendue* :
```json
{"statut": "ok"}
```[cite: 19]

### 3. Tester une Prédiction (API Inférence)
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 39,
  "workclass": "State-gov",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40
}'
```[cite: 5]

---

## 🔒 Analyse de Sécurité (Trivy)

L'image Docker a été scannée avec **Trivy** pour analyser la surface d'attaque et identifier les vulnérabilités système et applicatives[cite: 19, 20] :

```bash
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest image --severity HIGH,CRITICAL adult-serve:naif
```[cite: 20]

* **Couche OS** : Réduction du nombre de paquets vulnérables en passant à l'image `python:3.11-slim` (qui élimine des utilitaires comme `wget`)[cite: 10, 21].
* **Paquets Python** : Analyse ciblée sur les dépendances répertoriées (ex: `starlette`, `wheel`, `jaraco.context`)[cite: 20, 21].
* **Recommandations** : Mettre à jour `fastapi` et `starlette` ($\ge$ 1.3.1) dans `requirements.txt`[cite: 21] et ré-exécuter le build régulièrement pour intégrer les derniers patchs Debian[cite: 21].

---

## 👤 Auteur & Encadrement
* **Réalisé par** : Anam Aya (3ᵉ année Cycle d'Ingénieur - Génie Informatique, option Data & Info)[cite: 1]
* **Encadrant** : Pr. Mohammed AMEKSA[cite: 1]
* **Établissement** : Université Mundiapolis[cite: 1]
