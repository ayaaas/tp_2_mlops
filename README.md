# TP MLOps — Conteneurisation, Optimisation et Déploiement d'un Modèle de Machine Learning

## 1. Présentation du projet

Ce projet met en œuvre un pipeline de Machine Learning de bout en bout, depuis la préparation des données et l'entraînement du modèle jusqu'à son déploiement dans une API REST conteneurisée.

L'objectif est de comparer une approche Docker naïve à une approche optimisée, en appliquant les bonnes pratiques étudiées dans le cours de virtualisation et de conteneurisation.

Le projet couvre notamment :

- La préparation des données et l'entraînement d'un modèle de classification.
- La sérialisation du modèle et de son pipeline de prétraitement.
- La création d'images Docker pour l'entraînement et l'inférence.
- L'optimisation des couches, du cache et de la taille des images.
- La sécurisation des conteneurs et l'analyse des vulnérabilités avec Trivy.
- L'orchestration du service d'inférence avec Docker Compose.
- L'évaluation des performances à l'aide de benchmarks.

### Objectifs pédagogiques

| Notion du cours | Application dans le projet |
|---|---|
| Dockerfile | Construction des images d'entraînement et d'inférence |
| Couches et cache Docker | Ordonnancement des instructions `COPY` et `RUN` |
| `.dockerignore` | Exclusion des données et artefacts inutiles du contexte de build |
| Images légères | Utilisation de `python:3.11-slim` |
| Multi-stage build | Séparation de la construction et de l'image finale |
| Sécurité | Exécution avec un utilisateur non-root |
| Healthcheck | Vérification de l'état du service d'inférence |
| Docker Compose | Déploiement reproductible de l'API |
| Trivy | Détection des vulnérabilités |
| Benchmark | Comparaison des temps de build et des tailles d'image |

---

## 2. Architecture et structure du projet

### Structure des fichiers

```text
.
├── .dockerignore              # Exclusion des fichiers inutiles du contexte de build
├── .gitignore                 # Exclusion des environnements, données et artefacts locaux
├── Dockerfile.train           # Image Docker pour l'entraînement du modèle
├── Dockerfile.serve           # Image d'inférence naïve
├── Dockerfile.serve.opt       # Image d'inférence optimisée
├── README.md                  # Documentation du projet
├── benchmark.sh               # Mesure des performances des builds
├── docker-compose.yml         # Orchestration du service d'inférence
├── requirements.txt           # Dépendances Python épinglées
├── artifacts/                 # Modèle sérialisé et métriques
│   ├── .gitkeep
│   ├── metrics.json
│   └── model.joblib
├── data/                      # Données locales, exclues de Git
│   ├── adult.csv
│   └── sample.csv
└── src/
    ├── __init__.py
    ├── common.py              # Prétraitement partagé
    ├── download.py            # Téléchargement ou génération des données
    ├── predict.py             # API REST FastAPI
    └── train.py               # Entraînement et évaluation
```

### Architecture fonctionnelle

```text
                  DONNÉES
                     |
                     v
             src/download.py
                     |
                     v
               data/adult.csv
                     |
                     v
               src/train.py
                     |
                     v
          Prétraitement + modèle ML
                     |
                     v
              artifacts/
              ├── model.joblib
              └── metrics.json
                     |
                     v
             Docker Compose
                     |
                     v
               API FastAPI
                     |
            +--------+--------+
            |                 |
            v                 v
       GET /health       POST /predict
                              |
                              v
                       Prédiction ML
```

Le modèle est entraîné séparément du service d'inférence. L'API charge ensuite le modèle sérialisé pour effectuer des prédictions sans réentraîner le modèle au démarrage.

---

## 3. Dépendances Python

Le fichier `requirements.txt` fixe les versions des dépendances afin de réduire les différences entre les environnements d'entraînement et d'inférence.

| Dépendance | Version | Rôle |
|---|---|---|
| `scikit-learn` | `1.5.2` | Prétraitement et modèle ML |
| `pandas` | `2.2.3` | Manipulation des données |
| `numpy` | `2.1.3` | Calcul numérique |
| `joblib` | `1.4.2` | Sérialisation du modèle |
| `fastapi` | `0.115.5` | API REST |
| `uvicorn[standard]` | `0.32.1` | Serveur ASGI |
| `pydantic` | `2.10.3` | Validation des données d'entrée |

L'épinglage des versions améliore la reproductibilité, mais ne garantit pas à lui seul une reproductibilité parfaite. Celle-ci dépend également de la version de Python, du système, des données et des paramètres d'entraînement.

---

## 4. Pipeline de Machine Learning

### 4.1. Prétraitement partagé — `src/common.py`

Le prétraitement est intégré au pipeline scikit-learn pour appliquer les mêmes transformations pendant l'entraînement et l'inférence.

**Variables numériques :**
- Imputation des valeurs manquantes par la médiane avec `SimpleImputer`.
- Normalisation avec `StandardScaler`.

**Variables catégorielles :**
- Imputation des valeurs manquantes par la valeur la plus fréquente.
- Encodage One-Hot avec `OneHotEncoder`.

L'utilisation d'un pipeline partagé limite les risques d'incohérence entre les données utilisées pendant l'entraînement et celles reçues par l'API.

### 4.2. Entraînement — `src/train.py`

Le pipeline d'entraînement comprend :

1. Le chargement et le nettoyage des données.
2. Le découpage stratifié en 80 % d'entraînement et 20 % de test.
3. L'application du prétraitement.
4. L'entraînement d'un `RandomForestClassifier`.
5. L'évaluation du modèle.
6. La sauvegarde des résultats dans `artifacts/`.

**Configuration du modèle :**

- Algorithme : `RandomForestClassifier`.
- Nombre d'arbres : 10.
- Profondeur maximale : 10.
- Parallélisation : `n_jobs=-1`.

**Artefacts produits :**

| Fichier | Contenu |
|---|---|
| `artifacts/model.joblib` | Modèle entraîné et pipeline de prétraitement |
| `artifacts/metrics.json` | Accuracy, durée d'entraînement et effectifs |

Le jeu de test est utilisé pour évaluer les performances sur des observations qui n'ont pas servi à l'entraînement.

### 4.3. API d'inférence — `src/predict.py`

L'API FastAPI expose deux fonctionnalités principales :

- `GET /health` : vérification de l'état du service.
- `POST /predict` : réception des données, validation avec Pydantic et production d'une prédiction.

Le modèle est chargé depuis `/artifacts/model.joblib`.

L'entraînement et l'inférence sont ainsi séparés, ce qui facilite le déploiement et évite de reconstruire le modèle à chaque requête.

---

## 5. Construction des images Docker

Le projet utilise trois Dockerfiles afin de comparer différentes approches de conteneurisation.

| Dockerfile | Fonction |
|---|---|
| `Dockerfile.train` | Entraînement du modèle |
| `Dockerfile.serve` | Image naïve du service d'inférence |
| `Dockerfile.serve.opt` | Image d'inférence optimisée |

### 5.1. Approche naïve

La version naïve sert de référence pour mesurer l'effet des optimisations.

Ses principaux points faibles sont :

- Une image de base plus complète que nécessaire.
- Une copie prématurée de l'ensemble du contexte avec `COPY . .`.
- Une invalidation potentiellement fréquente du cache des dépendances.
- L'absence éventuelle d'un `.dockerignore` suffisamment restrictif.
- La conservation possible de fichiers et de caches inutiles.
- L'exécution en tant que `root`.
- Une commande de démarrage qui peut utiliser la forme shell.

Ces choix peuvent augmenter la taille de l'image, ralentir les reconstructions et accroître la surface d'attaque.

### 5.2. Version optimisée

Le fichier `Dockerfile.serve.opt` applique les bonnes pratiques suivantes.

**1. Image de base allégée**

Utilisation de `python:3.11-slim` afin de réduire le nombre de paquets et la taille de l'image par rapport à une image Python complète.

**2. Ordonnancement des couches**

Le fichier `requirements.txt` est copié et les dépendances sont installées avant la copie du code source.

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
```

Ainsi, une modification du code source ne nécessite pas systématiquement de réinstaller les dépendances.

**3. Contexte de build réduit**

Le fichier `.dockerignore` exclut les données locales, les environnements virtuels, les caches Python, les fichiers Git et les artefacts qui ne sont pas nécessaires à la construction de l'image.

**4. Installation sans cache pip**

L'option `--no-cache-dir` évite de conserver le cache de téléchargement des paquets dans l'image.

**5. Build multi-étapes**

Une étape temporaire `builder` prépare les dépendances. L'image finale ne conserve que les éléments nécessaires à l'exécution.

**6. Sécurité et supervision**

Le service s'exécute avec un utilisateur non privilégié, `appuser` (UID 1000), et dispose d'un `HEALTHCHECK` pour vérifier son état.

> Un build multi-étapes ne garantit pas automatiquement une image finale plus petite : le résultat dépend des éléments copiés depuis l'étape de construction et des dépendances requises à l'exécution.

---

## 6. Optimisation du cache Docker

Docker construit une image à partir de couches successives. Lorsqu'une couche change, les couches suivantes peuvent devoir être reconstruites.

La stratégie consiste à placer les éléments les moins susceptibles de changer avant ceux qui changent fréquemment.

Ordre recommandé :

1. Image de base.
2. Dépendances système nécessaires.
3. Fichier `requirements.txt` et installation des dépendances Python.
4. Code source de l'application.
5. Autres fichiers strictement nécessaires.

Par exemple :

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
```

Avec cet ordre, une modification de `src/` peut réutiliser le cache de l'installation des dépendances si les instructions précédentes et leurs entrées n'ont pas changé.

### Rôle de `.dockerignore`

Le contexte de build correspond aux fichiers accessibles à Docker lors de la construction de l'image.

Le fichier `.dockerignore` peut exclure :

```text
.git/
.venv/
venv/
__pycache__/
*.py[cod]
*.ipynb
data/
mlruns/
checkpoints/
.env
```

Les règles doivent être adaptées au projet : si un fichier est nécessaire à l'exécution, il doit être copié ou fourni autrement.

L'exclusion de `artifacts/` est également pertinente si le modèle est fourni au conteneur par un volume ou par un mécanisme de déploiement séparé. Il faut cependant s'assurer que le modèle est disponible avant le démarrage de l'API.

---

## 7. Benchmark : comparaison des performances

Les mesures suivantes correspondent aux résultats observés pendant les expérimentations du projet.

| Métrique | Version naïve | Version optimisée | Interprétation |
|---|---:|---:|---|
| Taille de l'image | 1 597 Mo | 552 Mo | Réduction d'environ 65,4 % |
| Build à froid | 58–60 s | 86,9 s | Build optimisé plus long lors de cette mesure initiale |
| Rebuild après modification du code | 52–67 s | 1,6 s | Cache des dépendances beaucoup mieux exploité |
| Temps d'entraînement | 0,66 s | 0,2287 s sur l'hôte / 0,267 s en conteneur | Comparaison dépendante de l'environnement et du protocole |
| Utilisateur d'exécution | `root` | `appuser`, UID 1000 | Réduction des privilèges |

### Analyse des résultats

**Taille de l'image**

Le passage de 1 597 Mo à 552 Mo représente une réduction d'environ 65,4 %, soit environ 1 045 Mo économisés.

**Build à froid**

Le build optimisé mesuré à 86,9 secondes est plus long que le build naïf mesuré à 58–60 secondes. Cela peut notamment s'expliquer par le téléchargement initial de l'image de base Slim et par le coût du build multi-étapes.

Il ne faut donc pas conclure que toutes les optimisations accélèrent nécessairement le premier build.

**Rebuild après modification du code**

Le passage de 52–67 secondes à 1,6 seconde montre l'intérêt du cache Docker lorsque les dépendances restent inchangées.

**Temps d'entraînement**

Le temps d'entraînement est une mesure distincte du temps de construction de l'image. Les résultats dépendent notamment de la charge système, du cache, des données et des conditions d'exécution. Une comparaison fiable nécessite plusieurs répétitions dans des conditions identiques.

Le script `benchmark.sh` permet de reproduire et de comparer les mesures de performance.

---

## 8. Docker Compose : orchestration du service

### 8.1. Rôle

Docker Compose permet de décrire et de démarrer une application composée d'un ou plusieurs conteneurs à partir d'un fichier YAML.

Dans ce projet, `docker-compose.yml` orchestre le service d'inférence et fournit sa configuration d'exécution.

Il centralise notamment :

- La construction ou la sélection de l'image.
- La publication des ports.
- Les volumes.
- Les variables d'environnement.
- Les politiques de redémarrage.
- Les contrôles de santé.

### 8.2. Volume du modèle

Le modèle entraîné est stocké dans `artifacts/` et rendu accessible au conteneur.

Cette séparation permet de reconstruire l'image du service sans nécessairement réentraîner le modèle.

Le volume doit pointer vers le chemin attendu par l'application, ici `/artifacts/model.joblib`.

### 8.3. Réseau et ports

Le mapping :

```yaml
ports:
  - "8000:8000"
```

relie le port 8000 de la machine hôte au port 8000 du conteneur.

Sur le réseau Compose, les services peuvent communiquer en utilisant leurs noms de service comme noms d'hôte. En revanche, `localhost` à l'intérieur d'un conteneur désigne ce conteneur lui-même, et non les autres services.

### 8.4. Démarrage et supervision

Pour construire l'image et lancer le service :

```bash
docker compose up -d --build
```

Pour consulter l'état des services :

```bash
docker compose ps
```

Pour suivre les journaux de l'API :

```bash
docker compose logs -f
```

Pour arrêter les services :

```bash
docker compose down
```

Les volumes nommés sont conservés par défaut lors de `docker compose down`. L'option `-v` supprime les volumes concernés :

```bash
docker compose down -v
```

Cette dernière commande doit être utilisée avec prudence, car elle peut supprimer des données persistantes.

---

## 9. Guide d'exécution

### Étape 1 — Préparer les données

Depuis la racine du projet :

```bash
python3 -m src.download
```

Cette commande télécharge ou génère les données selon l'implémentation de `src/download.py`.

### Étape 2 — Entraîner le modèle

```bash
python3 -m src.train
```

Vérifier ensuite la présence des artefacts :

```text
artifacts/
├── metrics.json
└── model.joblib
```

### Étape 3 — Construire l'image naïve d'entraînement

```bash
docker build \
  -f Dockerfile.train \
  -t adult-train:naif .
```

Exécuter le conteneur en montant le répertoire des artefacts :

```bash
docker run --rm \
  -v "$(pwd)/artifacts:/artifacts" \
  adult-train:naif
```

Le montage permet de récupérer sur l'hôte les artefacts écrits dans `/artifacts`.

### Étape 4 — Démarrer l'API

Vérifier d'abord que le modèle existe dans `artifacts/`, puis lancer :

```bash
docker compose up -d --build
```

### Étape 5 — Vérifier l'état de l'API

```bash
curl http://localhost:8000/health
```

Réponse attendue selon l'implémentation actuelle :

```json
{
  "statut": "ok"
}
```

### Étape 6 — Tester une prédiction

```bash
curl -X POST \
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
```

La requête doit respecter le schéma d'entrée défini dans `src/predict.py`. La réponse exacte dépend du format de sortie implémenté par l'API.

---

## 10. Sécurité des images avec Trivy

Trivy est un outil d'analyse de sécurité capable de détecter notamment des vulnérabilités dans les images de conteneurs, les dépendances, les systèmes de fichiers et certaines configurations.

### 10.1. Analyse de l'image naïve

```bash
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image \
  --severity HIGH,CRITICAL \
  adult-serve:naif
```

Cette commande analyse les vulnérabilités de sévérité HIGH et CRITICAL de l'image `adult-serve:naif`.

### 10.2. Ignorer les vulnérabilités sans correctif disponible

```bash
trivy image \
  --severity HIGH,CRITICAL \
  --ignore-unfixed \
  adult-serve:opt
```

Cette option limite les résultats aux vulnérabilités pour lesquelles un correctif est disponible, dans le périmètre couvert par les filtres appliqués.

### 10.3. Analyser le dépôt

```bash
trivy fs \
  --scanners vuln,secret,misconfig \
  .
```

Cette commande recherche les vulnérabilités, les secrets et les mauvaises configurations détectables dans le système de fichiers analysé.

### 10.4. Faire échouer une vérification de sécurité

```bash
trivy image \
  --severity CRITICAL \
  --exit-code 1 \
  adult-serve:opt
```

Avec `--exit-code 1`, la commande renvoie un code de sortie non nul si des vulnérabilités correspondant au filtre sont détectées. Ce comportement peut être utilisé dans une pipeline CI/CD.

### 10.5. Principales sources de vulnérabilités

L'analyse doit couvrir trois catégories :

1. **Image de base :** vulnérabilités de l'image Python ou de la distribution système.
2. **Paquets système :** bibliothèques et utilitaires installés dans l'image.
3. **Dépendances applicatives :** bibliothèques Python telles que FastAPI, Starlette et leurs dépendances transitives.

Les résultats Trivy doivent être interprétés en tenant compte des versions réellement détectées, de la disponibilité des correctifs et de leur compatibilité avec le projet.

Le recours à une image Slim réduit potentiellement la surface d'attaque, mais ne garantit pas l'absence de vulnérabilités.

### Bonnes pratiques de sécurité

- Utiliser un utilisateur non-root.
- Éviter de copier les secrets dans l'image.
- Ne pas utiliser `ARG` ou `ENV` comme mécanismes de stockage sécurisé des secrets.
- Utiliser des secrets dédiés lorsque cela est nécessaire, notamment les mécanismes de secrets de BuildKit pour les secrets de construction.
- Exclure les fichiers sensibles avec `.dockerignore`.
- Épingler et maintenir les dépendances.
- Scanner régulièrement les images.
- Mettre à jour les images de base et reconstruire les images après les correctifs.

---

## 11. Limites et perspectives

Ce projet démontre la construction, l'optimisation et le déploiement d'un service ML sur une seule machine.

Docker Compose convient à ce scénario, mais ne remplace pas un orchestrateur distribué.

Ses principales limites dans un contexte de production à grande échelle sont :

- L'absence d'orchestration native entre plusieurs machines.
- Le scaling qui nécessite une action explicite.
- L'absence d'autoscaling natif fondé sur la charge.
- Des mécanismes de déploiement progressif et de rollback moins complets que ceux d'une plateforme d'orchestration dédiée.
- La nécessité d'intégrer des solutions adaptées pour les secrets, les métriques et la supervision avancée.

### Perspectives d'évolution

Le projet pourrait être enrichi par :

- L'intégration d'une pipeline CI/CD pour automatiser les tests, le build et les scans de sécurité.
- L'ajout de tests unitaires et de tests d'intégration de l'API.
- Le suivi des métriques d'inférence et des erreurs.
- L'ajout de MLflow pour le suivi des expériences et des modèles.
- L'intégration de Prometheus et Grafana pour la supervision.
- Le déploiement sur Kubernetes si des besoins d'orchestration multi-machines apparaissent.

---

## 12. Conclusion

Ce TP illustre le passage d'une conteneurisation fonctionnelle à une conteneurisation plus optimisée et plus sûre.

La comparaison entre les images naïve et optimisée montre l'importance de plusieurs bonnes pratiques : ordonnancement des couches, exploitation du cache Docker, réduction du contexte de build, choix d'une image de base adaptée, builds multi-étapes et exécution avec un utilisateur non privilégié.

Les benchmarks mettent en évidence une réduction significative de la taille de l'image et du temps de reconstruction après modification du code. Ils montrent également qu'un build à froid peut être plus long malgré une image finale plus légère.

Enfin, Docker Compose facilite le déploiement reproductible de l'API, tandis que Trivy contribue à identifier les risques de sécurité.

**À retenir :**

- **Dockerfile :** construit une image.
- **Docker Compose :** configure et lance une application multi-conteneurs.
- **Kubernetes :** orchestre des conteneurs à plus grande échelle.

---

## 👤 Auteur et encadrement

- **Réalisé par :** Anam Aya
- **Formation :** Cycle d'ingénieur en Génie Informatique — option Data & Informatique
- **Encadrant :** Pr. Mohammed AMEKSA
- **Établissement :** Université Mundiapolis
