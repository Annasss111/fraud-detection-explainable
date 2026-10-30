# Détection de fraude explicable

Projet Challenge 2 – Data Mining (ENSI, II3-IA, 2026-2027)
Auteur : Mohamed Anas Daoud

Application web interactive de détection de fraude sur cartes bancaires. Le projet compare une approche supervisée (XGBoost) et deux approches non supervisées (Isolation Forest, autoencodeur), fixe un seuil d'alerte à partir d'un coût métier exprimé en montants, et explique chaque alerte avec SHAP. Le modèle est exposé par une API FastAPI.

## Jeu de données

**Credit Card Transactions Fraud Detection Dataset** (simulateur Sparkov, synthétique), Kaggle, auteur `kartik2112`.

| | |
|---|---|
| Transactions | 1 852 394 (`fraudTrain.csv` + `fraudTest.csv` réunis) |
| Période | 1er janvier 2019 → 31 décembre 2020 |
| Cartes | 999 |
| Fraudes | 9 651 (0,52 %) |

Les deux fichiers d'origine sont réunis puis re-découpés chronologiquement (voir ci-dessous).

## Structure du projet

```
fraud-project/
├── data/                  # fraudTrain.csv, fraudTest.csv (non versionnés)
├── notebooks/
│   └── 01_exploration.ipynb   # modules 1 à 5 : données, variables, modèles, évaluation, SHAP
├── src/
│   ├── features.py        # construction des variables (sans fuite de données)
│   └── evaluate.py        # métriques : PR-AUC, ROC-AUC, précision@k, rappel@k
├── artifacts/             # générés par le notebook
│   ├── xgb.joblib
│   ├── meta.json
│   └── test_scored.parquet
├── app.py                 # application Streamlit
├── api.py                 # API FastAPI
└── requirements.txt
```

## Installation

Python 3.10 ou plus récent est recommandé.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

`requirements.txt` :

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
streamlit
fastapi
uvicorn
joblib
pyarrow
ipykernel
kagglehub
```

## Données

Télécharger le jeu sur Kaggle (vérifier la licence sur la page du jeu) et placer `fraudTrain.csv` et `fraudTest.csv` dans `data/`.

Alternative automatique, dans le notebook (nécessite un jeton Kaggle) :

```python
import kagglehub
path = kagglehub.dataset_download("kartik2112/fraud-detection")
```

## Utilisation

### 1. Exécuter le notebook

Ouvrir `notebooks/01_exploration.ipynb` avec le kernel `.venv`, puis **Run All**. La dernière cellule écrit les fichiers de `artifacts/`, dont dépendent l'application et l'API.

### 2. Lancer l'application web

```powershell
streamlit run app.py
```

L'application propose quatre onglets :

1. **Données** : taux de fraude, montants par classe, évolution mensuelle, taux par heure et par catégorie.
2. **Comparaison des modèles** : tableau des métriques et courbes précision-rappel superposées.
3. **Seuil et budget** : curseurs de seuil et de budget d'alertes k, avec mise à jour en direct des alertes, de la précision, du rappel, de F2, du MCC et du coût.
4. **Explication d'une alerte** : codes motifs et graphique SHAP waterfall.

### 3. Lancer l'API

```powershell
uvicorn api:app --reload
```

Documentation interactive : http://127.0.0.1:8000/docs

| Méthode | Endpoint | Rôle |
|---|---|---|
| GET | `/health` | Vérifier que l'API répond |
| GET | `/model-info` | Seuil, coût d'alerte, variables, catégories |
| POST | `/predict` | Probabilité de fraude et décision d'alerte |
| POST | `/explain` | Probabilité, codes motifs, valeurs SHAP (paramètre `top`) |

Exemple de corps JSON pour `/predict` et `/explain` :

```json
{
  "amt": 304.82,
  "hour": 2.5,
  "dayofweek": 3,
  "category": "grocery_pos",
  "tx_1h": 0,
  "tx_24h": 4,
  "amt_ratio_30d": 3.5,
  "hours_since_prev": 3.0,
  "dist_km": 70,
  "age": 45,
  "city_pop": 3400
}
```

Une catégorie inconnue renvoie une erreur 422 avec la liste des catégories valides.

## Méthodologie

### Variables (`src/features.py`)

Toutes les variables comportementales sont calculées **avec le seul passé de la même carte** (transaction courante exclue), à partir de la colonne datetime et non de `unix_time` (incohérent pour 559 lignes, ce qui produisait des écarts négatifs).

- Temporelles : heure en codage cyclique (sin/cos), jour de la semaine.
- Vélocité : nombre de transactions de la carte sur 1 h et sur 24 h.
- Écart au profil : montant rapporté à la moyenne des 30 derniers jours de la carte (1,0 sans historique).
- Délai depuis la transaction précédente (plafonné à 30 jours).
- Autres : log du montant, distance client-marchand, âge, log de la population de la ville, catégorie (one-hot).

Les identifiants (nom, rue, numéro de transaction) ne sont pas utilisés.

### Séparation chronologique

| Jeu | Transactions | Période | Fraudes |
|---|---|---|---|
| Entraînement | 1 111 436 | 2019-01-01 → 2020-04-07 | 6 401 (0,58 %) |
| Validation | 370 479 | 2020-04-07 → 2020-08-25 | 1 901 (0,51 %) |
| Test | 370 479 | 2020-08-25 → 2020-12-31 | 1 349 (0,36 %) |

Le taux de fraude baisse au fil du temps : il y a une dérive, d'où la confirmation du seuil sur le test. La standardisation est ajustée sur l'entraînement uniquement.

### Stratégie face au déséquilibre

Comparaison en validation pour XGBoost :

| Stratégie | PR-AUC (validation) |
|---|---|
| Sans correction | 0,9438 |
| `scale_pos_weight` = racine du ratio de classes | 0,9435 |

Aucun gain : **XGBoost sans correction est retenu**.

Pour l'Isolation Forest, entraîner sur tout le train ou sur les transactions normales seulement donne des résultats équivalents (PR-AUC 0,070 contre 0,068) : la variante sans étiquettes est retenue.

### Modèles

- **XGBoost** (supervisé) : 500 arbres maximum, profondeur 6, taux d'apprentissage 0,05, arrêt anticipé sur le PR-AUC de la validation.
- **Isolation Forest** (non supervisé) : 200 arbres, `max_samples=256`, entraîné sans étiquettes.
- **Autoencodeur** (non supervisé) : perceptron multicouche (16-6-16) de scikit-learn entraîné sur un sous-échantillon de 300 000 transactions ; le score est l'erreur de reconstruction. Le réseau n'a pas totalement convergé en 30 itérations.

Les étiquettes ne servent jamais à entraîner les méthodes non supervisées, seulement à les évaluer.

## Résultats

Budget d'alertes k = 1 852 (0,5 % des transactions, de l'ordre du nombre de fraudes attendues).

| Modèle | PR-AUC val | PR-AUC test | ROC-AUC test |
|---|---|---|---|
| Isolation Forest | 0,070 | 0,029 | 0,813 |
| Autoencodeur | 0,058 | 0,026 | 0,761 |
| **XGBoost** | **0,944** | **0,930** | **0,999** |

XGBoost au budget k = 1 852 : précision 0,897 et rappel 0,874 en validation ; précision 0,678 et rappel 0,931 sur le test. La précision plus faible sur le test s'explique par le nombre de fraudes réduit (1 349) : avec 1 852 alertes, la précision maximale possible est de 72,8 %.

### Seuil à coût minimal

Hypothèse de coût : une fraude manquée coûte son montant, une alerte coûte **5 $** d'analyse (réglable dans l'application). Le seuil est choisi sur la validation puis confirmé sur le test.

Seuil retenu : **0,0203** (probabilité XGBoost).

| | Validation | Test |
|---|---|---|
| Alertes | 3 754 | 3 053 |
| Précision | 0,487 | 0,425 |
| Rappel | 0,963 | 0,963 |
| F2 | 0,806 | 0,769 |
| MCC | 0,683 | 0,638 |
| Coût avec le modèle | 31 036 $ | 22 793 $ |
| Coût sans modèle | 989 313 $ | 733 912 $ |

Le coût est réduit d'environ 97 % sur les deux périodes. Le seuil est bas parce qu'une fraude coûte en moyenne environ 530 $ contre 5 $ par alerte : il est rentable d'enquêter dès que la probabilité dépasse environ 1 %. Deux modes de fonctionnement sont possibles : alerter tout ce qui dépasse le seuil (rappel élevé), ou traiter seulement les k meilleures alertes (précision élevée).

### Explicabilité

- **Globale** : `log_amt` domine largement (impact SHAP moyen 2,93), suivi de `amt_ratio_30d`, `hour_cos`, `gap_prev_s` et `hour_sin`. `dist_km` n'apporte pratiquement aucun signal.
- **Par alerte** : graphique SHAP waterfall et codes motifs (3 à 5 variables qui contribuent le plus), traduits en phrases lisibles pour un analyste.

## Limites

- Jeu **synthétique** : les fraudes suivent des motifs nets, donc le problème est plus facile qu'avec des données réelles. Un PR-AUC proche de 0,93 ne doit pas être extrapolé.
- **Dérive** : le taux de fraude passe de 0,58 % à 0,36 % entre l'entraînement et le test ; les détecteurs non supervisés s'en ressentent plus que XGBoost.
- Le coût d'analyse de 5 $ par alerte est une hypothèse, pas une donnée.
- Les variables SHAP corrélées (`log_amt`, `amt_ratio_30d`) se partagent le crédit : les barres ne doivent pas être lues isolément.
- L'application ne présente que la période de test ; elle n'offre pas de chargement de fichier.
- L'API reçoit les variables comportementales déjà calculées. En production, un service d'historique par carte devrait les calculer en temps réel à partir des transactions passées.
