# Plan d'Implémentation - TP IA

## Vue d'ensemble

Ce plan détaille les étapes d'implémentation pour optimiser les modèles de deep learning selon les critères d'évaluation du TP.

---

## Phase 1 : Infrastructure et Configuration ✅ TERMINÉE

### 1.1 Création d'une configuration centralisée des hyperparamètres ✅

**Fichier créé : `config_experiments.py`**

```python
EXPERIMENT_CONFIG = {
    "models": ["MLP", "LSTM", "CNN", "CNN_LSTM", "Transformer"],
    "optimizers": ["Adam", "Adagrad", "SGD"],
    "learning_rates": [0.0001, 0.001, 0.01, 0.1],
    "batch_sizes": [32, 64, 128, 256],
    "epochs": [10, 25, 50, 100],
    "architectures": {
        "MLP": {
            "hidden_layers": [[128], [256], [128, 64], [256, 128], [256, 128, 64]],
            "dropout": [0.0, 0.2, 0.3, 0.5]
        },
        "LSTM": {
            "hidden_sizes": [64, 128, 256],
            "num_layers": [1, 2, 3],
            "dropout": [0.0, 0.2, 0.3, 0.5]
        },
        "CNN": {
            "filters": [[32, 64], [64, 128], [32, 64, 128]],
            "kernel_sizes": [3, 5],
            "dropout": [0.0, 0.2, 0.3, 0.5]
        }
    }
}
```

**Points d'évaluation couverts :** 5 points (Testing different optimizer) + 5 points (Optimizing learning rate)

---

### 1.2 Système d'export des métriques ✅

**Fichier créé : `metrics_exporter.py`**

Fonctionnalités :
- [x] Export en CSV des métriques par époque (train loss, train acc, val loss, val acc)
- [x] Export en JSON des résultats finaux de chaque expérience
- [x] Sauvegarde automatique des hyperparamètres utilisés
- [x] Calcul du nombre de paramètres entraînables

Structure de sortie :
```
results/
├── experiments.csv          # Résumé de toutes les expériences
├── learning_curves/         # Courbes d'apprentissage par expérience
│   ├── MLP_exp001.csv
│   ├── LSTM_exp002.csv
│   └── ...
└── best_models/             # Meilleurs modèles sauvegardés
    ├── best_MLP.pth
    └── best_LSTM.pth
```

**Points d'évaluation couverts :** 5 points (Calculating learning curve) + 5 points (Analysing learning curves)

---

## Phase 2 : Optimisation des Architectures (Priorité Haute)

### 2.1 Optimisation MLP

**Fichier à modifier : `networks_2100078.py`**

Expériences à mener :
- [ ] Tester différentes profondeurs (1, 2, 3, 4 couches cachées)
- [ ] Tester différentes largeurs (64, 128, 256, 512 neurones)
- [ ] Tester différents taux de dropout (0, 0.2, 0.3, 0.5)
- [ ] Documenter le nombre de poids entraînables pour chaque configuration

**Points d'évaluation couverts :** 5 points (Optimizing MLP architecture)

---

### 2.2 Optimisation LSTM

**Fichier à modifier : `networks_2100078.py`**

Expériences à mener :
- [ ] Tester différents `hidden_size` (64, 128, 256)
- [ ] Tester différents `num_layers` (1, 2, 3)
- [ ] Tester bidirectionnel vs unidirectionnel
- [ ] Tester différents taux de dropout (0, 0.2, 0.3, 0.5)
- [ ] Documenter le nombre de poids entraînables pour chaque configuration

**Points d'évaluation couverts :** 5 points (Optimizing LSTM architecture)

---

### 2.3 Implémentation CNN (Bonus)

**Fichier à créer/modifier : `networks_2100078.py`**

Architecture proposée :
```python
class CNN(nn.Module):
    def __init__(self, input_size, output_size, filters, kernel_size, dropout):
        # Conv1D layers
        # BatchNorm layers
        # MaxPooling layers
        # Fully connected layers
```

Expériences :
- [ ] Tester différentes tailles de filtres
- [ ] Tester différentes tailles de kernel
- [ ] Optimiser l'architecture

**Points d'évaluation couverts :** 10 points (Implementation of CNN)

---

### 2.4 Implémentation CNN-LSTM ou Transformer (Bonus)

**Options :**

**Option A - CNN-LSTM :**
```python
class CNN_LSTM(nn.Module):
    # CNN pour extraction de features locales
    # LSTM pour séquences temporelles
```

**Option B - Transformer :**
```python
class TransformerModel(nn.Module):
    # Positional Encoding
    # Multi-Head Attention
    # Feed Forward layers
```

**Points d'évaluation couverts :** 10 points (Implementation of new architecture)

---

## Phase 3 : Entraînement et Évaluation ✅ TERMINÉE

### 3.1 Script d'entraînement automatisé ✅

**Fichier créé : `run_experiments.py`**

Fonctionnalités :
- [x] Boucle sur toutes les combinaisons d'hyperparamètres
- [x] Sauvegarde automatique des résultats
- [x] Early stopping pour éviter l'overfitting
- [x] Logging détaillé

```python
def run_experiment(model_type, config):
    # 1. Créer le modèle
    # 2. Configurer l'optimizer
    # 3. Entraîner
    # 4. Évaluer sur dev/test
    # 5. Sauvegarder les métriques
    # 6. Sauvegarder le modèle si meilleur
```

**Points d'évaluation couverts :** 5 points (Checking impact of epochs and batch size)

---

### 3.2 Utilisation de toutes les données ✅

**Fichier créé : `data_extended.py`**

Actions :
- [x] Modifier le chargement pour inclure toutes les parties (pas seulement les gagnants)
- [x] Adapter les labels en conséquence
- [x] Vérifier l'équilibrage des classes
- [x] Mode configurable (use_all_samples=True/False)

**Points d'évaluation couverts :** 5 points (Using all data)

---

### 3.3 Génération de courbes d'apprentissage ✅

**Fichier créé : `plot_learning_curves.py`**

Fonctionnalités :
- [x] Graphique Train Loss vs Val Loss par époque
- [x] Graphique Train Accuracy vs Val Accuracy par époque
- [x] Détection de l'overfitting/underfitting
- [x] Export en PNG/PDF
- [x] Heatmaps d'hyperparamètres
- [x] Comparaisons entre modèles
- [x] Rapport d'overfitting automatique

**Points d'évaluation couverts :** 5 points (Calculating learning curve) + 10 points (Analysing learning curves)

---

## Phase 4 : Métriques Avancées (Priorité Moyenne)

### 4.1 Métriques de jeu

**Fichier à créer : `game_metrics.py`**

Métriques à implémenter :
- [ ] **Game Win Ratio** : Ratio de parties gagnées par le modèle
- [ ] **Legal Move Ratio** : Ratio de coups légaux prédits par le modèle
- [ ] Matrice de confusion des coups

**Points d'évaluation couverts :** 5 points (Analysing different evaluation metrics)

---

## Phase 5 : Génération de Données (Bonus - Priorité Basse)

### 5.1 Système de génération de nouvelles parties

**Fichier à créer : `generate_data.py`**

Fonctionnalités :
- [ ] Faire jouer deux IA l'une contre l'autre
- [ ] Logger les coups et états du jeu
- [ ] Convertir les logs au format H5
- [ ] Ajouter au dataset d'entraînement

```python
def generate_games(model1, model2, num_games=1000):
    for i in range(num_games):
        game = Game()
        while not game.is_finished():
            if game.current_player == 1:
                move = model1.predict(game.state)
            else:
                move = model2.predict(game.state)
            game.play(move)
        save_game_to_h5(game)
```

**Points d'évaluation couverts :** 20 points (Generate new data)

---

## Phase 6 : Finalisation (Priorité Haute)

### 6.1 Entraînement final avec toutes les données

Actions :
- [ ] Identifier le meilleur modèle et ses hyperparamètres
- [ ] Réentraîner sur train + dev (ou train + test)
- [ ] Évaluer les performances finales

**Points d'évaluation couverts :** 5 points (Using more data in final training)

---

### 6.2 Documentation et Présentation

Actions :
- [ ] Compléter le rapport avec tous les résultats
- [ ] Créer des tableaux comparatifs
- [ ] Préparer la présentation de 5 minutes

**Points d'évaluation couverts :** 10 points (Presentation) + 5 points (Experiment design)

---

## Ordre de Priorité Recommandé

| Priorité | Tâche | Points | Temps estimé |
|----------|-------|--------|--------------|
| 1 | Baseline MLP + LSTM | 5 | 1h |
| 2 | Infrastructure métriques | 10 | 2h |
| 3 | Optimisation MLP | 5 | 2h |
| 4 | Optimisation LSTM | 5 | 2h |
| 5 | Test optimizers + learning rates | 10 | 2h |
| 6 | Test batch size + epochs | 5 | 1h |
| 7 | Courbes d'apprentissage | 15 | 2h |
| 8 | Utiliser toutes les données | 5 | 1h |
| 9 | Métriques de jeu | 5 | 2h |
| 10 | Implémentation CNN | 10 | 3h |
| 11 | CNN-LSTM ou Transformer | 10 | 4h |
| 12 | Génération de données | 20 | 5h |
| 13 | Entraînement final | 5 | 1h |
| 14 | Rapport et présentation | 15 | 3h |

**Total potentiel : 127 points** (certains bonus)

---

## Fichiers à Créer

1. ✅ `config_experiments.py` - Configuration centralisée
2. ✅ `metrics_exporter.py` - Export des métriques
3. ✅ `run_experiments.py` - Script d'entraînement automatisé
4. ✅ `plot_learning_curves.py` - Visualisation des courbes
5. ✅ `data_extended.py` - Chargement de toutes les données
6. ⏳ `game_metrics.py` - Métriques spécifiques au jeu
7. ⏳ `generate_data.py` - Génération de nouvelles données

---

## Fichiers à Modifier

1. ⏳ `networks_2100078.py` - Ajout CNN, CNN-LSTM, Transformer
2. ⏳ `training_Many2One.py` / `training_One2One.py` - Intégration du système de métriques (optionnel)

---

## Checklist de Validation

- [x] Les résultats sont évalués sur dev/test (pas train) → **Éviter -10 points**
- [x] Le nombre de paramètres est reporté pour chaque architecture
- [ ] Tous les tableaux/figures ont des nombres (en cours avec les expériences)
- [ ] Les conclusions sont logiques et justifiées (après analyse des résultats)
- [ ] La présentation respecte les 5 minutes

---

## État d'Avancement

### ✅ Phases Complétées

- **Phase 1** : Infrastructure et Configuration (10 points)
  - config_experiments.py
  - metrics_exporter.py
  
- **Phase 3** : Entraînement et Évaluation (35 points)
  - run_experiments.py
  - plot_learning_curves.py
  - data_extended.py

**Points acquis : ~45 points**

### ⏳ Phases En Cours

- **Phase 2** : Optimisation des Architectures (20-30 points)
  - Nécessite de lancer les expériences avec run_experiments.py
  - Analyse des résultats pour identifier les meilleures architectures

### 📋 Phases Restantes

- **Phase 4** : Métriques Avancées (5 points)
- **Phase 5** : Génération de Données (20 points - bonus)
- **Phase 6** : Finalisation et Présentation (20 points)

### 🚀 Prochaines Actions Recommandées

1. **Lancer les expériences baseline** :
   ```bash
   python run_experiments.py --models MLP LSTM --max-exp 10
   ```

2. **Générer les visualisations** :
   ```bash
   python plot_learning_curves.py
   ```

3. **Analyser les résultats** pour identifier :
   - Meilleurs optimizers
   - Meilleurs learning rates
   - Meilleures architectures MLP/LSTM

4. **Implémenter CNN** (Phase 2.3) si le temps le permet

5. **Implémenter game_metrics.py** (Phase 4.1)

6. **Rédiger le rapport** avec les résultats obtenus
