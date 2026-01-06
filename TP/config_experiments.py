# Configuration principale des expériences
EXPERIMENT_CONFIG = {
    # Types de modèles à tester
    "models": ["MLP", "LSTM", "CNN", "CNN_LSTM", "Transformer"],
    
    # Optimiseurs à tester (au moins 2 requis)
    "optimizers": ["Adam", "Adagrad", "SGD"],
    
    # Learning rates à tester (requis: 0.0001, 0.001, 0.01, 0.1)
    "learning_rates": [0.0001, 0.001, 0.01, 0.1],
    
    # Batch sizes à tester
    "batch_sizes": [32, 64, 128, 256],
    
    # Nombres d'époques à tester
    "epochs": [10, 25, 50, 100],
    
    # Architectures spécifiques pour chaque type de modèle
    "architectures": {
        "MLP": {
            # Différentes configurations de couches cachées
            "hidden_layers": [
                [128],              # 1 couche
                [256],              # 1 couche
                [128, 64],          # 2 couches
                [256, 128],         # 2 couches
                [256, 128, 64],     # 3 couches
                [512, 256, 128],    # 3 couches
                [512, 256, 128, 64] # 4 couches
            ],
            # Taux de dropout à tester
            "dropout": [0.0, 0.2, 0.3, 0.5],
            # Fonctions d'activation
            "activation": ["relu", "tanh", "leaky_relu"]
        },
        
        "LSTM": {
            # Tailles de hidden state
            "hidden_sizes": [64, 128, 256, 512],
            # Nombre de couches LSTM
            "num_layers": [1, 2, 3],
            # Taux de dropout
            "dropout": [0.0, 0.2, 0.3, 0.5],
            # LSTM bidirectionnel ou non
            "bidirectional": [False, True]
        },
        
        "CNN": {
            # Configurations de filtres (nombre de canaux par couche)
            "filters": [
                [32, 64],           # 2 couches conv
                [64, 128],          # 2 couches conv
                [32, 64, 128],      # 3 couches conv
                [64, 128, 256],     # 3 couches conv
            ],
            # Tailles de kernel
            "kernel_sizes": [3, 5, 7],
            # Taux de dropout
            "dropout": [0.0, 0.2, 0.3, 0.5],
            # Pooling
            "pooling": ["max", "avg"]
        },
        
        "CNN_LSTM": {
            # Nombre de filtres CNN
            "cnn_filters": [[32, 64], [64, 128]],
            # Taille de kernel CNN
            "cnn_kernel_size": [3, 5],
            # Hidden size LSTM
            "lstm_hidden_size": [64, 128, 256],
            # Nombre de couches LSTM
            "lstm_num_layers": [1, 2],
            # Dropout
            "dropout": [0.2, 0.3, 0.5]
        },
        
        "Transformer": {
            # Dimension du modèle
            "d_model": [64, 128, 256],
            # Nombre de têtes d'attention
            "nhead": [4, 8],
            # Nombre de couches d'encodeur
            "num_encoder_layers": [2, 4, 6],
            # Dimension du feedforward
            "dim_feedforward": [256, 512, 1024],
            # Dropout
            "dropout": [0.1, 0.2, 0.3]
        }
    }
}


# Configuration des optimiseurs avec leurs paramètres par défaut
OPTIMIZER_CONFIG = {
    "Adam": {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0
    },
    "Adagrad": {
        "lr_decay": 0.0,
        "weight_decay": 0.0,
        "eps": 1e-10
    },
    "SGD": {
        "momentum": 0.0,
        "weight_decay": 0.0,
        "dampening": 0.0,
        "nesterov": False
    }
}


# Configuration pour l'early stopping
EARLY_STOPPING_CONFIG = {
    "patience": 10,          # Nombre d'époques sans amélioration avant d'arrêter
    "min_delta": 0.001,      # Amélioration minimale pour être considérée comme significative
    "monitor": "val_loss"    # Métrique à surveiller (val_loss ou val_accuracy)
}


# Configuration pour la sauvegarde des résultats
SAVE_CONFIG = {
    "results_dir": "results",
    "models_dir": "save_models",
    "experiments_csv": "results/experiments.csv",
    "learning_curves_dir": "results/learning_curves",
    "plots_dir": "results/plots",
    "best_models_dir": "results/best_models"
}


# Configuration baseline pour commencer rapidement
BASELINE_CONFIG = {
    "MLP": {
        "hidden_layers": [256, 128],
        "dropout": 0.3,
        "activation": "relu",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 50
    },
    "LSTM": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": False,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 50
    }
}


# Configuration pour le grid search
GRID_SEARCH_CONFIG = {
    "enabled": True,
    "max_experiments": 1000,  # Limite du nombre d'expériences à lancer
    "random_search": False,   # Si True, sélectionne aléatoirement parmi les configs
    "n_random_samples": 50    # Nombre d'échantillons aléatoires si random_search=True
}


# Configuration des données
DATA_CONFIG = {
    "train_file": "train.txt",
    "dev_file": "dev.txt",
    "test_file": "test.txt",
    "samples_file": "samples.txt",
    "dataset_dir": "dataset",
    "use_all_samples": False,  # Si True, utilise tous les échantillons (pas seulement gagnants)
    "train_test_split": 0.8,
    "validation_split": 0.1
}


# Configuration pour la génération de données
DATA_GENERATION_CONFIG = {
    "enabled": False,
    "num_games": 1000,
    "save_to_dataset": True,
    "model1_path": None,  # Chemin vers le premier modèle
    "model2_path": None   # Chemin vers le second modèle
}


def get_experiment_name(model_type, optimizer, learning_rate, batch_size, architecture_params):
    """
    Génère un nom unique pour une expérience
    
    Args:
        model_type: Type de modèle (MLP, LSTM, etc.)
        optimizer: Nom de l'optimiseur
        learning_rate: Taux d'apprentissage
        batch_size: Taille du batch
        architecture_params: Dictionnaire des paramètres d'architecture
        
    Returns:
        str: Nom unique de l'expérience
    """
    arch_str = "_".join([f"{k}{v}" for k, v in sorted(architecture_params.items())])
    return f"{model_type}_{optimizer}_lr{learning_rate}_bs{batch_size}_{arch_str}"


def count_total_experiments(model_type=None):
    """
    Compte le nombre total d'expériences possibles
    
    Args:
        model_type: Si spécifié, compte seulement pour ce type de modèle
        
    Returns:
        int: Nombre total d'expériences
    """
    total = 0
    models = [model_type] if model_type else EXPERIMENT_CONFIG["models"]
    
    for model in models:
        if model not in EXPERIMENT_CONFIG["architectures"]:
            continue
            
        # Compte les combinaisons d'hyperparamètres généraux
        n_optimizers = len(EXPERIMENT_CONFIG["optimizers"])
        n_lr = len(EXPERIMENT_CONFIG["learning_rates"])
        n_batch = len(EXPERIMENT_CONFIG["batch_sizes"])
        n_epochs = len(EXPERIMENT_CONFIG["epochs"])
        
        # Compte les combinaisons d'architecture spécifiques
        arch_config = EXPERIMENT_CONFIG["architectures"][model]
        n_arch = 1
        for param_values in arch_config.values():
            n_arch *= len(param_values)
        
        total += n_optimizers * n_lr * n_batch * n_epochs * n_arch
    
    return total


def get_baseline_config(model_type):
    """
    Récupère la configuration baseline pour un type de modèle
    
    Args:
        model_type: Type de modèle (MLP ou LSTM)
        
    Returns:
        dict: Configuration baseline
    """
    return BASELINE_CONFIG.get(model_type, {})


def validate_config():
    """
    Valide que la configuration respecte les exigences du TP
    
    Returns:
        tuple: (bool, list) - (est_valide, liste_erreurs)
    """
    errors = []
    
    # Vérifie qu'il y a au moins 2 optimiseurs
    if len(EXPERIMENT_CONFIG["optimizers"]) < 2:
        errors.append("Au moins 2 optimiseurs requis")
    
    # Vérifie les learning rates requis
    required_lr = [0.0001, 0.001, 0.01, 0.1]
    for lr in required_lr:
        if lr not in EXPERIMENT_CONFIG["learning_rates"]:
            errors.append(f"Learning rate {lr} manquant")
    
    # Vérifie que Adam et au moins un autre optimiseur sont présents
    if "Adam" not in EXPERIMENT_CONFIG["optimizers"]:
        errors.append("Adam doit être dans les optimiseurs")
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # Test de validation
    is_valid, errors = validate_config()
    if is_valid:
        print("✓ Configuration valide")
    else:
        print("✗ Erreurs dans la configuration:")
        for error in errors:
            print(f"  - {error}")
    
    # Affiche le nombre total d'expériences
    print(f"\nNombre total d'expériences possibles: {count_total_experiments()}")
    for model in EXPERIMENT_CONFIG["models"]:
        count = count_total_experiments(model)
        if count > 0:
            print(f"  - {model}: {count} expériences")
    
    # Affiche les configurations baseline
    print("\nConfigurations baseline:")
    for model, config in BASELINE_CONFIG.items():
        print(f"  {model}: {config}")
