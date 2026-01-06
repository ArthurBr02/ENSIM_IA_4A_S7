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
    # "epochs": [10, 25, 50, 100],
    "epochs": [20],
    
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
            "dropout": [0.0, 0.2, 0.3, 0.5]
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

def get_model(type, config):
    return

def get_optimizer(type, config):
    return