"""
Script pour lancer toutes les expériences sur les différents modèles.
Ce script teste systématiquement différentes configurations de modèles (MLP, LSTM, etc.)
avec différents hyperparamètres (optimizers, learning rates, batch sizes, etc.)

Règles à respecter:
- Tester plus de 100 modèles différents
- Tous les modèles doivent être testés sur dev/test set
- Garder les métriques: loss, accuracy, temps de calcul, nombre de victoires
- Calculer les courbes d'apprentissage
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import csv
import os
import time
from datetime import datetime
from itertools import product

from data import CustomDatasetOne, CustomDatasetMany
from utile import BOARD_SIZE
from networks_2100078 import *

MLP_MODELS = [MLP, MLP_256_128, MLP_512_256_128, MLP_512_256_128_64]
LSTM_MODELS = [LSTMs]

# ==================== CONFIGURATION ====================

# Configuration des expériences
EXPERIMENT_CONFIG = {
    # Learning rates à tester (requis: 0.0001, 0.001, 0.01, 0.1)
    "learning_rates": [0.0001, 0.001, 0.01, 0.1],
    
    # Batch sizes à tester
    "batch_sizes": [64, 128, 256],
    
    # Nombre d'époques
    "epochs": [20],
    
    # Early stopping
    "early_stopping": 10,
}

# Configuration des modèles MLP à tester
MLP_CONFIGS = {
    # Optimiseurs à tester (au moins 2 requis)
    "optimizers": ["Adam", "Adagrad", "SGD"],
    # Taux de dropout
    "dropout": [0.0, 0.2, 0.3, 0.5]
}

# Configuration des modèles LSTM à tester
LSTM_CONFIGS = {
    # Tailles de hidden state à tester
    "hidden_sizes": [64, 128, 256, 512],
    
    # Optimiseurs à tester
    "optimizers": ["Adam", "Adagrad", "SGD"],
    # Taux de dropout
    "dropout": [0.0, 0.2, 0.3, 0.5]
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

# ==================== FONCTIONS UTILITAIRES ====================

def get_device():
    """Retourne le device disponible (CUDA ou CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def create_optimizer(optimizer_name, model_parameters, learning_rate):
    """
    Crée un optimizer selon le nom spécifié
    
    Args:
        optimizer_name: Nom de l'optimizer ("Adam", "Adagrad", "SGD")
        model_parameters: Paramètres du modèle
        learning_rate: Taux d'apprentissage
    
    Returns:
        Optimizer PyTorch configuré
    """
    if optimizer_name == "Adam":
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == "Adagrad":
        return optim.Adagrad(model_parameters, lr=learning_rate)
    elif optimizer_name == "SGD":
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {optimizer_name} non reconnu")


def load_datasets(model_type, batch_size):
    """
    Charge les datasets train, dev et test
    
    Args:
        model_type: Type de modèle ("MLP" pour One2One ou "LSTM" pour Many2One)
        batch_size: Taille des batchs
    
    Returns:
        Tuple de (train_loader, dev_loader, test_loader, len_samples)
    """
    # Déterminer le type de dataset et la longueur des samples
    if model_type == "MLP":
        DatasetClass = CustomDatasetOne
        len_samples = 1
    elif model_type == "LSTM":  # LSTM
        DatasetClass = CustomDatasetMany
        len_samples = 5
    
    # Configuration dataset train
    dataset_conf_train = {
        "filelist": "train.txt",
        "len_samples": len_samples,
        "path_dataset": "./dataset/",
        "batch_size": batch_size
    }
    
    # Configuration dataset dev
    dataset_conf_dev = {
        "filelist": "dev.txt",
        "len_samples": len_samples,
        "path_dataset": "./dataset/",
        "batch_size": batch_size
    }
    
    # Configuration dataset test
    dataset_conf_test = {
        "filelist": "test.txt",
        "len_samples": len_samples,
        "path_dataset": "./dataset/",
        "batch_size": batch_size
    }
    
    # Créer les datasets
    print(f"Chargement du dataset train...")
    if model_type == "MLP":
        ds_train = DatasetClass(dataset_conf_train, load_data_once4all=True)
        ds_dev = DatasetClass(dataset_conf_dev, load_data_once4all=True)
        ds_test = DatasetClass(dataset_conf_test, load_data_once4all=True)
    else:
        ds_train = DatasetClass(dataset_conf_train)
        ds_dev = DatasetClass(dataset_conf_dev)
        ds_test = DatasetClass(dataset_conf_test)
    
    # Créer les dataloaders
    train_loader = DataLoader(ds_train, batch_size=batch_size)
    dev_loader = DataLoader(ds_dev, batch_size=batch_size)
    test_loader = DataLoader(ds_test, batch_size=batch_size)
    
    return train_loader, dev_loader, test_loader, len_samples


def save_results_to_csv(results, filename="experiment_results.csv"):
    """
    Sauvegarde les résultats des expériences dans un fichier CSV
    
    Args:
        results: Liste de dictionnaires contenant les résultats
        filename: Nom du fichier CSV
    """
    # Créer le répertoire results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    # Déterminer les colonnes
    if results:
        fieldnames = list(results[0].keys())
        
        # Écrire dans le fichier CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Résultats sauvegardés dans {filepath}")


def save_results_to_json(results, filename="experiment_results.json"):
    """
    Sauvegarde les résultats des expériences dans un fichier JSON
    
    Args:
        results: Liste de dictionnaires contenant les résultats
        filename: Nom du fichier JSON
    """
    # Créer le répertoire results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    # Écrire dans le fichier JSON
    with open(filepath, 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=4, ensure_ascii=False)
    
    print(f"Résultats sauvegardés dans {filepath}")


# ==================== FONCTIONS D'ENTRAÎNEMENT ====================

def train_and_evaluate_model(model, train_loader, dev_loader, test_loader, 
                             num_epochs, optimizer, device, experiment_info):
    """
    Entraîne et évalue un modèle, puis retourne les métriques
    
    Args:
        model: Le modèle à entraîner
        train_loader: DataLoader pour l'entraînement
        dev_loader: DataLoader pour la validation
        test_loader: DataLoader pour le test
        num_epochs: Nombre d'époques
        optimizer: Optimizer PyTorch
        device: Device (CPU ou CUDA)
        experiment_info: Informations sur l'expérience (dict)
    
    Returns:
        Dictionnaire contenant toutes les métriques
    """
    print(f"\n{'='*80}")
    print(f"Début de l'entraînement: {experiment_info['model_name']}")
    print(f"Config: {experiment_info}")
    print(f"{'='*80}\n")
    
    # Mesurer le temps d'entraînement
    training_start_time = time.time()
    
    # Entraîner le modèle
    best_epoch = model.train_all(train_loader, dev_loader, num_epochs, device, optimizer)
    
    training_time = time.time() - training_start_time
    
    # Évaluer sur le dev set
    print("\nÉvaluation sur le dev set...")
    dev_start_time = time.time()
    dev_metrics = model.evalulate(dev_loader, device)
    dev_inference_time = time.time() - dev_start_time
    
    # Évaluer sur le test set
    print("\nÉvaluation sur le test set...")
    test_start_time = time.time()
    test_metrics = model.evalulate(test_loader, device)
    test_inference_time = time.time() - test_start_time
    
    # Compiler les résultats
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": experiment_info["model_type"],
        "model_name": experiment_info["model_name"],
        "optimizer": experiment_info["optimizer"],
        "learning_rate": experiment_info["learning_rate"],
        "batch_size": experiment_info["batch_size"],
        "epochs": num_epochs,
        "best_epoch": best_epoch,
        "params_number": count_parameters(model),
        
        # Métriques sur dev set
        "dev_accuracy": dev_metrics["weighted avg"]["recall"],
        "dev_precision": dev_metrics["weighted avg"]["precision"],
        "dev_f1": dev_metrics["weighted avg"]["f1-score"],
        "dev_inference_time": dev_inference_time,
        
        # Métriques sur test set
        "test_accuracy": test_metrics["weighted avg"]["recall"],
        "test_precision": test_metrics["weighted avg"]["precision"],
        "test_f1": test_metrics["weighted avg"]["f1-score"],
        "test_inference_time": test_inference_time,
        
        # Temps d'entraînement
        "training_time": training_time,
    }
    
    # Ajouter les infos spécifiques au modèle
    if "hidden_dim" in experiment_info:
        results["hidden_dim"] = experiment_info["hidden_dim"]
    if "dropout" in experiment_info:
        results["dropout"] = experiment_info["dropout"]
    if "activation" in experiment_info:
        results["activation"] = experiment_info["activation"]
    
    print(f"\n{'='*80}")
    print(f"Résultats pour {experiment_info['model_name']}:")
    print(f"  Dev Accuracy: {results['dev_accuracy']*100:.2f}%")
    print(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"  Training Time: {results['training_time']:.2f}s")
    print(f"{'='*80}\n")
    
    return results


def run_experiments_for_model(model_class, model_type="MLP"):
    """
    Lance toutes les expériences pour un modèle spécifique
    
    Args:
        model_class: La classe du modèle à tester (ex: MLP, MLP_128_64, LSTMs, etc.)
        model_type: Le type de modèle ("MLP" ou "LSTM")
    
    Returns:
        Liste des résultats pour ce modèle
    """
    model_type_name = model_class.__name__
    
    print(f"\n{'='*80}")
    print(f"DÉBUT DES EXPÉRIENCES POUR {model_type_name}")
    print(f"{'='*80}\n")
    
    device = get_device()
    print(f"Device utilisé: {device}\n")
    
    results = []
    experiment_count = 0
    
    # Déterminer la configuration selon le type de modèle
    if model_type == "MLP":
        configs = MLP_CONFIGS
        # Calculer le nombre total d'expériences pour MLP
        total_experiments = (len(configs["optimizers"]) * 
                           len(EXPERIMENT_CONFIG["learning_rates"]) * 
                           len(EXPERIMENT_CONFIG["batch_sizes"]) * 
                           len(EXPERIMENT_CONFIG["epochs"]) *
                           len(configs["dropout"]))
        
        param_combinations = product(
            configs["optimizers"],
            EXPERIMENT_CONFIG["learning_rates"],
            EXPERIMENT_CONFIG["batch_sizes"],
            EXPERIMENT_CONFIG["epochs"],
            configs["dropout"]
        )
    else:  # LSTM
        configs = LSTM_CONFIGS
        # Calculer le nombre total d'expériences pour LSTM
        total_experiments = (len(configs["optimizers"]) * 
                           len(configs["hidden_sizes"]) *
                           len(EXPERIMENT_CONFIG["learning_rates"]) * 
                           len(EXPERIMENT_CONFIG["batch_sizes"]) * 
                           len(EXPERIMENT_CONFIG["epochs"]) *
                           len(configs["dropout"]))
        
        param_combinations = product(
            configs["optimizers"],
            configs["hidden_sizes"],
            EXPERIMENT_CONFIG["learning_rates"],
            EXPERIMENT_CONFIG["batch_sizes"],
            EXPERIMENT_CONFIG["epochs"],
            configs["dropout"]
        )
    
    print(f"Nombre total d'expériences pour {model_type_name}: {total_experiments}\n")
    
    # Parcourir toutes les combinaisons d'hyperparamètres
    for params in param_combinations:
        experiment_count += 1
        print(f"\n--- Expérience {model_type_name} {experiment_count}/{total_experiments} ---")
        
        if model_type == "MLP":
            optimizer_name, lr, batch_size, num_epochs, dropout = params
            
            # Charger les datasets
            train_loader, dev_loader, test_loader, len_samples = load_datasets("MLP", batch_size)
            
            # Créer la configuration du modèle
            model_conf = {
                "board_size": BOARD_SIZE,
                "path_save": f"save_models_{model_type_name}_{optimizer_name}_lr{lr}_bs{batch_size}_drop{dropout}",
                "epoch": num_epochs,
                "earlyStopping": EXPERIMENT_CONFIG["early_stopping"],
                "len_inpout_seq": len_samples,
                "dropout": dropout,
            }
            
            # Informations sur l'expérience
            experiment_info = {
                "model_type": "MLP",
                "model_class": model_type_name,
                "model_name": f"{model_type_name}_{optimizer_name}_lr{lr}_bs{batch_size}_drop{dropout}_ep{num_epochs}",
                "optimizer": optimizer_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "dropout": dropout,
            }
            
        else:  # LSTM
            optimizer_name, hidden_size, lr, batch_size, num_epochs, dropout = params
            
            # Charger les datasets
            train_loader, dev_loader, test_loader, len_samples = load_datasets("LSTM", batch_size)
            
            # Créer la configuration du modèle
            model_conf = {
                "board_size": BOARD_SIZE,
                "path_save": f"save_models_{model_type_name}_{optimizer_name}_lr{lr}_bs{batch_size}_hid{hidden_size}_drop{dropout}",
                "epoch": num_epochs,
                "earlyStopping": EXPERIMENT_CONFIG["early_stopping"],
                "len_inpout_seq": len_samples,
                "LSTM_conf": {
                    "hidden_dim": hidden_size,
                    "dropout": dropout
                }
            }
            
            # Informations sur l'expérience
            experiment_info = {
                "model_type": "LSTM",
                "model_class": model_type_name,
                "model_name": f"{model_type_name}_{optimizer_name}_lr{lr}_bs{batch_size}_hid{hidden_size}_drop{dropout}_ep{num_epochs}",
                "optimizer": optimizer_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "hidden_dim": hidden_size,
                "dropout": dropout
            }
        
        # Créer le modèle
        model = model_class(model_conf).to(device)

        print("Nombre de paramètres du modèle: %s" % count_parameters(model))
        
        # Créer l'optimizer
        optimizer = create_optimizer(optimizer_name, model.parameters(), lr)
        
        # Entraîner et évaluer
        try:
            result = train_and_evaluate_model(
                model, train_loader, dev_loader, test_loader,
                num_epochs, optimizer, device, experiment_info
            )
            results.append(result)
            
            # Sauvegarder les résultats intermédiaires pour ce modèle
            save_results_to_csv(results, f"{model_type_name}_results_partial.csv")
            
        except Exception as e:
            print(f"ERREUR lors de l'expérience: {e}")
            print(f"Expérience ignorée: {experiment_info}")
    
    print(f"\n{'='*80}")
    print(f"EXPÉRIENCES {model_type_name} TERMINÉES: {len(results)}/{total_experiments} réussies")
    print(f"{'='*80}\n")
    
    # Sauvegarder les résultats finaux pour ce modèle
    save_results_to_csv(results, f"{model_type_name}_results.csv")
    save_results_to_json(results, f"{model_type_name}_results.json")
    
    return results


def run_all_mlp_experiments():
    """
    Lance toutes les expériences pour tous les modèles MLP (One2One)
    
    Returns:
        Liste des résultats de tous les modèles MLP
    """
    print("\n" + "="*80)
    print("DÉBUT DES EXPÉRIENCES POUR TOUS LES MODÈLES MLP")
    print("="*80 + "\n")
    
    all_results = []
    
    # Parcourir tous les modèles MLP
    for model_class in MLP_MODELS:
        model_results = run_experiments_for_model(model_class, model_type="MLP")
        all_results.extend(model_results)
    
    print(f"\n{'='*80}")
    print(f"TOUTES LES EXPÉRIENCES MLP TERMINÉES: {len(all_results)} expériences réussies")
    print(f"{'='*80}\n")
    
    return all_results


def run_all_lstm_experiments():
    """
    Lance toutes les expériences pour tous les modèles LSTM (Many2One)
    
    Returns:
        Liste des résultats de tous les modèles LSTM
    """
    print("\n" + "="*80)
    print("DÉBUT DES EXPÉRIENCES POUR TOUS LES MODÈLES LSTM")
    print("="*80 + "\n")
    
    all_results = []
    
    # Parcourir tous les modèles LSTM
    for model_class in LSTM_MODELS:
        model_results = run_experiments_for_model(model_class, model_type="LSTM")
        all_results.extend(model_results)
    
    print(f"\n{'='*80}")
    print(f"TOUTES LES EXPÉRIENCES LSTM TERMINÉES: {len(all_results)} expériences réussies")
    print(f"{'='*80}\n")
    
    return all_results


# ==================== FONCTION PRINCIPALE ====================

def main():
    """
    Fonction principale pour lancer toutes les expériences
    """
    print("\n" + "="*80)
    print("DÉBUT DE TOUTES LES EXPÉRIENCES")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Lancer les expériences pour tous les modèles MLP
    mlp_results = run_all_mlp_experiments()
    
    # Lancer les expériences pour tous les modèles LSTM
    lstm_results = run_all_lstm_experiments()
    
    # Combiner tous les résultats
    all_results = mlp_results + lstm_results
    
    # Sauvegarder les résultats finaux globaux
    save_results_to_csv(all_results, "all_experiments_results.csv")
    save_results_to_json(all_results, "all_experiments_results.json")
    
    # Résumé final
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"TOUTES LES EXPÉRIENCES TERMINÉES")
    print(f"{'='*80}")
    print(f"Nombre total d'expériences: {len(all_results)}")
    print(f"  - MLP (tous modèles): {len(mlp_results)}")
    print(f"  - LSTM (tous modèles): {len(lstm_results)}")
    print(f"Temps total: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"{'='*80}\n")
    
    # Trouver le meilleur modèle global
    if all_results:
        best_model = max(all_results, key=lambda x: x["test_accuracy"])
        print(f"Meilleur modèle global: {best_model['model_name']}")
        print(f"  Test Accuracy: {best_model['test_accuracy']*100:.2f}%")
        print(f"  Dev Accuracy: {best_model['dev_accuracy']*100:.2f}%")
        print(f"  Training Time: {best_model['training_time']:.2f}s")
        print(f"  Nombre de paramètres: {best_model['params_number']}")
        print(f"{'='*80}\n")
        
        # Meilleur MLP
        mlp_results_filtered = [r for r in all_results if r["model_type"] == "MLP"]
        if mlp_results_filtered:
            best_mlp = max(mlp_results_filtered, key=lambda x: x["test_accuracy"])
            print(f"Meilleur modèle MLP: {best_mlp['model_name']}")
            print(f"  Test Accuracy: {best_mlp['test_accuracy']*100:.2f}%")
            print(f"{'='*80}\n")
        
        # Meilleur LSTM
        lstm_results_filtered = [r for r in all_results if r["model_type"] == "LSTM"]
        if lstm_results_filtered:
            best_lstm = max(lstm_results_filtered, key=lambda x: x["test_accuracy"])
            print(f"Meilleur modèle LSTM: {best_lstm['model_name']}")
            print(f"  Test Accuracy: {best_lstm['test_accuracy']*100:.2f}%")
            print(f"{'='*80}\n")


def main_single_model(model_class, model_type="MLP"):
    """
    Fonction principale pour lancer les expériences sur un seul modèle spécifique
    
    Args:
        model_class: La classe du modèle à tester (ex: MLP, MLP_128_64, LSTMs, etc.)
        model_type: Le type de modèle ("MLP" ou "LSTM")
    
    Example:
        # Pour tester un modèle MLP spécifique
        main_single_model(MLP_128_64, model_type="MLP")
        
        # Pour tester un modèle LSTM spécifique
        main_single_model(LSTMs, model_type="LSTM")
    """
    print("\n" + "="*80)
    print(f"DÉBUT DES EXPÉRIENCES POUR LE MODÈLE {model_class.__name__}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Lancer les expériences pour ce modèle
    results = run_experiments_for_model(model_class, model_type=model_type)
    
    # Sauvegarder les résultats
    model_name = model_class.__name__
    save_results_to_csv(results, f"{model_name}_final_results.csv")
    save_results_to_json(results, f"{model_name}_final_results.json")
    
    # Résumé final
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"EXPÉRIENCES POUR {model_name} TERMINÉES")
    print(f"{'='*80}")
    print(f"Nombre total d'expériences: {len(results)}")
    print(f"Temps total: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"{'='*80}\n")
    
    # Trouver le meilleur résultat pour ce modèle
    if results:
        best_result = max(results, key=lambda x: x["test_accuracy"])
        print(f"Meilleur configuration pour {model_name}:")
        print(f"  Test Accuracy: {best_result['test_accuracy']*100:.2f}%")
        print(f"  Dev Accuracy: {best_result['dev_accuracy']*100:.2f}%")
        print(f"  Optimizer: {best_result['optimizer']}")
        print(f"  Learning Rate: {best_result['learning_rate']}")
        print(f"  Batch Size: {best_result['batch_size']}")
        if 'dropout' in best_result:
            print(f"  Dropout: {best_result['dropout']}")
        if 'activation' in best_result:
            print(f"  Activation: {best_result['activation']}")
        if 'hidden_dim' in best_result:
            print(f"  Hidden Dim: {best_result['hidden_dim']}")
        print(f"  Training Time: {best_result['training_time']:.2f}s")
        print(f"  Nombre de paramètres: {best_result['params_number']}")
        print(f"{'='*80}\n")
    
    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # main()
    main_single_model(MLP_128_64, model_type="MLP")