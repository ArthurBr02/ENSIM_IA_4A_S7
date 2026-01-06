import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import itertools
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm

from config_experiments import (
    EXPERIMENT_CONFIG, 
    OPTIMIZER_CONFIG, 
    EARLY_STOPPING_CONFIG,
    DATA_CONFIG,
    get_experiment_name
)
from metrics_exporter import MetricsExporter, EpochMetrics
from data import CustomDatasetMany
from networks_2100078 import MLP, LSTMs
from utile import BOARD_SIZE


def get_optimizer(optimizer_name: str, model_params, learning_rate: float):
    """
    Crée un optimiseur selon le nom et la learning rate
    
    Args:
        optimizer_name: Nom de l'optimiseur (Adam, Adagrad, SGD)
        model_params: Paramètres du modèle
        learning_rate: Taux d'apprentissage
        
    Returns:
        Optimiseur PyTorch
    """
    if optimizer_name == "Adam":
        return torch.optim.Adam(
            model_params, 
            lr=learning_rate,
            **OPTIMIZER_CONFIG["Adam"]
        )
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(
            model_params,
            lr=learning_rate,
            **OPTIMIZER_CONFIG["Adagrad"]
        )
    elif optimizer_name == "SGD":
        return torch.optim.SGD(
            model_params,
            lr=learning_rate,
            **OPTIMIZER_CONFIG["SGD"]
        )
    else:
        raise ValueError(f"Optimiseur inconnu: {optimizer_name}")


def create_model(model_type: str, architecture_params: Dict, device):
    """
    Crée un modèle selon le type et les paramètres d'architecture
    
    Args:
        model_type: Type de modèle (MLP, LSTM, etc.)
        architecture_params: Paramètres d'architecture
        device: Device PyTorch
        
    Returns:
        Modèle PyTorch
    """
    conf = {
        "board_size": BOARD_SIZE,
        "path_save": f"save_models_{model_type}",
        "earlyStopping": EARLY_STOPPING_CONFIG["patience"]
    }
    
    if model_type == "MLP":
        conf["len_inpout_seq"] = 1
        conf["MLP_conf"] = architecture_params
        model = MLP(conf)
        
    elif model_type == "LSTM":
        conf["len_inpout_seq"] = architecture_params.get("len_samples", 5)
        conf["LSTM_conf"] = {
            "hidden_dim": architecture_params.get("hidden_size", 128),
            "num_layers": architecture_params.get("num_layers", 2),
            "dropout": architecture_params.get("dropout", 0.3),
        }
        model = LSTMs(conf)
        
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    return model.to(device)


def train_epoch(model, train_loader, optimizer, device, loss_fn):
    """
    Entraîne le modèle pour une époque
    
    Returns:
        tuple: (loss moyenne, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch, labels, _ in tqdm(train_loader, desc="Training", leave=False):
        batch = batch.float().to(device)
        labels = labels.float().to(device)
        
        # Forward pass
        outputs = model(batch)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistiques
        total_loss += loss.item()
        predicted = outputs.argmax(dim=-1)
        target = labels.argmax(dim=-1)
        correct += (predicted == target).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def evaluate_model(model, data_loader, device, loss_fn):
    """
    Évalue le modèle sur un dataset
    
    Returns:
        tuple: (loss moyenne, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch, labels, _ in tqdm(data_loader, desc="Evaluating", leave=False):
            batch = batch.float().to(device)
            labels = labels.float().to(device)
            
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            predicted = outputs.argmax(dim=-1)
            target = labels.argmax(dim=-1)
            correct += (predicted == target).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def run_single_experiment(
    experiment_id: str,
    model_type: str,
    architecture_params: Dict,
    optimizer_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device,
    exporter: MetricsExporter
) -> Dict[str, Any]:
    """
    Lance une seule expérience d'entraînement
    
    Returns:
        Dictionnaire avec les résultats de l'expérience
    """
    print(f"\n{'='*80}")
    print(f"Expérience: {experiment_id}")
    print(f"Model: {model_type} | Optimizer: {optimizer_name} | LR: {learning_rate} | Batch: {batch_size}")
    print(f"Architecture: {architecture_params}")
    print(f"{'='*80}\n")
    
    # Créer le modèle
    model = create_model(model_type, architecture_params, device)
    
    # Compter les paramètres
    params_info = exporter.count_parameters(model)
    print(f"Paramètres entraînables: {params_info['trainable']:,}")
    
    # Créer l'optimiseur
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Tracker de métriques
    metrics_tracker = EpochMetrics()
    
    # Early stopping
    patience = EARLY_STOPPING_CONFIG["patience"]
    epochs_without_improvement = 0
    best_val_acc = 0.0
    
    # Temps de début
    start_time = time.time()
    
    # Boucle d'entraînement
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, loss_fn)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, dev_loader, device, loss_fn)
        
        # Mettre à jour les métriques
        metrics_tracker.update(train_loss, train_acc, val_loss, val_acc)
        
        # Afficher les résultats
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Sauvegarder le meilleur modèle
        if metrics_tracker.is_best_epoch():
            print(f"✓ Nouveau meilleur modèle! (Val Acc: {val_acc:.4f})")
            exporter.save_model(model, experiment_id, is_best=True)
            epochs_without_improvement = 0
            best_val_acc = val_acc
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping à l'époque {epoch} (pas d'amélioration depuis {patience} époques)")
            break
    
    # Temps total
    training_time = time.time() - start_time
    print(f"\nTemps d'entraînement: {training_time:.2f}s ({training_time/60:.2f}min)")
    
    # Sauvegarder la courbe d'apprentissage
    history = metrics_tracker.get_history()
    exporter.save_learning_curve(experiment_id, history)
    
    # Obtenir les meilleures métriques
    best_metrics = metrics_tracker.get_best_metrics()
    
    # Sauvegarder les résultats de l'expérience
    exporter.save_experiment_result(
        experiment_id=experiment_id,
        model_type=model_type,
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epoch,  # Nombre réel d'époques effectuées
        trainable_params=params_info['trainable'],
        architecture_params=architecture_params,
        best_epoch=best_metrics['epoch'],
        train_metrics={
            'loss': history['train_loss'][best_metrics['epoch']-1],
            'accuracy': history['train_acc'][best_metrics['epoch']-1]
        },
        val_metrics={
            'loss': best_metrics['val_loss'],
            'accuracy': best_metrics['val_acc']
        },
        training_time=training_time,
        notes=f"Early stopped at epoch {epoch}"
    )
    
    return {
        'experiment_id': experiment_id,
        'best_val_acc': best_val_acc,
        'best_epoch': best_metrics['epoch'],
        'training_time': training_time
    }


def run_all_experiments(
    model_types: Optional[List[str]] = None,
    max_experiments: Optional[int] = None,
    results_dir: str = "results"
):
    """
    Lance toutes les expériences selon la configuration
    
    Args:
        model_types: Liste des types de modèles à tester (None = tous)
        max_experiments: Nombre max d'expériences (None = toutes)
        results_dir: Répertoire pour sauvegarder les résultats
    """
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")
    
    # Créer l'exporteur de métriques
    exporter = MetricsExporter(results_dir=results_dir)
    
    # Charger les données une seule fois
    print("Chargement des données...")
    
    # Dataset train
    dataset_conf_train = {
        "filelist": DATA_CONFIG["train_file"],
        "len_samples": 5,  # Par défaut, sera ajusté pour MLP
        "path_dataset": DATA_CONFIG["dataset_dir"] + "/",
        "batch_size": 64  # Par défaut
    }
    
    # Dataset dev
    dataset_conf_dev = {
        "filelist": DATA_CONFIG["dev_file"],
        "len_samples": 5,
        "path_dataset": DATA_CONFIG["dataset_dir"] + "/",
        "batch_size": 64
    }
    
    # Types de modèles à tester
    if model_types is None:
        model_types = ["MLP", "LSTM"]  # Commence avec baseline
    
    experiment_count = 0
    all_results = []
    
    # Boucle sur les types de modèles
    for model_type in model_types:
        print(f"\n{'#'*80}")
        print(f"# MODÈLE: {model_type}")
        print(f"{'#'*80}\n")
        
        if model_type not in EXPERIMENT_CONFIG["architectures"]:
            print(f"Architecture non définie pour {model_type}, skip...")
            continue
        
        arch_config = EXPERIMENT_CONFIG["architectures"][model_type]
        
        # Générer toutes les combinaisons d'hyperparamètres
        # Architecture params
        arch_keys = list(arch_config.keys())
        arch_values = [arch_config[k] for k in arch_keys]
        arch_combinations = list(itertools.product(*arch_values))
        
        # Hyperparamètres généraux
        optimizers = EXPERIMENT_CONFIG["optimizers"]
        learning_rates = EXPERIMENT_CONFIG["learning_rates"]
        batch_sizes = EXPERIMENT_CONFIG["batch_sizes"]
        
        # Pour chaque combinaison
        for arch_vals in arch_combinations:
            arch_params = dict(zip(arch_keys, arch_vals))
            
            for optimizer in optimizers:
                for lr in learning_rates:
                    for batch_size in batch_sizes:
                        
                        # Vérifier la limite d'expériences
                        if max_experiments and experiment_count >= max_experiments:
                            print(f"\nLimite de {max_experiments} expériences atteinte!")
                            exporter.print_summary()
                            return all_results
                        
                        experiment_count += 1
                        
                        # Créer un ID unique
                        exp_id = get_experiment_name(
                            model_type, optimizer, lr, batch_size, arch_params
                        )
                        
                        # Ajuster len_samples selon le modèle
                        len_samples = 1 if model_type == "MLP" else arch_params.get("len_samples", 5)
                        
                        # Créer les dataloaders avec le bon batch_size
                        dataset_conf_train["len_samples"] = len_samples
                        dataset_conf_train["batch_size"] = batch_size
                        dataset_conf_dev["len_samples"] = len_samples
                        dataset_conf_dev["batch_size"] = batch_size
                        
                        print(f"\nCréation des datasets (len_samples={len_samples}, batch_size={batch_size})...")
                        ds_train = CustomDatasetMany(dataset_conf_train)
                        ds_dev = CustomDatasetMany(dataset_conf_dev)
                        
                        train_loader = DataLoader(ds_train, batch_size=batch_size)
                        dev_loader = DataLoader(ds_dev, batch_size=batch_size)
                        
                        # Lancer l'expérience
                        try:
                            result = run_single_experiment(
                                experiment_id=exp_id,
                                model_type=model_type,
                                architecture_params=arch_params,
                                optimizer_name=optimizer,
                                learning_rate=lr,
                                batch_size=batch_size,
                                num_epochs=50,  # Nombre d'époques par défaut
                                train_loader=train_loader,
                                dev_loader=dev_loader,
                                device=device,
                                exporter=exporter
                            )
                            all_results.append(result)
                            
                        except Exception as e:
                            print(f"\nErreur pendant l'expérience {exp_id}: {str(e)}")
                            continue
    
    # Afficher le résumé final
    print(f"\n{'='*80}")
    print(f"TOUTES LES EXPÉRIENCES TERMINÉES")
    print(f"{'='*80}\n")
    exporter.print_summary()
    
    # Générer le rapport final
    summary = exporter.generate_summary_report(f"{results_dir}/final_summary.json")
    print(f"\n✓ Rapport final sauvegardé dans {results_dir}/final_summary.json")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Script d'entraînement automatisé")
    parser.add_argument("--models", nargs="+", default=["MLP", "LSTM"],
                       help="Types de modèles à entraîner")
    parser.add_argument("--max-exp", type=int, default=None,
                       help="Nombre maximum d'expériences")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Répertoire pour sauvegarder les résultats")
    
    args = parser.parse_args()
    
    print(f"""
    {'='*80}
    SCRIPT D'ENTRAÎNEMENT AUTOMATISÉ
    {'='*80}
    
    Modèles: {', '.join(args.models)}
    Max expériences: {args.max_exp if args.max_exp else 'Toutes'}
    Résultats: {args.results_dir}
    
    {'='*80}
    """)
    
    results = run_all_experiments(
        model_types=args.models,
        max_experiments=args.max_exp,
        results_dir=args.results_dir
    )
    
    print(f"\n{len(results)} expériences terminées avec succès!")
