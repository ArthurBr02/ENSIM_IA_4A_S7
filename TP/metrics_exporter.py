import os
import csv
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn


class MetricsExporter:
    """
    Classe pour exporter et gérer les métriques d'entraînement
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialise l'exporteur de métriques
        
        Args:
            results_dir: Répertoire racine pour sauvegarder les résultats
        """
        self.results_dir = results_dir
        self.learning_curves_dir = os.path.join(results_dir, "learning_curves")
        self.best_models_dir = os.path.join(results_dir, "best_models")
        self.plots_dir = os.path.join(results_dir, "plots")
        self.experiments_csv = os.path.join(results_dir, "experiments.csv")
        
        # Créer les répertoires s'ils n'existent pas
        self._create_directories()
        
        # Initialiser le fichier CSV principal s'il n'existe pas
        self._init_experiments_csv()
    
    def _create_directories(self):
        """Crée tous les répertoires nécessaires"""
        for directory in [self.results_dir, self.learning_curves_dir, 
                         self.best_models_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _init_experiments_csv(self):
        """Initialise le fichier CSV principal avec les en-têtes"""
        if not os.path.exists(self.experiments_csv):
            headers = [
                'experiment_id', 'timestamp', 'model_type', 'optimizer', 
                'learning_rate', 'batch_size', 'epochs', 'trainable_params',
                'architecture_params', 'best_epoch',
                'train_loss', 'train_accuracy', 
                'val_loss', 'val_accuracy',
                'test_loss', 'test_accuracy',
                'training_time_seconds', 'notes'
            ]
            with open(self.experiments_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """
        Compte le nombre de paramètres d'un modèle
        
        Args:
            model: Modèle PyTorch
            
        Returns:
            dict: Dictionnaire avec total, trainable et non-trainable params
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    def save_learning_curve(self, experiment_id: str, history: Dict[str, List[float]]):
        """
        Sauvegarde la courbe d'apprentissage (métriques par époque)
        
        Args:
            experiment_id: Identifiant unique de l'expérience
            history: Dictionnaire contenant les listes de métriques par époque
                    Format attendu: {'train_loss': [...], 'train_acc': [...], 
                                    'val_loss': [...], 'val_acc': [...]}
        """
        csv_path = os.path.join(self.learning_curves_dir, f"{experiment_id}.csv")
        
        # Créer un DataFrame à partir de l'historique
        df = pd.DataFrame(history)
        df.insert(0, 'epoch', range(1, len(df) + 1))
        
        # Sauvegarder en CSV
        df.to_csv(csv_path, index=False)
        
        # Sauvegarder aussi en JSON pour faciliter la lecture
        json_path = os.path.join(self.learning_curves_dir, f"{experiment_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        return csv_path
    
    def save_experiment_result(self, 
                              experiment_id: str,
                              model_type: str,
                              optimizer: str,
                              learning_rate: float,
                              batch_size: int,
                              epochs: int,
                              trainable_params: int,
                              architecture_params: Dict[str, Any],
                              best_epoch: int,
                              train_metrics: Dict[str, float],
                              val_metrics: Dict[str, float],
                              test_metrics: Optional[Dict[str, float]] = None,
                              training_time: float = 0.0,
                              notes: str = ""):
        """
        Sauvegarde les résultats d'une expérience dans le CSV principal
        
        Args:
            experiment_id: Identifiant unique de l'expérience
            model_type: Type de modèle (MLP, LSTM, etc.)
            optimizer: Nom de l'optimiseur
            learning_rate: Taux d'apprentissage
            batch_size: Taille du batch
            epochs: Nombre d'époques effectuées
            trainable_params: Nombre de paramètres entraînables
            architecture_params: Paramètres d'architecture du modèle
            best_epoch: Époque avec les meilleures performances
            train_metrics: Métriques finales sur train (loss, accuracy)
            val_metrics: Métriques finales sur validation (loss, accuracy)
            test_metrics: Métriques finales sur test (loss, accuracy) - optionnel
            training_time: Temps d'entraînement en secondes
            notes: Notes additionnelles
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Préparer les métriques de test (ou mettre NA si non disponibles)
        test_loss = test_metrics.get('loss', 'NA') if test_metrics else 'NA'
        test_acc = test_metrics.get('accuracy', 'NA') if test_metrics else 'NA'
        
        # Convertir architecture_params en string JSON pour le CSV
        arch_params_str = json.dumps(architecture_params)
        
        row = [
            experiment_id,
            timestamp,
            model_type,
            optimizer,
            learning_rate,
            batch_size,
            epochs,
            trainable_params,
            arch_params_str,
            best_epoch,
            train_metrics.get('loss', 'NA'),
            train_metrics.get('accuracy', 'NA'),
            val_metrics.get('loss', 'NA'),
            val_metrics.get('accuracy', 'NA'),
            test_loss,
            test_acc,
            training_time,
            notes
        ]
        
        # Ajouter au CSV
        with open(self.experiments_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Sauvegarder aussi les détails complets en JSON
        self._save_experiment_json(
            experiment_id, model_type, optimizer, learning_rate, batch_size,
            epochs, trainable_params, architecture_params, best_epoch,
            train_metrics, val_metrics, test_metrics, training_time, notes
        )
    
    def _save_experiment_json(self, experiment_id, model_type, optimizer, learning_rate,
                             batch_size, epochs, trainable_params, architecture_params,
                             best_epoch, train_metrics, val_metrics, test_metrics,
                             training_time, notes):
        """Sauvegarde les détails complets de l'expérience en JSON"""
        json_path = os.path.join(self.results_dir, f"{experiment_id}_details.json")
        
        data = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'hyperparameters': {
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs
            },
            'architecture': architecture_params,
            'model_info': {
                'trainable_params': trainable_params
            },
            'results': {
                'best_epoch': best_epoch,
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics if test_metrics else {}
            },
            'training_time_seconds': training_time,
            'notes': notes
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def save_model(self, model: nn.Module, experiment_id: str, 
                   is_best: bool = False, epoch: Optional[int] = None):
        """
        Sauvegarde un modèle
        
        Args:
            model: Modèle PyTorch à sauvegarder
            experiment_id: Identifiant de l'expérience
            is_best: Si True, sauvegarde dans best_models
            epoch: Numéro de l'époque (optionnel)
        """
        if is_best:
            model_path = os.path.join(self.best_models_dir, f"{experiment_id}_best.pth")
        elif epoch is not None:
            model_path = os.path.join(self.results_dir, f"{experiment_id}_epoch{epoch}.pth")
        else:
            model_path = os.path.join(self.results_dir, f"{experiment_id}_final.pth")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'experiment_id': experiment_id,
            'epoch': epoch
        }, model_path)
        
        return model_path
    
    def load_experiments_dataframe(self) -> pd.DataFrame:
        """
        Charge tous les résultats d'expériences dans un DataFrame
        
        Returns:
            DataFrame pandas avec tous les résultats
        """
        if os.path.exists(self.experiments_csv):
            return pd.read_csv(self.experiments_csv)
        else:
            return pd.DataFrame()
    
    def get_best_experiments(self, metric: str = 'val_accuracy', 
                           n: int = 10, ascending: bool = False) -> pd.DataFrame:
        """
        Récupère les N meilleures expériences selon une métrique
        
        Args:
            metric: Métrique à utiliser pour le tri
            n: Nombre d'expériences à retourner
            ascending: Si True, tri croissant (pour loss), sinon décroissant (pour accuracy)
            
        Returns:
            DataFrame avec les N meilleures expériences
        """
        df = self.load_experiments_dataframe()
        if df.empty:
            return df
        
        # Trier et retourner les N meilleurs
        df_sorted = df.sort_values(by=metric, ascending=ascending)
        return df_sorted.head(n)
    
    def get_experiments_by_model(self, model_type: str) -> pd.DataFrame:
        """
        Récupère toutes les expériences pour un type de modèle
        
        Args:
            model_type: Type de modèle (MLP, LSTM, etc.)
            
        Returns:
            DataFrame avec les expériences du modèle
        """
        df = self.load_experiments_dataframe()
        if df.empty:
            return df
        
        return df[df['model_type'] == model_type]
    
    def generate_summary_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Génère un rapport résumé de toutes les expériences
        
        Args:
            output_path: Chemin où sauvegarder le rapport (optionnel)
            
        Returns:
            Dictionnaire avec les statistiques résumées
        """
        df = self.load_experiments_dataframe()
        
        if df.empty:
            return {'message': 'Aucune expérience trouvée'}
        
        summary = {
            'total_experiments': len(df),
            'models_tested': df['model_type'].unique().tolist(),
            'best_overall': {
                'by_val_accuracy': df.loc[df['val_accuracy'].idxmax()].to_dict() if 'val_accuracy' in df else None,
                'by_val_loss': df.loc[df['val_loss'].idxmin()].to_dict() if 'val_loss' in df else None
            },
            'by_model_type': {}
        }
        
        # Statistiques par type de modèle
        for model_type in df['model_type'].unique():
            model_df = df[df['model_type'] == model_type]
            summary['by_model_type'][model_type] = {
                'count': len(model_df),
                'best_val_accuracy': float(model_df['val_accuracy'].max()) if 'val_accuracy' in model_df else None,
                'best_val_loss': float(model_df['val_loss'].min()) if 'val_loss' in model_df else None,
                'avg_trainable_params': int(model_df['trainable_params'].mean()) if 'trainable_params' in model_df else None
            }
        
        # Sauvegarder le rapport si un chemin est fourni
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        
        return summary
    
    def print_summary(self):
        """Affiche un résumé des expériences dans la console"""
        df = self.load_experiments_dataframe()
        
        if df.empty:
            print("Aucune expérience trouvée.")
            return
        
        print("=" * 80)
        print(f"RÉSUMÉ DES EXPÉRIENCES ({len(df)} au total)")
        print("=" * 80)
        
        for model_type in df['model_type'].unique():
            model_df = df[df['model_type'] == model_type]
            print(f"\n{model_type}:")
            print(f"  Expériences: {len(model_df)}")
            if 'val_accuracy' in model_df.columns:
                best_idx = model_df['val_accuracy'].idxmax()
                best = model_df.loc[best_idx]
                print(f"  Meilleure val_accuracy: {best['val_accuracy']:.4f} ({best['experiment_id']})")
            if 'val_loss' in model_df.columns:
                best_idx = model_df['val_loss'].idxmin()
                best = model_df.loc[best_idx]
                print(f"  Meilleure val_loss: {best['val_loss']:.4f} ({best['experiment_id']})")


class EpochMetrics:
    """
    Classe pour tracker les métriques pendant l'entraînement
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def update(self, train_loss: float, train_acc: float, 
               val_loss: float, val_acc: float):
        """
        Met à jour les métriques pour l'époque courante
        
        Args:
            train_loss: Loss sur train
            train_acc: Accuracy sur train
            val_loss: Loss sur validation
            val_acc: Accuracy sur validation
        """
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
        self.current_epoch += 1
        
        # Mettre à jour le meilleur modèle (basé sur val_accuracy)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = self.current_epoch
    
    def is_best_epoch(self) -> bool:
        """Retourne True si l'époque courante est la meilleure"""
        return self.current_epoch == self.best_epoch
    
    def get_history(self) -> Dict[str, List[float]]:
        """Retourne l'historique complet"""
        return self.history
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de la meilleure époque"""
        return {
            'epoch': self.best_epoch,
            'val_loss': self.best_val_loss,
            'val_acc': self.best_val_acc
        }


if __name__ == "__main__":
    # Test du système d'export
    print("Test du système d'export de métriques")
    print("=" * 60)
    
    # Créer un exporteur
    exporter = MetricsExporter(results_dir="results_test")
    
    # Simuler quelques expériences
    for i in range(3):
        experiment_id = f"TEST_MLP_exp{i:03d}"
        
        # Simuler un historique d'entraînement
        history = {
            'train_loss': [0.5 - i*0.1, 0.3 - i*0.1, 0.2 - i*0.1],
            'train_acc': [0.6 + i*0.05, 0.7 + i*0.05, 0.8 + i*0.05],
            'val_loss': [0.6 - i*0.1, 0.4 - i*0.1, 0.3 - i*0.1],
            'val_acc': [0.5 + i*0.05, 0.6 + i*0.05, 0.7 + i*0.05]
        }
        
        # Sauvegarder la courbe d'apprentissage
        exporter.save_learning_curve(experiment_id, history)
        
        # Sauvegarder les résultats
        exporter.save_experiment_result(
            experiment_id=experiment_id,
            model_type="MLP",
            optimizer="Adam",
            learning_rate=0.001,
            batch_size=64,
            epochs=3,
            trainable_params=10000 + i*1000,
            architecture_params={'hidden_layers': [256, 128]},
            best_epoch=3,
            train_metrics={'loss': 0.2 - i*0.1, 'accuracy': 0.8 + i*0.05},
            val_metrics={'loss': 0.3 - i*0.1, 'accuracy': 0.7 + i*0.05},
            training_time=120.5,
            notes="Test expérience"
        )
        
        print(f"✓ Expérience {experiment_id} sauvegardée")
    
    # Afficher le résumé
    print("\n")
    exporter.print_summary()
    
    # Générer un rapport
    summary = exporter.generate_summary_report("results_test/summary_report.json")
    print(f"\n✓ Rapport résumé généré: {len(summary)} sections")
    
    print("\n✓ Test terminé avec succès!")
