"""
Script pour générer et visualiser les courbes d'apprentissage
Auteur: Étudiant ENSIM
Date: 2026-01-06
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import json
import numpy as np


# Configuration du style des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class LearningCurvePlotter:
    """
    Classe pour générer des visualisations des courbes d'apprentissage
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialise le plotter
        
        Args:
            results_dir: Répertoire contenant les résultats
        """
        self.results_dir = results_dir
        self.learning_curves_dir = os.path.join(results_dir, "learning_curves")
        self.plots_dir = os.path.join(results_dir, "plots")
        
        # Créer le répertoire de plots s'il n'existe pas
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def plot_single_experiment(self, experiment_id: str, save: bool = True, show: bool = False):
        """
        Génère le graphique pour une seule expérience
        
        Args:
            experiment_id: ID de l'expérience
            save: Si True, sauvegarde le graphique
            show: Si True, affiche le graphique
        """
        csv_path = os.path.join(self.learning_curves_dir, f"{experiment_id}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Fichier non trouvé: {csv_path}")
            return
        
        # Charger les données
        df = pd.read_csv(csv_path)
        
        # Créer une figure avec 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Graphique 1: Loss
        ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Loss Curve - {experiment_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Accuracy
        ax2.plot(df['epoch'], df['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(df['epoch'], df['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Accuracy Curve - {experiment_id}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.plots_dir, f"{experiment_id}_curves.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(self, experiment_ids: List[str], metric: str = 'val_acc',
                       title: Optional[str] = None, save: bool = True, show: bool = False):
        """
        Compare plusieurs expériences sur un même graphique
        
        Args:
            experiment_ids: Liste des IDs d'expériences à comparer
            metric: Métrique à afficher (train_loss, train_acc, val_loss, val_acc)
            title: Titre du graphique (optionnel)
            save: Si True, sauvegarde le graphique
            show: Si True, affiche le graphique
        """
        plt.figure(figsize=(12, 6))
        
        for exp_id in experiment_ids:
            csv_path = os.path.join(self.learning_curves_dir, f"{exp_id}.csv")
            
            if not os.path.exists(csv_path):
                print(f"Fichier non trouvé: {csv_path}")
                continue
            
            df = pd.read_csv(csv_path)
            plt.plot(df['epoch'], df[metric], label=exp_id, linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title or f'Comparison - {metric.replace("_", " ").title()}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.plots_dir, f"comparison_{metric}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique de comparaison sauvegardé: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_all_experiments(self):
        """
        Génère les graphiques pour toutes les expériences
        """
        if not os.path.exists(self.learning_curves_dir):
            print(f"Répertoire non trouvé: {self.learning_curves_dir}")
            return
        
        csv_files = [f for f in os.listdir(self.learning_curves_dir) if f.endswith('.csv')]
        
        print(f"Génération de {len(csv_files)} graphiques...")
        
        for csv_file in csv_files:
            exp_id = csv_file.replace('.csv', '')
            self.plot_single_experiment(exp_id, save=True, show=False)
        
        print(f"✓ {len(csv_files)} graphiques générés!")
    
    def analyze_overfitting(self, experiment_id: str, threshold: float = 0.05):
        """
        Analyse si un modèle est en overfitting
        
        Args:
            experiment_id: ID de l'expérience
            threshold: Seuil de différence entre train et val pour détecter l'overfitting
            
        Returns:
            dict: Résultats de l'analyse
        """
        csv_path = os.path.join(self.learning_curves_dir, f"{experiment_id}.csv")
        
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        
        # Calculer la différence moyenne entre train et val
        loss_diff = (df['train_loss'] - df['val_loss']).mean()
        acc_diff = (df['train_acc'] - df['val_acc']).mean()
        
        # Détecter la tendance
        is_overfitting_loss = loss_diff < -threshold
        is_overfitting_acc = acc_diff > threshold
        
        # Calculer la variance
        val_loss_variance = df['val_loss'].var()
        val_acc_variance = df['val_acc'].var()
        
        analysis = {
            'experiment_id': experiment_id,
            'loss_diff': loss_diff,
            'acc_diff': acc_diff,
            'is_overfitting': is_overfitting_loss or is_overfitting_acc,
            'val_loss_variance': val_loss_variance,
            'val_acc_variance': val_acc_variance,
            'final_train_acc': df['train_acc'].iloc[-1],
            'final_val_acc': df['val_acc'].iloc[-1],
            'best_val_acc': df['val_acc'].max(),
            'best_epoch': df['val_acc'].idxmax() + 1
        }
        
        return analysis
    
    def generate_overfitting_report(self, output_path: Optional[str] = None):
        """
        Génère un rapport d'analyse d'overfitting pour toutes les expériences
        
        Args:
            output_path: Chemin où sauvegarder le rapport (optionnel)
            
        Returns:
            DataFrame avec les analyses
        """
        if not os.path.exists(self.learning_curves_dir):
            print(f"Répertoire non trouvé: {self.learning_curves_dir}")
            return None
        
        csv_files = [f for f in os.listdir(self.learning_curves_dir) if f.endswith('.csv')]
        
        analyses = []
        for csv_file in csv_files:
            exp_id = csv_file.replace('.csv', '')
            analysis = self.analyze_overfitting(exp_id)
            if analysis:
                analyses.append(analysis)
        
        df = pd.DataFrame(analyses)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✓ Rapport d'overfitting sauvegardé: {output_path}")
        
        return df
    
    def plot_model_comparison_grid(self, model_types: List[str], metric: str = 'val_acc'):
        """
        Crée un grid de graphiques comparant les performances par type de modèle
        
        Args:
            model_types: Liste des types de modèles
            metric: Métrique à afficher
        """
        experiments_csv = os.path.join(self.results_dir, "experiments.csv")
        
        if not os.path.exists(experiments_csv):
            print(f"Fichier non trouvé: {experiments_csv}")
            return
        
        df = pd.read_csv(experiments_csv)
        
        # Créer un subplot pour chaque type de modèle
        n_models = len(model_types)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, model_type in enumerate(model_types):
            model_df = df[df['model_type'] == model_type]
            
            if len(model_df) == 0:
                continue
            
            # Grouper par optimizer et learning_rate
            grouped = model_df.groupby(['optimizer', 'learning_rate'])[metric].max()
            
            # Créer un barplot
            grouped.plot(kind='bar', ax=axes[idx])
            axes[idx].set_title(f'{model_type} - Best {metric}')
            axes[idx].set_xlabel('Optimizer & Learning Rate')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f"model_comparison_{metric}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique de comparaison sauvegardé: {save_path}")
        plt.close()
    
    def plot_hyperparameter_heatmap(self, model_type: str, param1: str, param2: str,
                                   metric: str = 'val_accuracy'):
        """
        Crée une heatmap montrant l'impact de deux hyperparamètres
        
        Args:
            model_type: Type de modèle
            param1: Premier hyperparamètre (ex: 'learning_rate')
            param2: Second hyperparamètre (ex: 'batch_size')
            metric: Métrique à afficher
        """
        experiments_csv = os.path.join(self.results_dir, "experiments.csv")
        
        if not os.path.exists(experiments_csv):
            print(f"Fichier non trouvé: {experiments_csv}")
            return
        
        df = pd.read_csv(experiments_csv)
        model_df = df[df['model_type'] == model_type]
        
        if len(model_df) == 0:
            print(f"Aucune expérience trouvée pour {model_type}")
            return
        
        # Créer un pivot table
        pivot = model_df.pivot_table(values=metric, index=param1, columns=param2, aggfunc='max')
        
        # Créer la heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': metric})
        plt.title(f'{model_type} - {metric} par {param1} et {param2}')
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, f"heatmap_{model_type}_{param1}_{param2}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap sauvegardée: {save_path}")
        plt.close()


def generate_all_plots(results_dir: str = "results"):
    """
    Génère tous les graphiques et analyses
    
    Args:
        results_dir: Répertoire contenant les résultats
    """
    print(f"{'='*80}")
    print("GÉNÉRATION DES GRAPHIQUES D'APPRENTISSAGE")
    print(f"{'='*80}\n")
    
    plotter = LearningCurvePlotter(results_dir)
    
    # 1. Graphiques individuels pour chaque expérience
    print("1. Génération des courbes individuelles...")
    plotter.plot_all_experiments()
    
    # 2. Rapport d'overfitting
    print("\n2. Analyse de l'overfitting...")
    overfitting_report = plotter.generate_overfitting_report(
        os.path.join(results_dir, "overfitting_analysis.csv")
    )
    
    if overfitting_report is not None:
        print(f"\nNombre d'expériences en overfitting: {overfitting_report['is_overfitting'].sum()}")
        print(f"Meilleure val_acc: {overfitting_report['best_val_acc'].max():.4f}")
    
    # 3. Comparaison par type de modèle
    print("\n3. Comparaison des modèles...")
    experiments_csv = os.path.join(results_dir, "experiments.csv")
    
    if os.path.exists(experiments_csv):
        df = pd.read_csv(experiments_csv)
        model_types = df['model_type'].unique().tolist()
        
        if len(model_types) > 0:
            plotter.plot_model_comparison_grid(model_types, metric='val_accuracy')
            
            # 4. Heatmaps pour chaque type de modèle
            print("\n4. Génération des heatmaps...")
            for model_type in model_types:
                try:
                    plotter.plot_hyperparameter_heatmap(
                        model_type, 
                        'learning_rate', 
                        'batch_size',
                        'val_accuracy'
                    )
                except Exception as e:
                    print(f"Erreur lors de la création de la heatmap pour {model_type}: {e}")
    
    print(f"\n{'='*80}")
    print("✓ GÉNÉRATION DES GRAPHIQUES TERMINÉE")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Génération des courbes d'apprentissage")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Répertoire contenant les résultats")
    parser.add_argument("--exp-id", type=str, default=None,
                       help="ID d'une expérience spécifique à visualiser")
    parser.add_argument("--show", action="store_true",
                       help="Afficher les graphiques au lieu de seulement les sauvegarder")
    
    args = parser.parse_args()
    
    if args.exp_id:
        # Afficher une seule expérience
        plotter = LearningCurvePlotter(args.results_dir)
        plotter.plot_single_experiment(args.exp_id, save=True, show=args.show)
    else:
        # Générer tous les graphiques
        generate_all_plots(args.results_dir)
