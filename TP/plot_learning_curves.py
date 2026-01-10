"""
Script pour tracer les courbes d'apprentissage à partir des fichiers CSV générés.

Usage:
    python plot_learning_curves.py                          # Tracer la dernière courbe
    python plot_learning_curves.py --all                    # Tracer toutes les courbes
    python plot_learning_curves.py --file learning_curves_MLP_256_128_20260107_143045.csv
    python plot_learning_curves.py --model MLP_256_128      # Toutes les courbes de ce modèle
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from glob import glob

def plot_learning_curve(csv_filepath, save_fig=True, display=False):
    """
    Trace les courbes d'apprentissage à partir d'un fichier CSV.
    
    Args:
        csv_filepath: Chemin vers le fichier CSV
        save_fig: Si True, sauvegarde la figure en PNG
    """
    # Charger les données
    df = pd.read_csv(csv_filepath)
    
    # Extraire le nom du modèle du nom de fichier
    filename = os.path.basename(csv_filepath)
    model_name = filename.replace('learning_curves_', '').rsplit('_', 2)[0]
    timestamp = filename.rsplit('_', 2)[1] + '_' + filename.rsplit('_', 2)[2].replace('.csv', '')
    
    # Créer une figure avec 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Accuracy
    ax1.plot(df['epoch'], df['train_accuracy'] * 100, 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(df['epoch'], df['dev_accuracy'] * 100, 'r-', label='Dev Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Accuracy - {model_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Annoter la meilleure accuracy dev
    best_dev_idx = df['dev_accuracy'].idxmax()
    best_dev_acc = df.loc[best_dev_idx, 'dev_accuracy'] * 100
    best_epoch = df.loc[best_dev_idx, 'epoch']
    ax1.plot(best_epoch, best_dev_acc, 'g*', markersize=15, label=f'Best Dev: {best_dev_acc:.2f}%')
    ax1.legend(fontsize=10)
    
    # Subplot 2: Loss
    ax2.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax2.plot(df['epoch'], df['dev_loss'], 'r-', label='Dev Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'Loss - {model_name}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Titre global
    fig.suptitle(f'Courbes d\'apprentissage - {model_name} ({timestamp})', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    if save_fig:
        plots_dir = 'results/plots'
        os.makedirs(plots_dir, exist_ok=True)
        plot_filepath = os.path.join(plots_dir, f'learning_curve_{model_name}_{timestamp}.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {plot_filepath}")
    
    if display:
        plt.show()


def compare_models(csv_filepaths, metric='dev_accuracy'):
    """
    Compare les courbes d'apprentissage de plusieurs modèles.
    
    Args:
        csv_filepaths: Liste de chemins vers les fichiers CSV
        metric: Métrique à comparer ('dev_accuracy', 'dev_loss', 'train_accuracy', 'train_loss')
    """
    plt.figure(figsize=(12, 6))
    
    for csv_filepath in csv_filepaths:
        df = pd.read_csv(csv_filepath)
        filename = os.path.basename(csv_filepath)
        model_name = filename.replace('learning_curves_', '').rsplit('_', 2)[0]
        timestamp = filename.rsplit('_', 2)[1] + '_' + filename.rsplit('_', 2)[2].replace('.csv', '')
        
        label = f"{model_name} ({timestamp})"
        
        if 'accuracy' in metric:
            plt.plot(df['epoch'], df[metric] * 100, label=label, linewidth=2)
            plt.ylabel('Accuracy (%)', fontsize=12)
        else:
            plt.plot(df['epoch'], df[metric], label=label, linewidth=2)
            plt.ylabel('Loss', fontsize=12)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.title(f'Comparaison - {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder
    plots_dir = 'results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    plot_filepath = os.path.join(plots_dir, f'comparison_{metric}.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Comparaison sauvegardée: {plot_filepath}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Tracer les courbes d\'apprentissage')
    parser.add_argument('--file', type=str, help='Fichier CSV spécifique à tracer')
    parser.add_argument('--model', type=str, help='Nom du modèle (trace toutes les courbes de ce modèle)')
    parser.add_argument('--all', action='store_true', help='Tracer toutes les courbes')
    parser.add_argument('--compare', action='store_true', help='Comparer plusieurs modèles')
    parser.add_argument('--metric', type=str, default='dev_accuracy', 
                       choices=['dev_accuracy', 'dev_loss', 'train_accuracy', 'train_loss'],
                       help='Métrique à comparer')
    parser.add_argument('--display', action='store_true', help='Afficher les graphiques dans une fenêtre')
    
    args = parser.parse_args()
    
    results_dir = 'results'

    display = args.display
    
    # Vérifier que le dossier results existe
    if not os.path.exists(results_dir):
        print("❌ Le dossier 'results' n'existe pas. Aucun résultat à tracer.")
        return
    
    # Trouver tous les fichiers de courbes d'apprentissage
    all_files = sorted(glob(os.path.join(results_dir, 'learning_curves_*.csv')))
    
    if len(all_files) == 0:
        print("❌ Aucun fichier de courbes d'apprentissage trouvé dans 'results/'")
        return
    
    if args.file:
        # Tracer un fichier spécifique
        filepath = args.file if os.path.isabs(args.file) else os.path.join(results_dir, args.file)
        if os.path.exists(filepath):
            plot_learning_curve(filepath)
        else:
            print(f"❌ Fichier non trouvé: {filepath}")
    
    elif args.model:
        # Tracer toutes les courbes d'un modèle spécifique
        model_files = [f for f in all_files if args.model in os.path.basename(f)]
        if len(model_files) == 0:
            print(f"❌ Aucune courbe trouvée pour le modèle '{args.model}'")
        else:
            print(f"Tracé de {len(model_files)} courbe(s) pour {args.model}")
            for filepath in model_files:
                plot_learning_curve(filepath)
    
    elif args.compare:
        # Comparer plusieurs modèles
        compare_models(all_files, metric=args.metric)
    
    elif args.all:
        # Tracer toutes les courbes
        print(f"Tracé de {len(all_files)} courbe(s)")
        for filepath in all_files:
            plot_learning_curve(filepath)
    
    else:
        # Par défaut: tracer la dernière courbe
        latest_file = all_files[-1]
        print(f"Tracé de la dernière courbe: {os.path.basename(latest_file)}")
        plot_learning_curve(csv_filepath=latest_file, save_fig=True, display=display)


if __name__ == '__main__':
    main()
