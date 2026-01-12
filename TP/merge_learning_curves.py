"""
Script pour fusionner et comparer les courbes d'apprentissage de plusieurs modèles.

Usage:
    # Comparer plusieurs fichiers spécifiques
    python merge_learning_curves.py file1.csv file2.csv file3.csv
    
    # Comparer tous les fichiers correspondant à un pattern
    python merge_learning_curves.py --pattern "CNN_32_64_128*"
    
    # Comparer les meilleurs modèles de chaque catégorie
    python merge_learning_curves.py --best-models
    
    # Spécifier les métriques à afficher
    python merge_learning_curves.py file1.csv file2.csv --metrics accuracy loss
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from glob import glob
import numpy as np

def extract_model_info(csv_filepath):
    """Extrait le nom du modèle et le timestamp du nom de fichier."""
    filename = os.path.basename(csv_filepath)
    # Enlever 'learning_curves_' au début et '.csv' à la fin
    name_parts = filename.replace('learning_curves_', '').replace('.csv', '').rsplit('_', 2)
    
    if len(name_parts) >= 3:
        model_name = name_parts[0]
        timestamp = f"{name_parts[1]}_{name_parts[2]}"
    else:
        model_name = filename.replace('.csv', '')
        timestamp = ""
    
    return model_name, timestamp


def get_best_dev_accuracy(csv_filepath):
    """Retourne la meilleure accuracy dev d'un modèle."""
    try:
        df = pd.read_csv(csv_filepath)
        return df['dev_accuracy'].max() * 100
    except:
        return 0


def plot_merged_curves(csv_filepaths, metrics=['accuracy', 'loss'], output_name='merged_comparison'):
    """
    Fusionne et affiche les courbes de plusieurs modèles sur le même graphique.
    
    Args:
        csv_filepaths: Liste des chemins vers les fichiers CSV
        metrics: Liste des métriques à afficher ('accuracy', 'loss')
        output_name: Nom du fichier de sortie
    """
    if not csv_filepaths:
        print("❌ Aucun fichier CSV fourni.")
        return
    
    print(f"\n📊 Fusion de {len(csv_filepaths)} modèles...")
    
    # Trier les fichiers par meilleure accuracy dev (descendant)
    csv_filepaths = sorted(csv_filepaths, key=get_best_dev_accuracy, reverse=True)
    
    # Palette de couleurs distinctes
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(csv_filepaths))))
    
    # Créer la figure avec subplots selon les métriques demandées
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics * 2, figsize=(7 * n_metrics * 2, 7))
    
    if n_metrics == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    
    # Liste pour stocker les légendes
    legend_handles = []
    legend_labels = []
    
    # Stocker les infos pour le résumé
    models_info = []
    
    for idx, csv_filepath in enumerate(csv_filepaths):
        try:
            # Charger les données
            df = pd.read_csv(csv_filepath)
            model_name, timestamp = extract_model_info(csv_filepath)
            
            # Calculer les stats
            best_dev_acc = df['dev_accuracy'].max() * 100
            best_epoch = df.loc[df['dev_accuracy'].idxmax(), 'epoch']
            final_dev_loss = df['dev_loss'].iloc[-1]
            
            models_info.append({
                'name': model_name,
                'timestamp': timestamp,
                'best_acc': best_dev_acc,
                'best_epoch': best_epoch,
                'final_loss': final_dev_loss,
                'epochs': len(df)
            })
            
            color = colors[idx]
            label = f"{model_name}"
            
            # Ajouter à la légende globale (une seule fois par modèle)
            if idx == len(csv_filepaths) - len(csv_filepaths) + idx:
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D([0], [0], color=color, linewidth=3))
                legend_labels.append(f"{label} ({best_dev_acc:.2f}%)")
            
            # Afficher selon les métriques demandées
            ax_idx = 0
            
            if 'accuracy' in metrics:
                # Train Accuracy
                axes[ax_idx].plot(df['epoch'], df['train_accuracy'] * 100, 
                                color=color, linestyle='--', alpha=0.5, linewidth=1.5)
                # Dev Accuracy
                axes[ax_idx].plot(df['epoch'], df['dev_accuracy'] * 100, 
                                color=color, linestyle='-', linewidth=2.5)
                
                # Marquer le meilleur point
                axes[ax_idx].scatter(best_epoch, best_dev_acc, color=color, 
                                   s=100, marker='*', zorder=5, edgecolors='black', linewidth=0.5)
                
                ax_idx += 1
            
            if 'loss' in metrics:
                # Train Loss
                axes[ax_idx].plot(df['epoch'], df['train_loss'], 
                                color=color, linestyle='--', alpha=0.5, linewidth=1.5)
                # Dev Loss
                axes[ax_idx].plot(df['epoch'], df['dev_loss'], 
                                color=color, linestyle='-', linewidth=2.5)
                ax_idx += 1
                
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement de {os.path.basename(csv_filepath)}: {e}")
            continue
    
    # Configuration des axes
    ax_idx = 0
    
    if 'accuracy' in metrics:
        axes[ax_idx].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[ax_idx].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[ax_idx].set_title('Dev Accuracy Comparison\n(ligne pleine=dev, pointillé=train)', 
                              fontsize=13, fontweight='bold')
        axes[ax_idx].grid(True, alpha=0.3, linestyle='--')
        ax_idx += 1
    
    if 'loss' in metrics:
        axes[ax_idx].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[ax_idx].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[ax_idx].set_title('Loss Comparison\n(ligne pleine=dev, pointillé=train)', 
                              fontsize=13, fontweight='bold')
        axes[ax_idx].grid(True, alpha=0.3, linestyle='--')
    
    # Titre global
    fig.suptitle(f'Comparaison de {len(csv_filepaths)} modèles', 
                fontsize=16, fontweight='bold', y=0.96)
    
    # Ajouter la légende centralisée en dessous des graphiques
    fig.legend(legend_handles, legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, -0.05),
              ncol=min(3, len(legend_labels)),
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Sauvegarder
    plots_dir = 'results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, f'{output_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique fusionné sauvegardé: {output_path}")
    
    # Afficher le résumé
    print("\n" + "="*80)
    print("📋 RÉSUMÉ DES MODÈLES COMPARÉS")
    print("="*80)
    
    models_info_sorted = sorted(models_info, key=lambda x: x['best_acc'], reverse=True)
    
    print(f"{'Rang':<6} {'Modèle':<50} {'Best Acc':<12} {'Epoch':<8} {'Final Loss':<12}")
    print("-"*80)
    for rank, info in enumerate(models_info_sorted, 1):
        print(f"{rank:<6} {info['name']:<50} {info['best_acc']:>10.2f}% {info['best_epoch']:>6} {info['final_loss']:>10.4f}")
    
    print("="*80)
    print(f"\n🏆 Meilleur modèle: {models_info_sorted[0]['name']} avec {models_info_sorted[0]['best_acc']:.2f}% d'accuracy")
    
    # Calculer les améliorations
    if len(models_info_sorted) > 1:
        improvement = models_info_sorted[0]['best_acc'] - models_info_sorted[-1]['best_acc']
        print(f"📈 Amélioration vs pire modèle: +{improvement:.2f}% d'accuracy")


def find_best_models():
    """Trouve automatiquement les meilleurs modèles de chaque catégorie."""
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        print(f"❌ Le dossier {results_dir} n'existe pas.")
        return []
    
    all_csvs = glob(os.path.join(results_dir, 'learning_curves_*.csv'))
    
    if not all_csvs:
        print(f"❌ Aucun fichier learning_curves_*.csv trouvé dans {results_dir}")
        return []
    
    # Grouper par type de modèle
    categories = {}
    for csv_file in all_csvs:
        model_name, _ = extract_model_info(csv_file)
        
        # Déterminer la catégorie
        if model_name.startswith('CNN'):
            category = 'CNN'
        elif 'LSTM' in model_name:
            category = 'LSTM'
        elif model_name.startswith('MLP'):
            category = 'MLP'
        else:
            category = 'Other'
        
        if category not in categories:
            categories[category] = []
        
        categories[category].append(csv_file)
    
    # Trouver le meilleur de chaque catégorie
    best_models = []
    print("\n🔍 Recherche des meilleurs modèles par catégorie...")
    
    for category, files in categories.items():
        if not files:
            continue
        
        best_file = max(files, key=get_best_dev_accuracy)
        best_acc = get_best_dev_accuracy(best_file)
        model_name, _ = extract_model_info(best_file)
        
        print(f"  {category}: {model_name} ({best_acc:.2f}%)")
        best_models.append(best_file)
    
    return best_models


def main():
    parser = argparse.ArgumentParser(description='Fusionner et comparer les courbes d\'apprentissage')
    parser.add_argument('files', nargs='*', help='Fichiers CSV à comparer')
    parser.add_argument('--pattern', type=str, help='Pattern pour trouver les fichiers (ex: "CNN_32*")')
    parser.add_argument('--best-models', action='store_true', 
                       help='Comparer automatiquement les meilleurs modèles de chaque catégorie')
    parser.add_argument('--metrics', nargs='+', default=['accuracy', 'loss'],
                       choices=['accuracy', 'loss'],
                       help='Métriques à afficher (accuracy, loss)')
    parser.add_argument('--output', type=str, default='merged_comparison',
                       help='Nom du fichier de sortie (sans extension)')
    
    args = parser.parse_args()
    
    csv_files = []
    
    # Mode: meilleurs modèles automatiques
    if args.best_models:
        csv_files = find_best_models()
    
    # Mode: pattern de recherche
    elif args.pattern:
        results_dir = 'results'
        pattern_path = os.path.join(results_dir, f'learning_curves_{args.pattern}*.csv')
        csv_files = glob(pattern_path)
        
        if not csv_files:
            print(f"❌ Aucun fichier trouvé pour le pattern: {args.pattern}")
            return
        
        print(f"\n✅ {len(csv_files)} fichiers trouvés pour le pattern '{args.pattern}'")
    
    # Mode: fichiers spécifiés
    elif args.files:
        for file in args.files:
            if os.path.exists(file):
                csv_files.append(file)
            else:
                # Essayer dans le dossier results
                results_path = os.path.join('results', file)
                if os.path.exists(results_path):
                    csv_files.append(results_path)
                else:
                    print(f"⚠️  Fichier non trouvé: {file}")
    
    # Aucun fichier trouvé
    else:
        print("❌ Veuillez spécifier des fichiers, un pattern ou --best-models")
        parser.print_help()
        return
    
    if not csv_files:
        print("❌ Aucun fichier valide à traiter.")
        return
    
    # Afficher les fichiers qui seront comparés
    print(f"\n📁 Fichiers à comparer ({len(csv_files)}):")
    for f in csv_files:
        model_name, timestamp = extract_model_info(f)
        best_acc = get_best_dev_accuracy(f)
        print(f"  • {model_name} ({timestamp}) - Best: {best_acc:.2f}%")
    
    # Générer le graphique fusionné
    plot_merged_curves(csv_files, metrics=args.metrics, output_name=args.output)


if __name__ == '__main__':
    main()
