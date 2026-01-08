from data import CustomDatasetMany
from utile import BOARD_SIZE
from networks_2100078 import *

import time
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.linear import Linear
import csv
import os
from datetime import datetime

models = [
    # ReLU models
    LSTMHiddenState_Dropout_Relu_Gridsearch_64,
    LSTMHiddenState_Dropout_Relu_Gridsearch_256,
    LSTMHiddenState_Dropout_Relu_Gridsearch_512_256,
    LSTMHiddenState_Dropout_Relu_Gridsearch_512_256_128,
    # Tanh models
    LSTMHiddenState_Dropout_Tanh_Gridsearch_64,
    LSTMHiddenState_Dropout_Tanh_Gridsearch_256,
    LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256,
    LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256_128
]

torch.serialization.add_safe_globals(models)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print('Running on ' + str(device))

    # Configuration globale
    len_samples = 5

    conf={}
    conf["board_size"]=BOARD_SIZE
    conf["path_save"]="save_models"
    conf['epoch']=20
    conf["earlyStopping"]=5
    conf["len_inpout_seq"]=len_samples
    conf["LSTM_conf"]={}
    conf["LSTM_conf"]["hidden_dim"]=64

    # Grille de recherche
    learning_rates = [0.001, 0.005, 0.01]
    optimizers = ["Adam"]
    dropouts = [0.2, 0.3]
    batch_sizes = [64, 128, 256, 512]

    # Liste pour stocker tous les résultats
    all_results = []

    print(f"\n{'='*80}")
    print(f"GRID SEARCH - LSTM avec activation et softmax")
    print(f"{'='*80}")
    print(f"Modèles à tester: {len(models)}")
    print(f"Learning rates: {learning_rates}")
    print(f"Optimizers: {optimizers}")
    print(f"Dropouts: {dropouts}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Combinaisons totales par modèle: {len(learning_rates) * len(optimizers) * len(dropouts) * len(batch_sizes)}")
    print(f"Total d'entraînements: {len(models) * len(learning_rates) * len(optimizers) * len(dropouts) * len(batch_sizes)}")
    print(f"{'='*80}\n")

    experiment_count = 0
    total_experiments = len(models) * len(learning_rates) * len(optimizers) * len(dropouts) * len(batch_sizes)

    for batch_size in batch_sizes:
        dataset_conf={}  
        # self.filelist : a list of all games for train/dev/test
        dataset_conf["filelist"]="train.txt"
        #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
        dataset_conf["len_samples"]=len_samples
        dataset_conf["path_dataset"]="./dataset/"
        dataset_conf['batch_size']=batch_size

        print(f"\n{'='*80}")
        print(f"Chargement des datasets avec batch_size={batch_size}")
        print(f"{'='*80}")
        
        print("Training Dataset ... ")
        ds_train = CustomDatasetMany(dataset_conf, load_data_once4all=True)
        trainSet = DataLoader(ds_train, 
                            batch_size=dataset_conf['batch_size'],
                            num_workers=0,  # 0 car données déjà en GPU
                            pin_memory=False)  # False car données déjà dans le GPU

        dataset_conf_dev={}  
        # self.filelist : a list of all games for train/dev/test
        dataset_conf_dev["filelist"]="dev.txt"
        #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
        dataset_conf_dev["len_samples"]=len_samples
        dataset_conf_dev["path_dataset"]="./dataset/"
        dataset_conf_dev['batch_size']=batch_size

        print("Development Dataset ... ")
        ds_dev = CustomDatasetMany(dataset_conf_dev, load_data_once4all=True)
        devSet = DataLoader(ds_dev, 
                            batch_size=dataset_conf_dev['batch_size'],
                            num_workers=0,  # 0 car données déjà en GPU
                            pin_memory=False)  # False car données déjà dans le GPU
        for model_class in models:
            for dropout in dropouts:
                conf['dropout'] = dropout
                
                for optimizer_name in optimizers:
                    for lr in learning_rates:
                        experiment_count += 1
                        
                        print(f"\n{'='*80}")
                        print(f"Expérience {experiment_count}/{total_experiments}")
                        print(f"Modèle: {model_class.__name__}")
                        print(f"Batch Size: {batch_size}, Dropout: {dropout}, Optimizer: {optimizer_name}, LR: {lr}")
                        print(f"{'='*80}")
                        
                        # Informations de l'expérience
                        experiment_info = {
                            'experiment_id': experiment_count,
                            'model_name': model_class.__name__,
                            'batch_size': batch_size,
                            'dropout': dropout,
                            'optimizer': optimizer_name,
                            'learning_rate': lr,
                            'len_samples': len_samples
                        }
                        
                        # Créer le modèle
                        model = model_class(conf).to(device)
                        print(model)
                        
                        n = count_parameters(model)
                        print(f"Nombre de paramètres: {n:,}")
                        experiment_info['num_parameters'] = n
                        
                        # Créer l'optimizer
                        if optimizer_name == "Adam":
                            opt = torch.optim.Adam(model.parameters(), lr=lr)
                        elif optimizer_name == "Adagrad":
                            opt = torch.optim.Adagrad(model.parameters(), lr=lr)
                        elif optimizer_name == "SGD":
                            opt = torch.optim.SGD(model.parameters(), lr=lr)
                        else:
                            print(f"Optimizer {optimizer_name} non reconnu, skip")
                            continue
                        
                        # Entraîner
                        start_time = time.time()
                        
                        try:
                            best_epoch = model.train_all(trainSet,
                                                        devSet,
                                                        conf['epoch'],
                                                        device, opt)
                            
                            training_time = time.time() - start_time
                            
                            # Évaluer sur train et dev pour obtenir les métriques finales
                            model.eval()
                            train_metrics = model.evalulate(trainSet, device)
                            dev_metrics = model.evalulate(devSet, device)
                            
                            # Récupérer les métriques
                            experiment_info['best_epoch'] = best_epoch
                            experiment_info['training_time_sec'] = round(training_time, 2)
                            experiment_info['train_accuracy'] = train_metrics['weighted avg']['recall']
                            experiment_info['train_precision'] = train_metrics['weighted avg']['precision']
                            experiment_info['train_f1'] = train_metrics['weighted avg']['f1-score']
                            experiment_info['dev_accuracy'] = dev_metrics['weighted avg']['recall']
                            experiment_info['dev_precision'] = dev_metrics['weighted avg']['precision']
                            experiment_info['dev_f1'] = dev_metrics['weighted avg']['f1-score']
                            experiment_info['status'] = 'success'
                            
                            print(f"\n✓ Entraînement terminé en {training_time:.2f}s")
                            print(f"  Meilleure époque: {best_epoch}")
                            print(f"  Train Acc: {experiment_info['train_accuracy']*100:.2f}%")
                            print(f"  Dev Acc: {experiment_info['dev_accuracy']*100:.2f}%")
                            
                        except Exception as e:
                            print(f"\n❌ Erreur lors de l'entraînement: {e}")
                            experiment_info['status'] = 'failed'
                            experiment_info['error'] = str(e)
                        
                        # Ajouter les résultats à la liste
                        all_results.append(experiment_info)

    # Exporter tous les résultats dans un fichier CSV consolidé
    print(f"\n{'='*80}")
    print(f"EXPORT DES RÉSULTATS")
    print(f"{'='*80}\n")

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    gridsearch_csv = os.path.join(results_dir, f'gridsearch_results_{timestamp_str}.csv')

    # Définir les colonnes du CSV
    fieldnames = [
        'experiment_id',
        'model_name',
        'batch_size',
        'dropout',
        'optimizer',
        'learning_rate',
        'len_samples',
        'num_parameters',
        'best_epoch',
        'training_time_sec',
        'train_accuracy',
        'train_precision',
        'train_f1',
        'dev_accuracy',
        'dev_precision',
        'dev_f1',
        'status'
    ]

    with open(gridsearch_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for result in all_results:
            writer.writerow(result)

        print(f"✓ Résultats du grid search exportés dans: {gridsearch_csv}")
        print(f"  Nombre total d'expériences: {len(all_results)}")
        print(f"  Expériences réussies: {sum(1 for r in all_results if r.get('status') == 'success')}")
        print(f"  Expériences échouées: {sum(1 for r in all_results if r.get('status') == 'failed')}")

        # Trouver et afficher la meilleure configuration
        successful_results = [r for r in all_results if r.get('status') == 'success' and 'dev_accuracy' in r]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['dev_accuracy'])
            print(f"\n{'='*80}")
            print(f"MEILLEURE CONFIGURATION")
            print(f"{'='*80}")
            print(f"Modèle: {best_result['model_name']}")
            print(f"Batch Size: {best_result['batch_size']}")
            print(f"Dropout: {best_result['dropout']}")
            print(f"Optimizer: {best_result['optimizer']}")
            print(f"Learning Rate: {best_result['learning_rate']}")
            print(f"Dev Accuracy: {best_result['dev_accuracy']*100:.2f}%")
            print(f"Train Accuracy: {best_result['train_accuracy']*100:.2f}%")
            print(f"Best Epoch: {best_result['best_epoch']}")
            print(f"Training Time: {best_result['training_time_sec']:.2f}s")
            print(f"{'='*80}\n")

        print(f"\n{'='*80}")
        print(f"GRID SEARCH TERMINÉ")
        print(f"Total d'expériences réalisées: {experiment_count}")
        print(f"Résultats consolidés dans: {gridsearch_csv}")
        print(f"Résultats détaillés dans: results/training_results.csv")
        print(f"Courbes d'apprentissage dans: results/learning_curves_*.csv")
        print(f"{'='*80}\n")