from data import CustomDatasetMany
from utile import BOARD_SIZE
from networks_2100078 import *

import time
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.linear import Linear
import csv
import os
from datetime import datetime

models = [
    # ReLU models
    CNN_32_Dropout_Gridsearch_Relu,
    CNN_64_Dropout_Gridsearch_Relu,
    CNN_32_64_Dropout_Gridsearch_Relu,
    CNN_64_128_Dropout_Gridsearch_Relu,
    CNN_32_64_128_Dropout_Gridsearch_Relu,
    CNN_64_128_256_Dropout_Gridsearch_Relu,
    # Tanh models
    CNN_32_Dropout_Gridsearch_Tanh,
    CNN_64_Dropout_Gridsearch_Tanh,
    CNN_32_64_Dropout_Gridsearch_Tanh,
    CNN_64_128_Dropout_Gridsearch_Tanh,
    CNN_32_64_128_Dropout_Gridsearch_Tanh,
    CNN_64_128_256_Dropout_Gridsearch_Tanh
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
    len_samples = 1  # CNN utilise une seule image (One2One)

    conf={}
    conf["board_size"]=BOARD_SIZE
    conf["path_save"]="save_models"
    conf['epoch']=20
    conf["earlyStopping"]=5
    conf["len_inpout_seq"]=len_samples

    # Grille de recherche
    learning_rates = [0.001, 0.005, 0.01]
    optimizers = ["Adam"]
    dropouts = [0.1, 0.2]
    batch_sizes = [64, 128]

    # Liste pour stocker tous les résultats
    all_results = []
    
    # Créer le fichier CSV global une seule fois
    csv_filename = f"results/gridsearch_cnn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs('results', exist_ok=True)
    
    # Créer le fichier et écrire l'en-tête
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['model', 'learning_rate', 'optimizer', 'dropout', 
                     'batch_size', 'best_epoch', 'training_time', 
                     'num_parameters', 'train_accuracy', 'dev_accuracy',
                     'train_loss', 'dev_loss', 'accuracy_diff_pct', 
                     'loss_diff_pct', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print(f"\n{'='*80}")
    print(f"GRID SEARCH - CNN avec Dropout et activation (ReLU/Tanh)")
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
                            num_workers=0,
                            pin_memory=False)

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
                        num_workers=0,
                        pin_memory=False)

        for ModelClass in models:
            model_name = ModelClass.__name__
            
            for dropout in dropouts:
                conf['dropout'] = dropout

                for optimizer_name in optimizers:
                    for lr in learning_rates:
                        experiment_count += 1
                        
                        print(f"\n{'='*80}")
                        print(f"Expérience {experiment_count}/{total_experiments}")
                        print(f"Modèle: {model_name}")
                        print(f"LR={lr}, Optimizer={optimizer_name}, Dropout={dropout}, Batch={batch_size}")
                        print(f"{'='*80}")

                        # Créer le modèle
                        model = ModelClass(conf).to(device)
                        
                        # Compter les paramètres
                        n = count_parameters(model)
                        print(f"Nombre de paramètres: {n:,}")

                        # Créer l'optimizer
                        if optimizer_name == "Adam":
                            opt = torch.optim.Adam(model.parameters(), lr=lr)
                        elif optimizer_name == "Adagrad":
                            opt = torch.optim.Adagrad(model.parameters(), lr=lr)
                        elif optimizer_name == "SGD":
                            opt = torch.optim.SGD(model.parameters(), lr=lr)
                        else:
                            print(f"Optimizer {optimizer_name} non reconnu, passage au suivant")
                            continue

                        # Entraînement
                        start_time = time.time()
                        try:
                            best_epoch = model.train_all(trainSet, devSet, conf['epoch'], device, opt)
                            training_time = time.time() - start_time
                            
                            # Charger le meilleur modèle pour obtenir les métriques finales
                            best_model = torch.load(model.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
                            best_model.eval()
                            
                            # Calculer les métriques finales sur train
                            train_clas_rep = best_model.evalulate(trainSet, device)
                            final_train_acc = train_clas_rep['weighted avg']['recall']
                            
                            # Calculer la loss finale sur train
                            train_loss_total = 0.0
                            train_nb_batch = 0
                            for batch, labels, _ in trainSet:
                                with torch.no_grad():
                                    outputs = best_model(batch.float().to(device))
                                    train_loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                                    train_loss_total += train_loss.item()
                                    train_nb_batch += 1
                            final_train_loss = train_loss_total / train_nb_batch if train_nb_batch > 0 else 0.0
                            
                            # Calculer les métriques finales sur dev
                            dev_clas_rep = best_model.evalulate(devSet, device)
                            final_dev_acc = dev_clas_rep['weighted avg']['recall']
                            
                            # Calculer la loss finale sur dev
                            dev_loss_total = 0.0
                            dev_nb_batch = 0
                            for batch, labels, _ in devSet:
                                with torch.no_grad():
                                    outputs = best_model(batch.float().to(device))
                                    dev_loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                                    dev_loss_total += dev_loss.item()
                                    dev_nb_batch += 1
                            final_dev_loss = dev_loss_total / dev_nb_batch if dev_nb_batch > 0 else 0.0
                            
                            # Calculer les différences en pourcentage
                            # Différence accuracy : (train - dev) / train * 100
                            accuracy_diff_pct = ((final_train_acc - final_dev_acc) / final_train_acc * 100) if final_train_acc > 0 else 0.0
                            
                            # Différence loss : (dev - train) / train * 100 (si dev_loss > train_loss, c'est du overfitting)
                            loss_diff_pct = ((final_dev_loss - final_train_loss) / final_train_loss * 100) if final_train_loss > 0 else 0.0
                            
                            print(f"\n✓ Entraînement terminé en {training_time:.2f}s")
                            print(f"  Meilleure epoch: {best_epoch}")
                            print(f"  Train Acc: {final_train_acc*100:.2f}%, Loss: {final_train_loss:.4f}")
                            print(f"  Dev Acc: {final_dev_acc*100:.2f}%, Loss: {final_dev_loss:.4f}")
                            print(f"  Écart Acc: {accuracy_diff_pct:+.2f}%, Écart Loss: {loss_diff_pct:+.2f}%")

                            # Sauvegarder les résultats
                            result = {
                                'model': model_name,
                                'learning_rate': lr,
                                'optimizer': optimizer_name,
                                'dropout': dropout,
                                'batch_size': batch_size,
                                'best_epoch': best_epoch,
                                'training_time': training_time,
                                'num_parameters': n,
                                'train_accuracy': final_train_acc,
                                'dev_accuracy': final_dev_acc,
                                'train_loss': final_train_loss,
                                'dev_loss': final_dev_loss,
                                'accuracy_diff_pct': accuracy_diff_pct,
                                'loss_diff_pct': loss_diff_pct,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            all_results.append(result)

                            # Ajouter au fichier CSV global
                            with open(csv_filename, 'a', newline='') as csvfile:
                                fieldnames = ['model', 'learning_rate', 'optimizer', 'dropout', 
                                            'batch_size', 'best_epoch', 'training_time', 
                                            'num_parameters', 'train_accuracy', 'dev_accuracy',
                                            'train_loss', 'dev_loss', 'accuracy_diff_pct', 
                                            'loss_diff_pct', 'timestamp']
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writerow(result)
                            
                            print(f"✓ Résultats sauvegardés dans {csv_filename}")
                            
                            # Libérer le meilleur modèle
                            del best_model

                        except Exception as e:
                            print(f"\n✗ Erreur lors de l'entraînement: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            continue

                        # Libérer la mémoire
                        del model
                        del opt
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Libérer les datasets après chaque batch_size
        del ds_train, ds_dev, trainSet, devSet
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'='*80}")
    print(f"GRID SEARCH TERMINÉ")
    print(f"{'='*80}")
    print(f"Total d'expériences réussies: {len(all_results)}/{total_experiments}")
    print(f"Résultats sauvegardés dans: {csv_filename}")
    print(f"{'='*80}\n")
