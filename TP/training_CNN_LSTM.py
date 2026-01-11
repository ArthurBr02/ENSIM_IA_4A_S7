import torch
from torch.utils.data import DataLoader

from data import *
from utile import BOARD_SIZE
from networks_2100078 import *

import time



if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
print('Running on ' + str(device))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

len_samples=5

dataset_conf={}  
# self.filelist : a list of all games for train/dev/test
dataset_conf["filelist"]="train_generated.txt"
#len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
dataset_conf["len_samples"]=len_samples
dataset_conf["path_dataset"]="./generated_dataset/h5/"
dataset_conf['batch_size']=1000

print("Training Dataste ... ")
ds_train = CustomDatasetManyAugmented(dataset_conf,load_data_once4all=True)
trainSet = DataLoader(ds_train, 
                      batch_size=dataset_conf['batch_size'])

dataset_conf={}  
# self.filelist : a list of all games for train/dev/test
dataset_conf["filelist"]="dev.txt"
#len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
dataset_conf["len_samples"]=len_samples
dataset_conf["path_dataset"]="./dataset/"
dataset_conf['batch_size']=1000

print("Development Dataste ... ")
ds_dev = CustomDatasetManyAugmented(dataset_conf,load_data_once4all=True)
devSet = DataLoader(ds_dev, 
                    batch_size=dataset_conf['batch_size'])

# dataset_conf={}  
# # self.filelist : a list of all games for train/dev/test
# dataset_conf["filelist"]="train.txt"
# #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
# dataset_conf["len_samples"]=len_samples
# dataset_conf["path_dataset"]="./dataset/"
# dataset_conf['batch_size']=128

# print("Training Dataste ... ")
# ds_train = CustomDatasetMany(dataset_conf,load_data_once4all=True)
# trainSet = DataLoader(ds_train, 
#                       batch_size=dataset_conf['batch_size'])

# dataset_conf={}  
# # self.filelist : a list of all games for train/dev/test
# dataset_conf["filelist"]="dev.txt"
# #len_samples is 1 for one2one but it can be more than 1 for seq2one modeling
# dataset_conf["len_samples"]=len_samples
# dataset_conf["path_dataset"]="./dataset/"
# dataset_conf['batch_size']=128

# print("Development Dataste ... ")
# ds_dev = CustomDatasetMany(dataset_conf,load_data_once4all=True)
# devSet = DataLoader(ds_dev, 
#                     batch_size=dataset_conf['batch_size'])

conf={}
conf["board_size"]=BOARD_SIZE
conf["path_save"]="save_models"
conf['epoch']=50
conf["earlyStopping"]=10
conf["len_inpout_seq"]=len_samples
conf["LSTM_conf"]={}
conf["LSTM_conf"]["hidden_dim"]=128
conf["dropout"]=0.1

learning_rates = [0.001]
optimizers = ["Adam"]
dropouts = [0.2]
add_scheduler = True

for dropout in dropouts:
    conf['dropout'] = dropout

    for optimizer in optimizers:
        for lr in learning_rates:
            
            model = CNN_64_64_64_LSTM_Res_128_BatchNorm_5_History_Base_Dataset_Scheduler_50epochs(conf).to(device)
            print(model)

            n = count_parameters(model)
            print("Number of parameters: %s" % n)

            if optimizer == "Adam":
                opt = torch.optim.Adam(model.parameters(), lr=lr)
            elif optimizer == "Adagrad":
                opt = torch.optim.Adagrad(model.parameters(), lr=lr)
            elif optimizer == "SGD":
                opt = torch.optim.SGD(model.parameters(), lr=lr)
            else:
                print("Pas d'optimizer trouvé")
                break

            scheduler = None
            if add_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)

            start_time = time.time()
            best_epoch=model.train_all(trainSet,
                                devSet,
                                conf['epoch'],
                                device, opt, scheduler)
                                
            print("Fin entrainement", model.name, "sur", conf['epoch'], "epoch en", (time.time() - start_time, "sc"), "| Paramètres: Learning rate=", lr, "- Optimizer=", optimizer, "- Dropout=", dropout, "- Batch Size=", dataset_conf['batch_size'])
            print("best_epoch:", best_epoch)
# model = torch.load(conf["path_save"] + '/model_2.pt')
# model.eval()
# train_clas_rep=model.evalulate(trainSet, device)
# acc_train=train_clas_rep["weighted avg"]["recall"]
# print(f"Accuracy Train:{round(100*acc_train,2)}%")

