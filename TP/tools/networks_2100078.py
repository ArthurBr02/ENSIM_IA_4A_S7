import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from tqdm import tqdm
import numpy as np
import os
import time
import csv
from datetime import datetime


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)

"""
[128],              # 1 couche
"""
class MLP(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP"

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[64],              # 1 couche
"""
class MLP_64(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_64, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_64"

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256, 128],              # 2 couches
"""
class MLP_256_128(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 128 -> 64 -> Output(64)

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_256_128, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_256_128/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_256_128"

        # Define the layers of the MLP: Input -> 128 -> 64 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128],    # 3 couches
"""
class MLP_512_256_128(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> 128 -> Output(64)

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_128, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_128/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_128"

        # Define the layers of the MLP: Input -> 512 -> 256 -> 128 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        x = self.lin3(x)
        outp = self.lin4(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)


"""
[512, 256, 128, 64] # 4 couches
"""
class MLP_512_256_128_64(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> 128 -> 64 -> Output(64)

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_128_64, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_128_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_128_64"

        # Define the layers of the MLP: Input -> 512 -> 256 -> 128 -> 64 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        outp = self.lin5(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
MLP Gridsearch models with Dropout and ReLU activation
"""

"""
[64],              # 1 couche
"""
class MLP_64_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 64 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_64_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_64_Dropout_Gridsearch_Relu/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_64_Dropout_Gridsearch_Relu"

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 64)
        self.lin2 = nn.Linear(64, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin2(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256],              # 1 couche
"""
class MLP_256_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 256 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_256_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_256_Dropout_Gridsearch_Relu/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_256_Dropout_Gridsearch_Relu"

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 256)
        self.lin2 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin2(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256, 128],              # 2 couches
"""
class MLP_256_128_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 256 -> 128 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_256_128_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_256_128_Dropout_Gridsearch_Relu/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_256_128_Dropout_Gridsearch_Relu"

        # Define the layers of the MLP: Input -> 256 -> 128 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin3(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256],              # 2 couches
"""
class MLP_512_256_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_Dropout_Gridsearch_Relu/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_Dropout_Gridsearch_Relu"

        # Define the layers of the MLP: Input -> 512 -> 256 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin3(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)
    
"""
[512, 256],              # 2 couches
"""
class MLP_512_256_Dropout_Relu_Post_Optimisation(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_Dropout_Relu_Post_Optimisation, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_Dropout_Relu_Post_Optimisation/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_Dropout_Relu_Post_Optimisation"

        # Define the layers of the MLP: Input -> 512 -> 256 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin3(x)

        # Softmax
        # outp = F.softmax(outp, dim=1)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)
    
"""
[512, 256],              # 2 couches
"""
class MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_50epochs(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_50epochs, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_50epochs/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_50epochs"

        # Define the layers of the MLP: Input -> 512 -> 256 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin3(x)

        # Softmax
        # outp = F.softmax(outp, dim=1)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)
    
"""
[512, 256],              # 2 couches
"""
class MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_50epochs_Generation_Data(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_50epochs_Generation_Data, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_50epochs_Generation_Data/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_50epochs_Generation_Data"

        # Define the layers of the MLP: Input -> 512 -> 256 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin3(x)

        # Softmax
        # outp = F.softmax(outp, dim=1)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)
    
"""
[512, 256],              # 2 couches
"""
class MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_200epochs(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_200epochs, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_200epochs/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_Dropout_Relu_Post_Optimisation_DataAugmentation_200epochs"

        # Define the layers of the MLP: Input -> 512 -> 256 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin3(x)

        # Softmax
        # outp = F.softmax(outp, dim=1)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256],              # 1 couche
"""
class LSTMHiddenState_Dropout_Relu_256_Post_Optimisation(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [256] - 1 couche with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Relu_256_Post_Optimisation, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Relu_256_Post_Optimisation"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Relu_256_Post_Optimisation/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 256, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(256*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = F.relu(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)
    
"""
[256],              # 1 couche
"""
class LSTMHiddenState_Dropout_Relu_256_Post_Optimisation_DataAugmentation_20epochs(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [256] - 1 couche with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Relu_256_Post_Optimisation_DataAugmentation_20epochs, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Relu_256_Post_Optimisation_DataAugmentation_20epochs"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Relu_256_Post_Optimisation_DataAugmentation_20epochs/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 256, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(256*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = F.relu(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128],              # 3 couches
"""
class MLP_512_256_128_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> 128 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_128_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_128_Dropout_Gridsearch_Relu/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_128_Dropout_Gridsearch_Relu"

        # Define the layers of the MLP: Input -> 512 -> 256 -> 128 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin4(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128, 64],              # 4 couches
"""
class MLP_512_256_128_64_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> 128 -> 64 -> Output(64) with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_128_64_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_128_64_Dropout_Gridsearch_Relu/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_128_64_Dropout_Gridsearch_Relu"

        # Define the layers of the MLP: Input -> 512 -> 256 -> 128 -> 64 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin4(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.lin5(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
MLP Gridsearch models with Dropout and Tanh activation
"""

"""
[64],              # 1 couche
"""
class MLP_64_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 64 -> Output(64) with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_64_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_64_Dropout_Gridsearch_Tanh/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_64_Dropout_Gridsearch_Tanh"

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 64)
        self.lin2 = nn.Linear(64, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        outp = self.lin2(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256],              # 1 couche
"""
class MLP_256_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 256 -> Output(64) with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_256_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_256_Dropout_Gridsearch_Tanh/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_256_Dropout_Gridsearch_Tanh"

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 256)
        self.lin2 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        outp = self.lin2(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256, 128],              # 2 couches
"""
class MLP_256_128_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 256 -> 128 -> Output(64) with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_256_128_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_256_128_Dropout_Gridsearch_Tanh/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_256_128_Dropout_Gridsearch_Tanh"

        # Define the layers of the MLP: Input -> 256 -> 128 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        outp = self.lin3(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256],              # 2 couches
"""
class MLP_512_256_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> Output(64) with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_Dropout_Gridsearch_Tanh/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_Dropout_Gridsearch_Tanh"

        # Define the layers of the MLP: Input -> 512 -> 256 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        outp = self.lin3(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128],              # 3 couches
"""
class MLP_512_256_128_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> 128 -> Output(64) with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_128_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_128_Dropout_Gridsearch_Tanh/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_128_Dropout_Gridsearch_Tanh"

        # Define the layers of the MLP: Input -> 512 -> 256 -> 128 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.lin3(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        outp = self.lin4(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128, 64],              # 4 couches
"""
class MLP_512_256_128_64_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> 128 -> 64 -> Output(64) with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_128_64_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_128_64_Dropout_Gridsearch_Tanh/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_128_64_Dropout_Gridsearch_Tanh"

        # Define the layers of the MLP: Input -> 512 -> 256 -> 128 -> 64 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.lin3(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.lin4(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        outp = self.lin5(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)


class MLP_64_Dropout_Gridsearch(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 64 -> Output(64) with Dropout

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_64_Dropout_Gridsearch, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_64_Dropout_Gridsearch/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_64_Dropout_Gridsearch"

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 64)
        self.lin2 = nn.Linear(64, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.dropout(x)
        outp = self.lin2(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256],              # 1 couche
"""
class MLP_256_Dropout_Gridsearch(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 256 -> Output(64) with Dropout

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_256_Dropout_Gridsearch, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_256_Dropout_Gridsearch/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_256_Dropout_Gridsearch"

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 256)
        self.lin2 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.dropout(x)
        outp = self.lin2(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256, 128],              # 2 couches
"""
class MLP_256_128_Dropout_Gridsearch(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 256 -> 128 -> Output(64) with Dropout

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_256_128_Dropout_Gridsearch, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_256_128_Dropout_Gridsearch/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_256_128_Dropout_Gridsearch"

        # Define the layers of the MLP: Input -> 256 -> 128 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        outp = self.lin3(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256],              # 2 couches
"""
class MLP_512_256_Dropout_Gridsearch(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> Output(64) with Dropout

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_Dropout_Gridsearch, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_Dropout_Gridsearch/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_Dropout_Gridsearch"

        # Define the layers of the MLP: Input -> 512 -> 256 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        outp = self.lin3(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128],              # 3 couches
"""
class MLP_512_256_128_Dropout_Gridsearch(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> 128 -> Output(64) with Dropout

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_128_Dropout_Gridsearch, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_128_Dropout_Gridsearch/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_128_Dropout_Gridsearch"

        # Define the layers of the MLP: Input -> 512 -> 256 -> 128 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        x = self.lin3(x)
        x = self.dropout(x)
        outp = self.lin4(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128, 64],              # 4 couches
"""
class MLP_512_256_128_64_Dropout_Gridsearch(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.
        Architecture: Input(64) -> 512 -> 256 -> 128 -> 64 -> Output(64) with Dropout

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP_512_256_128_64_Dropout_Gridsearch, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP_512_256_128_64_Dropout_Gridsearch/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        self.name = "MLP_512_256_128_64_Dropout_Gridsearch"

        # Define the layers of the MLP: Input -> 512 -> 256 -> 128 -> 64 -> Output
        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = self.lin1(seq)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        x = self.lin3(x)
        x = self.dropout(x)
        x = self.lin4(x)
        x = self.dropout(x)
        outp = self.lin5(x)
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)


class LSTMs(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMs, self).__init__()
        
        self.name = "LSTM"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTM/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

         # Define the layers of the LSTM model
        self.lstm = nn.LSTM(self.board_size*self.board_size, self.hidden_dim,batch_first=True)
        
        #1st option: using hidden states
        # self.hidden2output = nn.Linear(self.hidden_dim*2, self.board_size*self.board_size)
        
        #2nd option: using output sequence
        self.hidden2output = nn.Linear(self.hidden_dim, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        
        #1st option: using hidden states as below
        # outp = self.hidden2output(torch.cat((hn,cn),-1))
        
        #2nd option: using output sequence as below 
        #(lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[64],               # 1 couche
"""
class LSTMHiddenState_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_64, self).__init__()
        
        self.name = "LSTMHiddenState_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        
        # Using hidden states
        outp = self.hidden2output(torch.cat((hn.squeeze(0),cn.squeeze(0)),-1))
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)
    
class LSTMHiddenState_Dropout_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_64, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        # hidden_concat = F.relu(hidden_concat)  # Activation
        outp = self.hidden2output(hidden_concat)
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

class LSTMHiddenState_Dropout_Relu_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Relu_64, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Relu_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Relu_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = F.relu(hidden_concat)  # Activation
        outp = self.hidden2output(hidden_concat)
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

class LSTMHiddenState_Dropout_Tanh_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Tanh_64, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Tanh_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Tanh_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = torch.tanh(hidden_concat)  # Activation
        outp = self.hidden2output(hidden_concat)
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

class LSTMHiddenState_Dropout_Relu_Softmax_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Relu_Softmax_64, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Relu_Softmax_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Relu_Softmax_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = F.relu(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        if len(seq.shape)>2:
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

class LSTMHiddenState_Dropout_Tanh_Softmax_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Tanh_Softmax_64, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Tanh_Softmax_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Tanh_Softmax_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = torch.tanh(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        if len(seq.shape)>2:
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            outp = F.softmax(outp).squeeze()

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

class LSTMHiddenState_Dropout_Relu_Softmax_Gridsearch_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Relu_Softmax_Gridsearch_64, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Relu_Softmax_Gridsearch_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Relu_Softmax_Gridsearch_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = F.relu(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        if len(seq.shape)>2:
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

class LSTMHiddenState_Dropout_Tanh_Softmax_Gridsearch_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Tanh_Softmax_Gridsearch_64, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Tanh_Softmax_Gridsearch_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Tanh_Softmax_Gridsearch_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = torch.tanh(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        if len(seq.shape)>2:
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            outp = F.softmax(outp).squeeze()

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
Gridsearch classes with ReLU activation
"""

"""
[64],              # 1 couche
"""
class LSTMHiddenState_Dropout_Relu_Gridsearch_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Relu_Gridsearch_64, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Relu_Gridsearch_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Relu_Gridsearch_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = F.relu(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256],              # 1 couche
"""
class LSTMHiddenState_Dropout_Relu_Gridsearch_256(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [256] - 1 couche with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Relu_Gridsearch_256, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Relu_Gridsearch_256"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Relu_Gridsearch_256/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 256, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(256*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = F.relu(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256],         # 2 couches
"""
class LSTMHiddenState_Dropout_Relu_Gridsearch_512_256(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [512, 256] - 2 couches with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Relu_Gridsearch_512_256, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Relu_Gridsearch_512_256"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Relu_Gridsearch_512_256/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [512, 256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(256*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out, (hn, cn) = self.lstm2(lstm_out)
        
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = F.relu(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128],    # 3 couches
"""
class LSTMHiddenState_Dropout_Relu_Gridsearch_512_256_128(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [512, 256, 128] - 3 couches with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Relu_Gridsearch_512_256_128, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Relu_Gridsearch_512_256_128"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Relu_Gridsearch_512_256_128/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [512, 256, 128]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.lstm3 = nn.LSTM(256, 128, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(128*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out, (hn, cn) = self.lstm2(lstm_out)
        lstm_out, (hn, cn) = self.lstm3(lstm_out)
        
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = F.relu(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
Gridsearch classes with Tanh activation
"""

"""
[64],              # 1 couche
"""
class LSTMHiddenState_Dropout_Tanh_Gridsearch_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Tanh_Gridsearch_64, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Tanh_Gridsearch_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Tanh_Gridsearch_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(64*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = torch.tanh(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256],              # 1 couche
"""
class LSTMHiddenState_Dropout_Tanh_Gridsearch_256(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [256] - 1 couche with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Tanh_Gridsearch_256, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Tanh_Gridsearch_256"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Tanh_Gridsearch_256/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 256, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(256*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = torch.tanh(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256],         # 2 couches
"""
class LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [512, 256] - 2 couches with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [512, 256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(256*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out, (hn, cn) = self.lstm2(lstm_out)
        
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = torch.tanh(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128],    # 3 couches
"""
class LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256_128(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [512, 256, 128] - 3 couches with Dropout and Tanh

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256_128, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256_128"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_Tanh_Gridsearch_512_256_128/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [512, 256, 128]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.lstm3 = nn.LSTM(256, 128, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(128*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out, (hn, cn) = self.lstm2(lstm_out)
        lstm_out, (hn, cn) = self.lstm3(lstm_out)
        
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        hidden_concat = torch.tanh(hidden_concat)  # Activation
        
        # Utiliser hidden_concat au lieu de lstm_out
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256],              # 1 couche
"""
class LSTMHiddenState_256(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [256] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_256, self).__init__()
        
        self.name = "LSTMHiddenState_256"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_256/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 256, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(256*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        
        # Using hidden states
        outp = self.hidden2output(torch.cat((hn.squeeze(0),cn.squeeze(0)),-1))

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256],         # 2 couches
"""
class LSTMHiddenState_512_256(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [512, 256] - 2 couches

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_512_256, self).__init__()
        
        self.name = "LSTMHiddenState_512_256"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_512_256/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [512, 256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(256*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out, (hn, cn) = self.lstm2(lstm_out)
        
        # Using hidden states
        outp = self.hidden2output(torch.cat((hn.squeeze(0),cn.squeeze(0)),-1))

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128],    # 3 couches with Dropout
"""
class LSTMHiddenState_Dropout_512_256_128(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [512, 256, 128] - 3 couches with Dropout between layers

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_Dropout_512_256_128, self).__init__()
        
        self.name = "LSTMHiddenState_Dropout_512_256_128"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_Dropout_512_256_128/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [512, 256, 128]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.lstm3 = nn.LSTM(256, 128, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(128*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        # First LSTM layer
        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out = self.dropout(lstm_out)  # Dropout after first layer
        
        # Second LSTM layer
        lstm_out, (hn, cn) = self.lstm2(lstm_out)
        lstm_out = self.dropout(lstm_out)  # Dropout after second layer
        
        # Third LSTM layer
        lstm_out, (hn, cn) = self.lstm3(lstm_out)
        
        # Apply dropout on hidden states
        hn = self.dropout(hn.squeeze(0))
        cn = self.dropout(cn.squeeze(0))
        hidden_concat = torch.cat((hn, cn), -1)
        
        # Output layer
        outp = self.hidden2output(hidden_concat)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128],    # 3 couches
"""
class LSTMHiddenState_512_256_128(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [512, 256, 128] - 3 couches

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMHiddenState_512_256_128, self).__init__()
        
        self.name = "LSTMHiddenState_512_256_128"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMHiddenState_512_256_128/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']

        # Define the layers of the LSTM model: [512, 256, 128]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.lstm3 = nn.LSTM(256, 128, batch_first=True)
        
        # Using hidden states
        self.hidden2output = nn.Linear(128*2, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out, (hn, cn) = self.lstm2(lstm_out)
        lstm_out, (hn, cn) = self.lstm3(lstm_out)
        
        # Using hidden states
        outp = self.hidden2output(torch.cat((hn.squeeze(0),cn.squeeze(0)),-1))

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

class LSTMOutputSequence(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMOutputSequence, self).__init__()
        
        self.name = "LSTMOutputSequence"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMOutputSequence/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]
        self.conf_dropout=conf['dropout']

         # Define the layers of the LSTM model
        self.lstm = nn.LSTM(self.board_size*self.board_size, self.hidden_dim,batch_first=True)
        
        #2nd option: using output sequence
        self.hidden2output = nn.Linear(self.hidden_dim, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)

        #(lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[64],               # 1 couche
"""
class LSTMOutputSequence_64(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [64] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMOutputSequence_64, self).__init__()
        
        self.name = "LSTMOutputSequence_64"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMOutputSequence_64/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        # Define the layers of the LSTM model: [64]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 64, batch_first=True)
        
        # Using output sequence
        self.hidden2output = nn.Linear(64, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)

        # (lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[256],              # 1 couche
"""
class LSTMOutputSequence_256(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [256] - 1 couche

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMOutputSequence_256, self).__init__()
        
        self.name = "LSTMOutputSequence_256"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMOutputSequence_256/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        # Define the layers of the LSTM model: [256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 256, batch_first=True)
        
        # Using output sequence
        self.hidden2output = nn.Linear(256, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)

        # (lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256],         # 2 couches
"""
class LSTMOutputSequence_512_256(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [512, 256] - 2 couches

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMOutputSequence_512_256, self).__init__()
        
        self.name = "LSTMOutputSequence_512_256"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMOutputSequence_512_256/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        # Define the layers of the LSTM model: [512, 256]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        
        # Using output sequence
        self.hidden2output = nn.Linear(256, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out, (hn, cn) = self.lstm2(lstm_out)

        # (lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
[512, 256, 128],    # 3 couches
"""
class LSTMOutputSequence_512_256_128(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.
        Architecture: [512, 256, 128] - 3 couches

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMOutputSequence_512_256_128, self).__init__()
        
        self.name = "LSTMOutputSequence_512_256_128"

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTMOutputSequence_512_256_128/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.conf_dropout=conf['dropout']

        # Define the layers of the LSTM model: [512, 256, 128]
        self.lstm = nn.LSTM(self.board_size*self.board_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.lstm3 = nn.LSTM(256, 128, batch_first=True)
        
        # Using output sequence
        self.hidden2output = nn.Linear(128, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        lstm_out, (hn, cn) = self.lstm2(lstm_out)
        lstm_out, (hn, cn) = self.lstm3(lstm_out)

        # (lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

"""
CNN Gridsearch models with Dropout and ReLU activation
"""

"""
[32],              # 1 couche
"""
class CNN_32_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """
        Convolutional Neural Network (CNN) model for the Othello game.
        Architecture: Conv[32] -> Flatten -> Linear with Dropout and ReLU

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(CNN_32_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_32_Dropout_Gridsearch_Relu/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_32_Dropout_Gridsearch_Relu"
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # Fully connected layer
        self.fc = nn.Linear(32 * self.board_size * self.board_size, self.board_size * self.board_size)
        
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        """Forward pass of the CNN."""
        seq = np.squeeze(seq)
        
        # Reshape to (batch, 1, 8, 8)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[64],              # 1 couche
"""
class CNN_64_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """CNN model with 64 filters."""
        super(CNN_64_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_64_Dropout_Gridsearch_Relu/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_64_Dropout_Gridsearch_Relu"
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[32, 64],          # 2 couches
"""
class CNN_32_64_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """CNN model with 2 convolutional layers: 32, 64."""
        super(CNN_32_64_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_32_64_Dropout_Gridsearch_Relu/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_32_64_Dropout_Gridsearch_Relu"
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[64, 128],         # 2 couches
"""
class CNN_64_128_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """CNN model with 2 convolutional layers: 64, 128."""
        super(CNN_64_128_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_64_128_Dropout_Gridsearch_Relu/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_64_128_Dropout_Gridsearch_Relu"
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[32, 64, 128],     # 3 couches
"""
class CNN_32_64_128_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """CNN model with 3 convolutional layers: 32, 64, 128."""
        super(CNN_32_64_128_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_32_64_128_Dropout_Gridsearch_Relu/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_32_64_128_Dropout_Gridsearch_Relu"
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[32, 64, 128],     # 3 couches
"""
class CNN_32_64_128_Dropout_Gridsearch_Relu_Optimisation_DataAugmentation_20epochs(nn.Module):
    def __init__(self, conf):
        """CNN model with 3 convolutional layers: 32, 64, 128."""
        super(CNN_32_64_128_Dropout_Gridsearch_Relu_Optimisation_DataAugmentation_20epochs, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_32_64_128_Dropout_Gridsearch_Relu_Optimisation_DataAugmentation_20epochs/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_32_64_128_Dropout_Gridsearch_Relu_Optimisation_DataAugmentation_20epochs"
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[64, 128, 256],    # 3 couches
"""
class CNN_64_128_256_Dropout_Gridsearch_Relu(nn.Module):
    def __init__(self, conf):
        """CNN model with 3 convolutional layers: 64, 128, 256."""
        super(CNN_64_128_256_Dropout_Gridsearch_Relu, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_64_128_256_Dropout_Gridsearch_Relu/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_64_128_256_Dropout_Gridsearch_Relu"
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)
    
"""
[64, 128, 256],    # 3 couches
"""
class CNN_64_128_256_Dropout_Gridsearch_Relu_Optimisation_DataAugmentation_20epochs(nn.Module):
    def __init__(self, conf):
        """CNN model with 3 convolutional layers: 64, 128, 256."""
        super(CNN_64_128_256_Dropout_Gridsearch_Relu_Optimisation_DataAugmentation_20epochs, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_64_128_256_Dropout_Gridsearch_Relu_Optimisation_DataAugmentation_20epochs/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_64_128_256_Dropout_Gridsearch_Relu_Optimisation_DataAugmentation_20epochs"
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
CNN Gridsearch models with Dropout and Tanh activation
"""

"""
[32],              # 1 couche
"""
class CNN_32_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """CNN model with 32 filters and Tanh activation."""
        super(CNN_32_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_32_Dropout_Gridsearch_Tanh/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_32_Dropout_Gridsearch_Tanh"
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[64],              # 1 couche
"""
class CNN_64_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """CNN model with 64 filters and Tanh activation."""
        super(CNN_64_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_64_Dropout_Gridsearch_Tanh/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_64_Dropout_Gridsearch_Tanh"
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[32, 64],          # 2 couches
"""
class CNN_32_64_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """CNN model with 2 convolutional layers: 32, 64 and Tanh."""
        super(CNN_32_64_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_32_64_Dropout_Gridsearch_Tanh/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_32_64_Dropout_Gridsearch_Tanh"
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[64, 128],         # 2 couches
"""
class CNN_64_128_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """CNN model with 2 convolutional layers: 64, 128 and Tanh."""
        super(CNN_64_128_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_64_128_Dropout_Gridsearch_Tanh/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_64_128_Dropout_Gridsearch_Tanh"
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[32, 64, 128],     # 3 couches
"""
class CNN_32_64_128_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """CNN model with 3 convolutional layers: 32, 64, 128 and Tanh."""
        super(CNN_32_64_128_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_32_64_128_Dropout_Gridsearch_Tanh/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_32_64_128_Dropout_Gridsearch_Tanh"
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)

"""
[64, 128, 256],    # 3 couches
"""
class CNN_64_128_256_Dropout_Gridsearch_Tanh(nn.Module):
    def __init__(self, conf):
        """CNN model with 3 convolutional layers: 64, 128, 256 and Tanh."""
        super(CNN_64_128_256_Dropout_Gridsearch_Tanh, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_64_128_256_Dropout_Gridsearch_Tanh/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.conf_dropout = conf['dropout']
        
        self.name = "CNN_64_128_256_Dropout_Gridsearch_Tanh"
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.conf_dropout)
        
    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) == 2:
            seq = seq.unsqueeze(0).unsqueeze(0)
        elif len(seq.shape) == 3:
            seq = seq.unsqueeze(1)
        
        x = self.conv1(seq)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        outp = self.fc(x)
        
        return outp.squeeze() if outp.size(0) == 1 else outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    
    def evalulate(self, test_loader, device):
        return evaluate(self, test_loader, device)


def train_all(self, train, dev, num_epoch, device, optimizer):
    if not os.path.exists(f"{self.path_save}"):
        os.mkdir(f"{self.path_save}")
    best_dev = 0.0
    best_dev_loss = float('inf')
    dev_epoch = 0
    notchange=0
    train_acc_list=[]
    dev_acc_list=[]
    train_loss_list=[]
    dev_loss_list=[]
    torch.autograd.set_detect_anomaly(True)
    init_time=time.time()
    best_model_name=""
    for epoch in range(1, num_epoch+1):
        start_time=time.time()
        loss = 0.0
        nb_batch =  0
        loss_batch = 0
        for batch, labels, _ in tqdm(train):
            outputs =self(batch.float().to(device))
            loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            nb_batch += 1
            loss_batch += loss.item()
        
        avg_train_loss = loss_batch/nb_batch
        train_loss_list.append(avg_train_loss)
        
        print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                str(avg_train_loss))
        last_training=time.time()-start_time

        self.eval()
        
        train_clas_rep=self.evalulate(train, device)
        acc_train=train_clas_rep["weighted avg"]["recall"]
        train_acc_list.append(acc_train)
        
        # Calculer la loss sur dev
        dev_loss_total = 0.0
        dev_nb_batch = 0
        for batch, labels, _ in dev:
            with torch.no_grad():
                outputs = self(batch.float().to(device))
                dev_loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                dev_loss_total += dev_loss.item()
                dev_nb_batch += 1
        avg_dev_loss = dev_loss_total / dev_nb_batch if dev_nb_batch > 0 else 0.0
        dev_loss_list.append(avg_dev_loss)
        
        dev_clas_rep=self.evalulate(dev, device)
        acc_dev=dev_clas_rep["weighted avg"]["recall"]
        dev_acc_list.append(acc_dev)
        
        last_prediction=time.time()-last_training-start_time
        
        print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                f"Time:{round(time.time()-init_time)}",
                f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

        if acc_dev > best_dev or best_dev == 0.0:
            notchange=0
            best_model_name = self.path_save + '/model_' + str(epoch) + '_' + str(time.time()) + '_' + str(acc_dev) + 'prct.pt'
            torch.save(self, best_model_name)
            best_dev = acc_dev
            best_dev_loss = avg_dev_loss
            best_epoch = epoch
        else:
            notchange+=1
            if notchange>self.earlyStopping:
                break
            
        self.train()
        
        print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

    self = torch.load(best_model_name, weights_only=False)
    self.eval()
    _clas_rep = self.evalulate(dev, device)
    final_dev_acc = _clas_rep['weighted avg']['recall']
    print(f"Recalculing the best DEV: WAcc : {100*final_dev_acc}%")

    total_training_time = time.time() - init_time
    
    # Sauvegarder automatiquement les résultats
    _save_training_results(
        model_name=self.name,
        best_epoch=best_epoch,
        epochs_completed=epoch,
        best_dev_accuracy=final_dev_acc,
        best_dev_loss=best_dev_loss,
        total_training_time=total_training_time,
        train_acc_history=train_acc_list,
        dev_acc_history=dev_acc_list,
        train_loss_history=train_loss_list,
        dev_loss_history=dev_loss_list,
        optimizer_name=optimizer.__class__.__name__,
        learning_rate=optimizer.param_groups[0]['lr']
    )
    
    return best_epoch

def evaluate(self,test_loader, device):
    all_predicts=[]
    all_targets=[]
    
    for data, target_array,lengths in tqdm(test_loader):
        output = self(data.float().to(device))
        predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
        target=target_array.argmax(dim=-1).numpy()
        for i in range(len(predicted)):
            all_predicts.append(predicted[i])
            all_targets.append(target[i])
                        
    perf_rep=classification_report(all_targets,
                                    all_predicts,
                                    zero_division=1,
                                    digits=4,
                                    output_dict=True)
    perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
    
    return perf_rep

def _save_training_results(model_name, best_epoch, epochs_completed, best_dev_accuracy, 
                           best_dev_loss, total_training_time, train_acc_history, 
                           dev_acc_history, train_loss_history, dev_loss_history,
                           optimizer_name, learning_rate):
    """
    Fonction interne pour sauvegarder automatiquement les résultats d'entraînement.
    """
    # Créer le répertoire results s'il n'existe pas
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Sauvegarder dans un fichier CSV (résumé)
    csv_filepath = os.path.join(results_dir, 'training_results.csv')
    file_exists = os.path.isfile(csv_filepath)
    
    with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'timestamp',
            'model_name',
            'optimizer',
            'learning_rate',
            'best_epoch',
            'epochs_completed',
            'best_dev_accuracy',
            'best_dev_loss',
            'total_training_time_sec'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'optimizer': optimizer_name,
            'learning_rate': learning_rate,
            'best_epoch': best_epoch,
            'epochs_completed': epochs_completed,
            'best_dev_accuracy': best_dev_accuracy,
            'best_dev_loss': best_dev_loss,
            'total_training_time_sec': round(total_training_time, 2)
        })
    
    print(f"Résultats sauvegardés dans {csv_filepath}")
    
    # Sauvegarder les courbes d'apprentissage (epoch par epoch) dans un CSV
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    learning_curves_filepath = os.path.join(results_dir, f'learning_curves_{model_name}_{timestamp_str}.csv')
    
    with open(learning_curves_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['epoch', 'train_accuracy', 'dev_accuracy', 'train_loss', 'dev_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(train_acc_history)):
            writer.writerow({
                'epoch': i + 1,
                'train_accuracy': train_acc_history[i],
                'dev_accuracy': dev_acc_history[i],
                'train_loss': train_loss_history[i],
                'dev_loss': dev_loss_history[i]
            })
    
    print(f"✓ Courbes d'apprentissage sauvegardées dans {learning_curves_filepath}")
    
    # Sauvegarder l'historique détaillé dans un fichier texte
    history_filepath = os.path.join(results_dir, f'history_{model_name}_{timestamp_str}.txt')
    
    with open(history_filepath, 'w', encoding='utf-8') as f:
        f.write(f"Historique d'entraînement - {model_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Meilleure époque: {best_epoch}\n")
        f.write(f"Époques complétées: {epochs_completed}\n")
        f.write(f"Meilleure accuracy dev: {best_dev_accuracy:.4f} ({best_dev_accuracy*100:.2f}%)\n")
        f.write(f"Meilleure loss dev: {best_dev_loss:.4f}\n")
        f.write(f"Temps total d'entraînement: {total_training_time:.2f} sec\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Historique par époque:\n")
        f.write(f"{'Epoch':<8} {'Train Acc':<12} {'Dev Acc':<12} {'Train Loss':<12} {'Dev Loss':<12}\n")
        f.write(f"{'-'*60}\n")
        
        for i in range(len(train_acc_history)):
            f.write(f"{i+1:<8} {train_acc_history[i]:<12.4f} {dev_acc_history[i]:<12.4f} ")
            f.write(f"{train_loss_history[i]:<12.4f} {dev_loss_history[i]:<12.4f}\n")
    
    print(f"✓ Historique détaillé sauvegardé dans {history_filepath}")
