import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from tqdm import tqdm
import numpy as np
import os
import time


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


class CNN(nn.Module):
    def __init__(self, conf):
        return
    def train_all(self, train, dev, num_epoch, device, optimizer):
        return train_all(self, train, dev, num_epoch, device, optimizer)
    def evalulate(self,test_loader, device):
        return evaluate(self, test_loader, device)

def train_all(self, train, dev, num_epoch, device, optimizer):
    if not os.path.exists(f"{self.path_save}"):
        os.mkdir(f"{self.path_save}")
    best_dev = 0.0
    dev_epoch = 0
    notchange=0
    train_acc_list=[]
    dev_acc_list=[]
    torch.autograd.set_detect_anomaly(True)
    init_time=time.time()
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
        print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                str(loss_batch/nb_batch))
        last_training=time.time()-start_time

        self.eval()
        
        train_clas_rep=self.evalulate(train, device)
        acc_train=train_clas_rep["weighted avg"]["recall"]
        train_acc_list.append(acc_train)
        
        dev_clas_rep=self.evalulate(dev, device)
        acc_dev=dev_clas_rep["weighted avg"]["recall"]
        dev_acc_list.append(acc_dev)
        
        last_prediction=time.time()-last_training-start_time
        
        print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                f"Time:{round(time.time()-init_time)}",
                f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

        if acc_dev > best_dev or best_dev == 0.0:
            notchange=0
            torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
            best_dev = acc_dev
            best_epoch = epoch
        else:
            notchange+=1
            if notchange>self.earlyStopping:
                break
            
        self.train()
        
        print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

    self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
    self.eval()
    _clas_rep = self.evalulate(dev, device)
    print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

    
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