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


class MLP(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.

        UPDATED 2026-01-08: Now fully configurable architecture!

        Configuration via conf["MLP_conf"]:
        - hidden_layers (list): List of hidden layer sizes, e.g., [256, 512, 512, 256]
        - dropout_rate (float): Dropout probability (default: 0.3)
        - activation (str): Activation function - 'relu', 'tanh', 'leaky_relu' (default: 'relu')

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """

        super(MLP, self).__init__()

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]

        # MLP configuration
        mlp_conf = conf.get("MLP_conf", {})
        self.hidden_layers = mlp_conf.get("hidden_layers", [256, 512, 512, 256])
        self.dropout_rate = mlp_conf.get("dropout_rate", 0.3)
        self.activation = mlp_conf.get("activation", "relu")

        # Build layers dynamically
        input_size = self.board_size * self.board_size  # 64
        output_size = self.board_size * self.board_size  # 64

        self.layers = nn.ModuleList()

        # Input layer
        prev_size = input_size

        # Hidden layers
        for hidden_size in self.hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # Output layer
        self.layers.append(nn.Linear(prev_size, output_size))

        # Dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def get_activation(self):
        """Get activation function based on configuration"""
        if self.activation == "relu":
            return F.relu
        elif self.activation == "tanh":
            return torch.tanh
        elif self.activation == "leaky_relu":
            return F.leaky_relu
        else:
            return F.relu  # Default

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

        x = seq
        activation_fn = self.get_activation()

        # Apply all layers except the last one with activation + dropout
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = activation_fn(x)
            x = self.dropout(x)

        # Last layer (no activation before softmax)
        outp = self.layers[-1](x)

        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
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
                loss = loss_fnc(outputs, labels.argmax(dim=-1).to(device))
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
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

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

        # Save learning curves
        import json
        learning_curves = {
            'train_accuracy': [float(acc) for acc in train_acc_list],
            'dev_accuracy': [float(acc) for acc in dev_acc_list],
            'best_epoch': best_epoch,
            'total_epochs': epoch
        }
        with open(self.path_save + '/learning_curves.json', 'w') as f:
            json.dump(learning_curves, f, indent=2)

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")


        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
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
    
    

class LSTMs(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.

        UPDATED 2026-01-08: Now fully configurable architecture!

        Configuration via conf["LSTM_conf"]:
        - hidden_dim (int): Hidden size of LSTM layers (default: 128)
        - num_layers (int): Number of stacked LSTM layers (default: 1, recommend: 2-3)
        - dropout_rate (float): Dropout rate between LSTM layers AND in FC layers (default: 0.1)
        - fc_layers (list): List of FC layer sizes after LSTM, e.g., [256, 128] (default: [])
        - bidirectional (bool): Use bidirectional LSTM (default: False)

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMs, self).__init__()

        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTM/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]

        # LSTM configuration
        lstm_conf = conf["LSTM_conf"]
        self.hidden_dim = lstm_conf.get("hidden_dim", 128)
        self.num_layers = lstm_conf.get("num_layers", 1)
        self.dropout_rate = lstm_conf.get("dropout_rate", 0.1)
        self.fc_layers = lstm_conf.get("fc_layers", [])
        self.bidirectional = lstm_conf.get("bidirectional", False)

        # IMPROVED: Stacked LSTM with dropout between layers
        # Note: dropout parameter is applied BETWEEN LSTM layers (not after the last one)
        self.lstm = nn.LSTM(
            input_size=self.board_size*self.board_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,  # Only apply if >1 layer
            batch_first=True,
            bidirectional=self.bidirectional
        )

        # Calculate LSTM output dimension
        lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

        # IMPROVED: Build FC layers dynamically
        self.fc_modules = nn.ModuleList()
        prev_size = lstm_output_dim

        for fc_size in self.fc_layers:
            self.fc_modules.append(nn.Linear(prev_size, fc_size))
            prev_size = fc_size

        # Final output layer
        self.hidden2output = nn.Linear(prev_size, self.board_size*self.board_size)

        # Dropout for FC layers
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, seq):
        """
        Forward pass of the IMPROVED LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of board states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        # LSTM forward pass (now with multiple layers)
        lstm_out, (hn, cn) = self.lstm(seq)

        # Get last timestep output
        if len(seq.shape)>2:  # Batch mode
            x = lstm_out[:, -1, :]
        else:  # Single sample mode
            x = lstm_out[-1, :]

        # Apply dropout after LSTM
        x = self.dropout(x)

        # IMPROVED: Pass through FC layers with ReLU activation
        for fc_layer in self.fc_modules:
            x = fc_layer(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Final output layer
        outp = self.hidden2output(x)

        # Softmax
        if len(seq.shape)>2:
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            outp = F.softmax(outp).squeeze()

        return outp
    
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

        # Save learning curves
        import json
        learning_curves = {
            'train_accuracy': [float(acc) for acc in train_acc_list],
            'dev_accuracy': [float(acc) for acc in dev_acc_list],
            'best_epoch': best_epoch,
            'total_epochs': epoch
        }
        with open(self.path_save + '/learning_curves.json', 'w') as f:
            json.dump(learning_curves, f, indent=2)

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")


        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
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


class CNN(nn.Module):
    def __init__(self, conf):
        """
        Convolutional Neural Network (CNN) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(CNN, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        # CNN configuration
        cnn_conf = conf["CNN_conf"]
        self.num_filters = cnn_conf["num_filters"]  # e.g., [32, 64, 128]
        self.kernel_size = cnn_conf["kernel_size"]  # e.g., 3 or 5
        self.num_conv_layers = cnn_conf["num_conv_layers"]  # e.g., 2, 3, 4
        self.dropout_rate = cnn_conf.get("dropout_rate", 0.1)
        self.fc_hidden_dim = cnn_conf.get("fc_hidden_dim", 256)

        # Build convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        in_channels = 1

        for i in range(self.num_conv_layers):
            out_channels = self.num_filters[i] if i < len(self.num_filters) else self.num_filters[-1]

            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2  # Maintain spatial dimensions
            )
            self.conv_layers.append(conv)
            in_channels = out_channels

        # Calculate flattened size after convolutions
        # Spatial dimensions remain 8x8 due to padding
        final_filters = self.num_filters[min(self.num_conv_layers-1, len(self.num_filters)-1)]
        self.flatten_size = final_filters * self.board_size * self.board_size

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, self.fc_hidden_dim)
        self.fc2 = nn.Linear(self.fc_hidden_dim, self.board_size * self.board_size)

        # Dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, seq):
        """
        Forward pass of the CNN model.

        Parameters:
        - seq (torch.Tensor): Board state as input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        # Handle input shape
        seq = np.squeeze(seq)

        # Convert to tensor if numpy array
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq).float()

        # Ensure 4D tensor: (batch, channels, height, width)
        if len(seq.shape) == 2:  # Single board (8, 8)
            seq = seq.unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 8)
        elif len(seq.shape) == 3:  # Batch of boards (batch, 8, 8)
            seq = seq.unsqueeze(1)  # (batch, 1, 8, 8)
        # If already (batch, 1, 8, 8) or (batch, channels, h, w), leave as is

        # Apply convolutional layers
        x = seq
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = F.relu(x)
            if i > 0:  # Apply dropout after first conv layer
                x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        outp = self.fc2(x)

        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        """
        Train the CNN model with early stopping.

        Parameters:
        - train: Training data loader
        - dev: Development data loader
        - num_epoch: Maximum number of epochs
        - device: torch device (cuda or cpu)
        - optimizer: Optimizer for training

        Returns:
        - best_epoch: Epoch with best dev accuracy
        """
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange = 0  # to manage early stopping
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()

        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0

            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.argmax(dim=-1).to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()

            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch/nb_batch))
            last_training = time.time() - start_time

            self.eval()

            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)

            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)

            last_prediction = time.time() - last_training - start_time

            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0

                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

            self.train()

            print("*"*15, f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        # Save learning curves
        import json
        learning_curves = {
            'train_accuracy': [float(acc) for acc in train_acc_list],
            'dev_accuracy': [float(acc) for acc in dev_acc_list],
            'best_epoch': best_epoch,
            'total_epochs': epoch
        }
        with open(self.path_save + '/learning_curves.json', 'w') as f:
            json.dump(learning_curves, f, indent=2)

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        return best_epoch

    def evalulate(self, test_loader, device):
        all_predicts = []
        all_targets = []

        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().numpy()

            # Convert targets to class indices
            if target.ndim > 1 and target.shape[1] > 1:  # one-hot
                target = target.argmax(dim=-1).numpy()
            else:  # already indices
                target = target.numpy()

            all_predicts.extend(predicted)
            all_targets.extend(target)

        perf_rep = classification_report(
            all_targets,
            all_predicts,
            zero_division=1,
            digits=4,
            output_dict=True
        )

        return perf_rep


class CNN_LSTM(nn.Module):
    """Hybrid CNN-LSTM: CNN extracts spatial features, LSTM captures temporal dynamics"""

    def __init__(self, conf):
        super(CNN_LSTM, self).__init__()
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        cnn_lstm_conf = conf["CNN_LSTM_conf"]
        self.cnn_filters = cnn_lstm_conf.get("cnn_filters", [32, 64])
        self.cnn_kernel_size = cnn_lstm_conf.get("cnn_kernel_size", 3)
        self.lstm_hidden_dim = cnn_lstm_conf.get("lstm_hidden_dim", 128)
        self.dropout_rate = cnn_lstm_conf.get("dropout_rate", 0.1)

        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for out_channels in self.cnn_filters:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, self.cnn_kernel_size, padding=self.cnn_kernel_size // 2))
            in_channels = out_channels

        self.cnn_output_dim = self.cnn_filters[-1] * self.board_size * self.board_size
        self.lstm = nn.LSTM(input_size=self.cnn_output_dim, hidden_size=self.lstm_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(self.lstm_hidden_dim, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def extract_cnn_features(self, board_state):
        if len(board_state.shape) == 2:
            board_state = board_state.unsqueeze(0).unsqueeze(0)
        elif len(board_state.shape) == 3:
            board_state = board_state.unsqueeze(1)
        x = board_state
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            if i > 0:
                x = self.dropout(x)
        return x.view(x.size(0), -1)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq).float()
        batch_mode = len(seq.shape) == 4
        if not batch_mode:
            seq = seq.unsqueeze(0)

        batch_size, seq_len, h, w = seq.shape
        cnn_features = torch.stack([self.extract_cnn_features(seq[:, t, :, :]) for t in range(seq_len)], dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        output = F.softmax(self.fc_out(self.dropout(lstm_out[:, -1, :])), dim=1)
        return output if batch_mode else output.squeeze(0)

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev, best_epoch, notchange = 0.0, 0, 0
        train_acc_list, dev_acc_list = [], []

        for epoch in range(1, num_epoch + 1):
            self.train()
            loss_batch, nb_batch = 0, 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()

            print(f"epoch : {epoch}/{num_epoch} - loss = {loss_batch/nb_batch:.6f}")
            self.eval()
            acc_train = self.evalulate(train, device)["weighted avg"]["recall"]
            acc_dev = self.evalulate(dev, device)["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            dev_acc_list.append(acc_dev)
            print(f"  Train: {100*acc_train:.2f}% | Dev: {100*acc_dev:.2f}%")

            torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
            if acc_dev > best_dev:
                best_dev, best_epoch, notchange = acc_dev, epoch, 0
            else:
                notchange += 1
            if notchange >= self.earlyStopping:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Best epoch: {best_epoch} with dev acc: {100*best_dev:.2f}%")
        import json
        with open(self.path_save + '/learning_curves.json', 'w') as f:
            json.dump({'train_accuracy': train_acc_list, 'dev_accuracy': dev_acc_list,
                      'best_dev_accuracy': best_dev, 'best_epoch': best_epoch, 'total_epochs': epoch}, f, indent=2)
        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        return best_epoch

    def evalulate(self, test_loader, device):
        all_predicts, all_targets = [], []
        self.eval()
        with torch.no_grad():
            for seq, target, _ in test_loader:
                predicted = self(seq.float().to(device)).argmax(dim=-1).cpu().numpy()
                target = target.argmax(dim=-1).numpy()
                all_predicts.extend(predicted)
                all_targets.extend(target)
        return classification_report(all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)


class CNN_LSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for Othello.
    
    Combines CNN for spatial feature extraction with LSTM for temporal dynamics.
    """
    
    def __init__(self, conf):
        super(CNN_LSTM, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        
        cnn_lstm_conf = conf["CNN_LSTM_conf"]
        self.cnn_filters = cnn_lstm_conf.get("cnn_filters", [32, 64])
        self.cnn_kernel_size = cnn_lstm_conf.get("cnn_kernel_size", 3)
        self.lstm_hidden_dim = cnn_lstm_conf.get("lstm_hidden_dim", 128)
        self.dropout_rate = cnn_lstm_conf.get("dropout_rate", 0.1)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for out_channels in self.cnn_filters:
            self.conv_layers.append(nn.Conv2d(
                in_channels, out_channels, 
                self.cnn_kernel_size, 
                padding=self.cnn_kernel_size // 2
            ))
            in_channels = out_channels
        
        self.cnn_output_dim = self.cnn_filters[-1] * self.board_size * self.board_size
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=self.lstm_hidden_dim,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(self.lstm_hidden_dim, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)
    
    def extract_cnn_features(self, board_state):
        if len(board_state.shape) == 2:
            board_state = board_state.unsqueeze(0).unsqueeze(0)
        elif len(board_state.shape) == 3:
            board_state = board_state.unsqueeze(1)
        
        x = board_state
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = F.relu(x)
            if i > 0:
                x = self.dropout(x)
        
        return x.view(x.size(0), -1)
    
    def forward(self, seq):
        seq = np.squeeze(seq)
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq).float()
        
        if len(seq.shape) == 3:
            batch_mode = False
            seq = seq.unsqueeze(0)
        else:
            batch_mode = True
        
        batch_size, seq_len, h, w = seq.shape
        
        cnn_features = []
        for t in range(seq_len):
            features_t = self.extract_cnn_features(seq[:, t, :, :])
            cnn_features.append(features_t)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, _ = self.lstm(cnn_features)
        last_output = self.dropout(lstm_out[:, -1, :])
        output = F.softmax(self.fc_out(last_output), dim=1)
        
        if not batch_mode:
            output = output.squeeze(0)
        
        return output
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        
        best_dev = 0.0
        best_epoch = 0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        
        for epoch in range(1, num_epoch + 1):
            self.train()
            loss_batch = 0
            nb_batch = 0
            
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            
            print(f"epoch : {epoch}/{num_epoch} - loss = {loss_batch/nb_batch:.6f}")
            
            self.eval()
            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            print(f"  Train Acc: {100*acc_train:.2f}% | Dev Acc: {100*acc_dev:.2f}%")
            
            torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
            
            if acc_dev > best_dev:
                best_dev = acc_dev
                best_epoch = epoch
                notchange = 0
            else:
                notchange += 1
            
            if notchange >= self.earlyStopping:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Best epoch: {best_epoch} with dev acc: {100*best_dev:.2f}%")
        
        import json
        learning_curves = {
            'train_accuracy': train_acc_list,
            'dev_accuracy': dev_acc_list,
            'best_dev_accuracy': best_dev,
            'best_epoch': best_epoch,
            'total_epochs': epoch
        }
        with open(self.path_save + '/learning_curves.json', 'w') as f:
            json.dump(learning_curves, f, indent=2)
        
        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        return best_epoch
    
    def evalulate(self, test_loader, device):
        all_predicts = []
        all_targets = []
        
        self.eval()
        with torch.no_grad():
            for seq, target, _ in test_loader:
                predicted = self(seq.float().to(device))
                predicted = predicted.argmax(dim=-1).cpu().numpy()
                target = target.argmax(dim=-1).numpy()
                for i in range(len(predicted)):
                    all_predicts.append(predicted[i])
                    all_targets.append(target[i])
        
        return classification_report(all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)
