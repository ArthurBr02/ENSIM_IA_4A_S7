J'ai modifié le nombre d'epoch de 200 à 20 pour réduire le temps d'entraînement.

```python
# Modèle de base MLP
conf["board_size"]=8
conf["path_save"]="save_models"
conf['epoch']=20
conf["earlyStopping"]=20
conf["len_inpout_seq"]=len_samples
conf["LSTM_conf"]={}
conf["LSTM_conf"]["hidden_dim"]=128

best DEV: WAcc : 14.97%
Fin entrainement MLP sur 20 epoch en (109.52468705177307, 'sc')
```

```python
# Modèle de base LSTM
conf["board_size"]=8
conf["path_save"]="save_models"
conf['epoch']=20
conf["earlyStopping"]=20
conf["len_inpout_seq"]=len_samples
conf["LSTM_conf"]={}
conf["LSTM_conf"]["hidden_dim"]=256

best DEV: WAcc : 22.073333333333334%
Fin entrainement LSTM sur 20 epoch en (403.4299657344818, 'sc')
```