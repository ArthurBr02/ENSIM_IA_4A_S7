J'ai modifié le nombre d'epoch de 200 à 20 pour réduire le temps d'entraînement.

# Calculate and Compare the results of provided MLP and LSTM baseline
Pour les modèles MLP et LSTM de base, j'ai obtenu les résultats suivants après 20 epochs d'entraînement.

```python
# Modèle de base MLP
conf["board_size"]=8
conf["path_save"]="save_models"
conf['epoch']=20
conf["earlyStopping"]=20
conf["len_inpout_seq"]=1
conf["dropout"]=0.1

Optimizer: Adam, learning rate: 0.001

Nombre de paramètres entraînables MLP: 33088

best DEV: Accuracy : 14.97%
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
conf["dropout"]=0.1

Optimizer: Adam, learning rate: 0.001

Nombre de paramètres entraînables LSTM: 346176

best DEV: Accuracy : 22.073333333333334%
Fin entrainement LSTM sur 20 epoch en (403.4299657344818, 'sc')
```
On remarque que le modèle LSTM performe mieux que le modèle MLP de base mais demande un temps d'entraînement 4x plus long (on a 67% de performance en plus pour un temps d'entraînement 4x plus long). On aussi un nombre de paramètres entraînables beaucoup plus élevé pour le LSTM (346176 contre 33088 pour le MLP), ce qui explique en partie le temps d'entraînement plus long.

## Exécution de game.py
J'ai testé les deux modèles de base en les faisant jouer des parties l'un contre l'autre, LSTM gagne à chaque fois.

# Optimizing the architecture of MLP (reporting the number of trainable weights is necessary)

## Choix du nombre de couches

Je vais faire des MLP avec les couches suivantes pour voir la différence. Cependant, plus il y a de couches plus le temps de calcul augmente.
```python
[64],              # 1 couche
[128],              # 1 couche
[256, 128],         # 2 couches
[512, 256, 128],    # 3 couches
[512, 256, 128, 64] # 4 couches
```

Les paramètres du modèle sont les suivants pour chaque architecture testée:

```python
conf["board_size"]=8
conf["path_save"]="save_models"
conf['epoch']=20
conf["earlyStopping"]=20
conf["len_inpout_seq"]=1
conf["dropout"]=0.1

Optimizer: Adam, learning rate: 0.001
```

Couches (entrainement sur 20 epochs):
- [64] : 12480 paramètres, temps d'entraînement 101sc, Accuracy 12.38%
- [128] : 33088 paramètres, temps d'entraînement 109sc, Accuracy 14.97%
- **[256, 128]** : 57792 paramètres, temps d'entraînement 151sc, Accuracy 15.53%
- [512, 256, 128] : 205760 paramètres, temps d'entraînement 333sc, Accuracy 13.78%
- [512, 256, 128, 64] : 209920 paramètres, temps d'entraînement 429sc, Accuracy 10.43%


D'après les données ci-dessus, on remarque que l'augmentation du nombre de couches améliore les performances du modèle MLP jusqu'à une certaine limite. En effet, le modèle avec deux couches ([256, 128]) offre la meilleure précision (15.53%) par rapport aux autres architectures testées. Cependant, au-delà de deux couches, les performances diminuent, ce qui peut être attribué à un surapprentissage ou à une complexité excessive du modèle. De plus, le temps d'entraînement augmente significativement avec le nombre de couches, ce qui peut ne pas être justifié par les gains de performance. Ainsi, pour ce jeu de données et cette tâche spécifique, une architecture MLP avec deux couches semble être le compromis optimal entre complexité et performance. J'ai aussi essayé de réduire le nombre de neurones par couche, mais les performances ont diminué et le temps de calcul n'a pas beaucoup changé.

## Changement des paramètres d'entraînement

Maintenat que j'ai choisi l'architecture [256, 128] pour le MLP, je vais modifier l'optimizer et le learning rate pour voir si je peux améliorer les performances.

### Optimizer TODO TODO TODO TODO
J'ai testé les optimizers suivants avec un learning rate ayant les valeurs suivantes [0.0001, 0.001, 0.01, 0.1]:
- Adam
- Adagrad
- SGD

Résultats (entrainement sur 20 epochs):
- Adam : Meilleur résultat avec lr=, Accuracy %, Temps d'entraînement sc, Nombre de paramètres
- Adagrad : Meilleur résultat avec lr=, Accuracy %, Temps d'entraînement sc, Nombre de paramètres
- SGD : Meilleur résultat avec lr=, Accuracy %, Temps d'entraînement sc, Nombre de paramètres
