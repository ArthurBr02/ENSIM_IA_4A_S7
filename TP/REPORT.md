## TODO

Faire une map avec les différents paramètres à tester pour les différents modèles (MLP, LSTM, CNN, CNN + LSTM, Transformer, etc.)
Ajouter les optimizers à tester dans la map (au moins deux) :
Adam
Adagrad
SGD

Faire un système pour exporter les métriques (accuracy, loss, etc.) dans des fichiers CSV ou JSON pour chaque modèle testé.

Ajouter les learning rates à tester dans la map (0.0001, 0.001, 0.01, 0.1)

Récupérer les meilleurs paramètres pour chaque modèle (nombre de couches, nombre de neurones par couche, dropout, etc.)
Tester différents batch sizes et nombres d'époques pour chaque modèle.

Calcul de la courbe d'apprentissage (learning curve) pour chaque modèle testé.

Faire des graphiques d'apprentissage (learning curves) pour chaque modèle testé.

Faire tourner deux IA pour générer de nouvelles données et les ajouter au dataset d'entraînement.


## Report of TP
This TP is 70% of your final note. It requires a report of your work that explain all experimental design and the result of your work that you have done in order to find your best model. The report should cover:
Your data partitioning or your approach for collecting more data.
Your proposed and tested architectures and hyperparameters
The result of your model’s performance based on accuracy for all tested architectures and hyperparameters (tables or figures with number).
At the end of TP course, you should present your work in 5 minutes as well

## Evaluation Chart
[] Calculate and Compare the results of provided MLP and LSTM baseline
5

[] Optimizing the architecture of MLP (reporting the number of trainable weights is necessary)
5

[] Optimizing the architecture of LSTM (reporting the number of trainable weights is necessary)
5

[] Testing different optimizer (at least two optimizers)
5

[] Optimizing learning rate (testing 0.0001, 0.001, 0.01, 0.1)
5

[] Checking the impact of different epochs and batch size
5

[] Using all data and not only winner samples
5

[] Calculating the learning curve (training/dev performance on epochs)
5

[] Analysing learning curves
10

[] Not choosing a valid evaluation metric (the reported accuracy, loss should be on dev/test)
-10

[] Experiment design with logical conclusion (how they find the best model?)
5

[] Analysing results to diagnose the problem or improve results
10

[] Analysing different evaluation metrics such as “game win ratio” or “legal move ratio”
5

[] Using more data in training after finding the best model (adding test or dev in final training)
5

[] Implementation of a CNN network and optimizing its architecture
10

[] Implementation of a new architecture like CNN-LSTM or attention based (Transformer)
10

[] Implementation of pretrained model/architecture, for example from huggingface
10

[] Generate new data and adding to training set (for example, using the log of two AI players)
20

[] Presentation (respecting time, choosing the important parts to present, reporting as table or figure)
10