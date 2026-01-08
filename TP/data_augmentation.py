import os
import re

train_generated_filename = 'train_generated.txt'
files = os.listdir('./generated_dataset')

"""On ajoute les fichiers générés au fichier train_generated.txt"""
# regex = r'(\d+)\.h5'

# for file in files:
#     match = re.match(regex, file)
#     if not match:
#         # Ajout des données générées au fichier train_generated.txt
#         with open(train_generated_filename, 'a') as f:
#             f.write(f'{file}\n')

"""On prend les parties et on les tourne de 90, 180 et 270 degrés pour augmenter le dataset."""
TODO

"""On fait une symétrie de toutes les parties pour augmenter le dataset."""        