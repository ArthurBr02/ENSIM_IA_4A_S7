"""
Script pour générer des fichiers .h5 contenant des parties d'Othello
en faisant jouer 2 modèles entre eux.

Usage:
    python generate_h5_files_from_games.py <model1_path> <model2_path> <num_games> <output_dir>
    
Exemple:
    python generate_h5_files_from_games.py save_models_MLP/best_model.pth save_models_CNN_32_Dropout_Gridsearch_Relu/best_model.pth 10 generated_dataset/
"""

import numpy as np
import torch
import h5py
import sys
import copy
import os
from tqdm import tqdm
from utile import get_legal_moves, initialze_board


def input_seq_generator(board_stats_seq, length_seq):
    """
    Génère une séquence d'états de plateau pour l'entrée du modèle.
    
    Parameters:
    - board_stats_seq (list): Séquence d'états de plateau.
    - length_seq (int): Longueur de la séquence à générer.
    
    Returns:
    - list: Séquence d'états de plateau de longueur length_seq.
    """
    board_stat_init = initialze_board()

    if len(board_stats_seq) >= length_seq:
        input_seq = board_stats_seq[-length_seq:]
    else:
        input_seq = [board_stat_init]
        # Padding avec l'état initial du plateau
        for i in range(length_seq - len(board_stats_seq) - 1):
            input_seq.append(board_stat_init)
        # Ajouter les états de la partie
        for i in range(len(board_stats_seq)):
            input_seq.append(board_stats_seq[i])
            
    return input_seq


def find_best_move(move_prob, legal_moves, use_sampling=True, temperature=1.0):
    """
    Choisit un coup parmi `legal_moves` à partir des scores `move_prob`.

    Si `use_sampling` est False -> argmax déterministe.
    Sinon échantillonne proportionnellement aux scores (applique `temperature`).
    """
    if not legal_moves:
        return None

    if not use_sampling:
        # deterministic argmax
        best_move = legal_moves[0]
        max_score = move_prob[best_move[0], best_move[1]]
        for move in legal_moves:
            score = move_prob[move[0], move[1]]
            if score > max_score:
                max_score = score
                best_move = move
        return best_move

    # sampling mode
    probs = np.array([move_prob[m[0], m[1]] for m in legal_moves], dtype=float)
    # shift to non-negative
    mn = probs.min()
    if mn < 0:
        probs = probs - mn
    if probs.sum() == 0:
        probs = np.ones_like(probs)

    if temperature is not None and temperature > 0 and temperature != 1.0:
        probs = probs ** (1.0 / float(temperature))

    probs = probs / probs.sum()
    idx = np.random.choice(len(legal_moves), p=probs)
    return legal_moves[idx]


def has_tile_to_flip(best_move, direction, board_stat, NgBlackPsWhith):
    """
    Vérifie s'il y a des pions à retourner dans une direction donnée.
    """
    from utile import is_valid_coord
    
    i = 1
    if is_valid_coord(best_move[0], best_move[1]):
        while True:
            row = best_move[0] + direction[0] * i
            col = best_move[1] + direction[1] * i
            if not is_valid_coord(row, col) or board_stat[row][col] == 0:
                return False
            elif board_stat[row][col] == NgBlackPsWhith:
                break
            else:
                i += 1
    return i > 1


def apply_flip(best_move, board_stat, NgBlackPsWhith):
    """
    Applique le retournement des pions sur le plateau.
    
    Parameters:
    - best_move (tuple): Coordonnées (row, col) du coup joué.
    - board_stat (numpy.ndarray): État actuel du plateau.
    - NgBlackPsWhith (int): Indicateur du joueur (-1 pour Noir, 1 pour Blanc).
    
    Returns:
    - numpy.ndarray: Plateau mis à jour après retournement des pions.
    """
    MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
                 (0, -1),           (0, +1),
                 (+1, -1), (+1, 0), (+1, +1)]

    for direction in MOVE_DIRS:
        if has_tile_to_flip(best_move, direction, board_stat, NgBlackPsWhith):
            i = 1
            while True:
                row = best_move[0] + direction[0] * i
                col = best_move[1] + direction[1] * i
                if board_stat[row][col] == board_stat[best_move[0], best_move[1]]:
                    break
                else:
                    board_stat[row][col] = board_stat[best_move[0], best_move[1]]
                    i += 1
                    
    return board_stat


def play_game(model1, model2, device, verbose=False, use_sampling=True, temperature=1.0):
    """
    Fait jouer une partie entre deux modèles.
    
    Parameters:
    - model1 (torch.nn.Module): Modèle pour le joueur Noir.
    - model2 (torch.nn.Module): Modèle pour le joueur Blanc.
    - device (torch.device): Device pour l'exécution (CPU ou CUDA).
    - verbose (bool): Afficher les détails de la partie.
    
    Returns:
    - tuple: (board_stats_seq, moves_prob_seq) où:
        - board_stats_seq: Liste de tous les états du plateau (60 max).
        - moves_prob_seq: Liste de toutes les probabilités de coups (60 max).
    """
    board_stat = initialze_board()
    board_stats_seq = []
    moves_prob_seq = []
    pass_count = 0
    
    model1.eval()
    model2.eval()
    
    # Récupérer len_inpout_seq ou utiliser 1 par défaut
    len_seq_model1 = getattr(model1, 'len_inpout_seq', 1)
    len_seq_model2 = getattr(model2, 'len_inpout_seq', 1)
    
    while len(board_stats_seq) < 60 and pass_count < 2:
        # Tour du joueur Noir (model1)
        NgBlackPsWhith = -1
        board_stats_seq.append(copy.copy(board_stat))
        
        input_seq_boards = input_seq_generator(board_stats_seq, len_seq_model1)
        # Si Noir est le joueur actuel, multiplier le plateau par -1
        model_input = np.array([input_seq_boards]) * -1
        move_prob = model1(torch.tensor(model_input).float().to(device))
        move_prob = move_prob.cpu().detach().numpy().reshape(8, 8)
        
        legal_moves = get_legal_moves(board_stat, NgBlackPsWhith)
        
        if len(legal_moves) > 0:
            best_move = find_best_move(move_prob, legal_moves, use_sampling=use_sampling, temperature=temperature)
            if verbose:
                print(f"Black: {best_move} from {len(legal_moves)} legal moves")
            
            board_stat[best_move[0], best_move[1]] = NgBlackPsWhith
            board_stat = apply_flip(best_move, board_stat, NgBlackPsWhith)
            
            # Créer une matrice de probabilité avec seulement le coup joué à 1
            move_matrix = np.zeros((8, 8))
            move_matrix[best_move[0], best_move[1]] = 1
            moves_prob_seq.append(move_matrix)
            
            pass_count = 0  # Réinitialiser le compteur de pass
        else:
            if verbose:
                print("Black passes")
            # Ajouter une matrice vide pour le pass
            moves_prob_seq.append(np.zeros((8, 8)))
            pass_count += 1
        
        # Vérifier si la partie est terminée
        if pass_count >= 2 or len(board_stats_seq) >= 60:
            break
        
        # Tour du joueur Blanc (model2)
        NgBlackPsWhith = +1
        board_stats_seq.append(copy.copy(board_stat))
        
        input_seq_boards = input_seq_generator(board_stats_seq, len_seq_model2)
        model_input = np.array([input_seq_boards])
        move_prob = model2(torch.tensor(model_input).float().to(device))
        move_prob = move_prob.cpu().detach().numpy().reshape(8, 8)
        
        legal_moves = get_legal_moves(board_stat, NgBlackPsWhith)
        
        if len(legal_moves) > 0:
            best_move = find_best_move(move_prob, legal_moves, use_sampling=use_sampling, temperature=temperature)
            if verbose:
                print(f"White: {best_move} from {len(legal_moves)} legal moves")
            
            board_stat[best_move[0], best_move[1]] = NgBlackPsWhith
            board_stat = apply_flip(best_move, board_stat, NgBlackPsWhith)
            
            # Créer une matrice de probabilité avec seulement le coup joué à 1
            move_matrix = np.zeros((8, 8))
            move_matrix[best_move[0], best_move[1]] = 1
            moves_prob_seq.append(move_matrix)
            
            pass_count = 0  # Réinitialiser le compteur de pass
        else:
            if verbose:
                print("White passes")
            # Ajouter une matrice vide pour le pass
            moves_prob_seq.append(np.zeros((8, 8)))
            pass_count += 1
    
    # Ajouter l'état final du plateau
    board_stats_seq.append(copy.copy(board_stat))
    
    # Padding pour avoir exactement 60 états
    while len(board_stats_seq) < 61:  # 61 car on a l'état initial + 60 coups max
        board_stats_seq.append(copy.copy(board_stat))
    
    while len(moves_prob_seq) < 60:
        moves_prob_seq.append(np.zeros((8, 8)))
    
    if verbose:
        final_score = np.sum(board_stat)
        if final_score < 0:
            print(f"Black wins with {-int(final_score)} points")
        elif final_score > 0:
            print(f"White wins with {int(final_score)} points")
        else:
            print("Draw")
    
    return board_stats_seq[:61], moves_prob_seq[:60]


def save_game_to_h5(board_stats_seq, moves_prob_seq, file_path, game_name):
    """
    Sauvegarde une partie dans un fichier .h5.
    
    Parameters:
    - board_stats_seq (list): Séquence des états du plateau (61 états: initial + 60 coups).
    - moves_prob_seq (list): Séquence des coups joués (60 coups).
    - file_path (str): Chemin du fichier .h5 à créer.
    - game_name (str): Nom de la partie (utilisé comme clé dans le fichier .h5).
    """
    # Convertir en numpy arrays
    boards = np.array(board_stats_seq, dtype=np.float32)
    moves = np.array(moves_prob_seq, dtype=np.float32)
    
    # Créer le dataset avec la même structure que les fichiers existants
    game_data = np.array([boards, moves], dtype=object)
    
    # Sauvegarder dans un fichier .h5
    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset(game_name, data=game_data, dtype=h5py.special_dtype(vlen=np.float32))
    
    print(f"Game saved to {file_path}")


def generate_games(model1_path, model2_path, num_games, output_dir, start_id=1000000, use_sampling=True, temperature=1.0):
    """
    Génère plusieurs parties entre deux modèles et les sauvegarde en .h5.
    
    Parameters:
    - model1_path (str): Chemin vers le premier modèle (joueur Noir).
    - model2_path (str): Chemin vers le deuxième modèle (joueur Blanc).
    - num_games (int): Nombre de parties à générer.
    - output_dir (str): Dossier de sortie pour les fichiers .h5.
    - start_id (int): ID de départ pour nommer les fichiers.
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Détecter le device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Charger les modèles
    print(f"Loading model 1: {model1_path}")
    if torch.cuda.is_available():
        model1 = torch.load(model1_path, weights_only=False)
    else:
        model1 = torch.load(model1_path, map_location=torch.device('cpu'), weights_only=False)
    
    print(f"Loading model 2: {model2_path}")
    if torch.cuda.is_available():
        model2 = torch.load(model2_path, weights_only=False)
    else:
        model2 = torch.load(model2_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Générer les parties
    stats = {"model1_wins": 0, "model2_wins": 0, "draws": 0}
    
    print(f"\nModel 1 len_inpout_seq: {getattr(model1, 'len_inpout_seq', 'NOT FOUND')}")
    print(f"Model 2 len_inpout_seq: {getattr(model2, 'len_inpout_seq', 'NOT FOUND')}")
    
    for i in tqdm(range(num_games), desc="Generating games"):
        # Alterner les joueurs pour équilibrer
        if i % 2 == 0:
            board_stats_seq, moves_prob_seq = play_game(model1, model2, device, verbose=(i==0), use_sampling=use_sampling, temperature=temperature)
            is_model1_black = True
        else:
            board_stats_seq, moves_prob_seq = play_game(model2, model1, device, verbose=(i==1), use_sampling=use_sampling, temperature=temperature)
            is_model1_black = False
        
        # Calculer le résultat
        final_score = np.sum(board_stats_seq[-1])
        if final_score < 0:  # Noir gagne
            if is_model1_black:
                stats["model1_wins"] += 1
            else:
                stats["model2_wins"] += 1
        elif final_score > 0:  # Blanc gagne
            if is_model1_black:
                stats["model2_wins"] += 1
            else:
                stats["model1_wins"] += 1
        else:
            stats["draws"] += 1
        
        # Sauvegarder la partie
        game_id = start_id + i
        game_name = str(game_id)
        file_path = os.path.join(output_dir, f"{game_name}.h5")
        save_game_to_h5(board_stats_seq, moves_prob_seq, file_path, game_name)
    
    print("\n=== Statistics ===")
    print(f"Model 1 wins: {stats['model1_wins']}")
    print(f"Model 2 wins: {stats['model2_wins']}")
    print(f"Draws: {stats['draws']}")
    print(f"\nAll {num_games} games saved to {output_dir}")
    
    # Créer un fichier texte listant tous les fichiers générés
    list_file = os.path.join(output_dir, "generated_games.txt")
    with open(list_file, 'w') as f:
        for i in range(num_games):
            game_id = start_id + i
            f.write(f"{game_id}.h5\n")
    print(f"Game list saved to {list_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate H5 game files from two models')
    parser.add_argument('model1_path')
    parser.add_argument('model2_path')
    parser.add_argument('num_games', type=int)
    parser.add_argument('--output-dir', default='generated_dataset/')
    parser.add_argument('--start-id', type=int, default=1000000)
    parser.add_argument('--no-sampling', dest='use_sampling', action='store_false', help='Disable stochastic sampling (use argmax)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (>0). 1.0 = default')

    args = parser.parse_args()

    generate_games(args.model1_path, args.model2_path, args.num_games, args.output_dir, args.start_id, use_sampling=args.use_sampling, temperature=args.temperature)
