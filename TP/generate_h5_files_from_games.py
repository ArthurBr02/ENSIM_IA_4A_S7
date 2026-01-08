import numpy as np
import torch
import h5py
import copy
import time
from utile import get_legal_moves, initialze_board, has_tile_to_flip
from game import input_seq_generator, find_best_move, apply_flip

def save_game_to_h5(board_stats_seq, moves_log, game_id, output_dir="dataset/"):
    """
    Sauvegarde une bataille en fichier .h5
    
    Parameters:
    - board_stats_seq: liste des états du plateau
    - moves_log: chaîne des coups joués (ex: "4455...")
    - game_id: identifiant unique du jeu
    - output_dir: dossier de destination
    """
    # Convertir moves_log en grilles 8x8 avec un 1 à la position du coup
    max_moves = 60  # Longueur fixe comme dans les fichiers existants
    moves_grids = np.zeros((max_moves, 8, 8), dtype=np.int8)
    
    move_idx = 0
    for i in range(0, len(moves_log), 2):
        if move_idx >= max_moves:
            break
            
        if moves_log[i:i+2] != "__":
            row = int(moves_log[i]) - 1
            col = int(moves_log[i+1]) - 1
            moves_grids[move_idx, row, col] = 1
        # Pour les "pass", on laisse la grille à zéros
        move_idx += 1
    
    # Padder board_stats_seq pour avoir exactement max_moves états
    boards_array = np.zeros((max_moves, 8, 8), dtype=np.int8)
    num_boards = min(len(board_stats_seq), max_moves)
    for i in range(num_boards):
        boards_array[i] = board_stats_seq[i]
    
    # Créer le tableau final avec shape (2, 60, 8, 8)
    game_data = np.array([boards_array, moves_grids], dtype=np.int8)
    
    # Sauvegarder dans un fichier .h5
    file_path = f"{output_dir}{game_id}.h5"
    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset(str(game_id), data=game_data)
    
    print(f"Bataille sauvegardée : {file_path}")
    return file_path

def generate_battle_dataset(player1_path, player2_path, num_games=10, output_dir="dataset/"):
    """
    Génère plusieurs batailles entre deux modèles et les sauvegarde en .h5
    
    Parameters:
    - player1_path: chemin du modèle 1
    - player2_path: chemin du modèle 2
    - num_games: nombre de parties à jouer
    - output_dir: dossier de sortie
    """
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    for g in range(num_games):
        # Alterner qui commence
        if g % 2 == 0:
            conf = {'player1': player1_path, 'player2': player2_path}
        else:
            conf = {'player1': player2_path, 'player2': player1_path}
        
        board_stat = initialze_board()
        moves_log = ""
        board_stats_seq = []
        pass2player = False
        
        # Jouer la partie
        while not np.all(board_stat) and not pass2player:
            # Tour du joueur noir
            NgBlackPsWhith = -1
            board_stats_seq.append(copy.copy(board_stat))
            
            model = torch.load(conf['player1'], map_location=device, weights_only=False)
            model.eval()
            
            input_seq_boards = input_seq_generator(board_stats_seq, model.len_inpout_seq)
            model_input = np.array([input_seq_boards]) * -1
            move1_prob = model(torch.tensor(model_input).float().to(device))
            move1_prob = move1_prob.cpu().detach().numpy().reshape(8, 8)
            
            legal_moves = get_legal_moves(board_stat, NgBlackPsWhith)
            
            if len(legal_moves) > 0:
                best_move = find_best_move(move1_prob, legal_moves)
                board_stat[best_move[0], best_move[1]] = NgBlackPsWhith
                moves_log += str(best_move[0] + 1) + str(best_move[1] + 1)
                board_stat = apply_flip(best_move, board_stat, NgBlackPsWhith)
            else:
                if moves_log[-2:] == "__":
                    pass2player = True
                moves_log += "__"
            
            # Tour du joueur blanc
            NgBlackPsWhith = +1
            board_stats_seq.append(copy.copy(board_stat))
            
            model = torch.load(conf['player2'], map_location=device, weights_only=False)
            model.eval()
            
            input_seq_boards = input_seq_generator(board_stats_seq, model.len_inpout_seq)
            model_input = np.array([input_seq_boards])
            move1_prob = model(torch.tensor(model_input).float().to(device))
            move1_prob = move1_prob.cpu().detach().numpy().reshape(8, 8)
            
            legal_moves = get_legal_moves(board_stat, NgBlackPsWhith)
            
            if len(legal_moves) > 0:
                best_move = find_best_move(move1_prob, legal_moves)
                board_stat[best_move[0], best_move[1]] = NgBlackPsWhith
                moves_log += str(best_move[0] + 1) + str(best_move[1] + 1)
                board_stat = apply_flip(best_move, board_stat, NgBlackPsWhith)
            else:
                if moves_log[-2:] == "__":
                    pass2player = True
                moves_log += "__"
        
        board_stats_seq.append(copy.copy(board_stat))
        
        # Générer un ID unique pour le jeu
        timestamp = int(time.time() * 1000)
        game_id = f"{timestamp}_{g}"
        
        # Sauvegarder en .h5
        save_game_to_h5(board_stats_seq, moves_log, game_id, output_dir)
        
        print(f"Partie {g+1}/{num_games} terminée")

if __name__ == "__main__":
    player1_model_path = "save_models_MLP_256_128/model_181.pt"
    player2_model_path = "save_models_MLP_256_128/model_181.pt"
    generate_battle_dataset(player1_model_path, player2_model_path, num_games=10000, output_dir="generated_dataset/")