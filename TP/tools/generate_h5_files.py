import json
from time import time
import numpy as np
import copy
import random

from game import input_seq_generator, find_best_move, apply_flip

import torch
from utile import get_legal_moves,initialze_board

NB_PARTIES = 200

model_1 ='../best_models/MLP_optimise_model_48_1767919116.7898524_0.46385prct.pt'
model_2 = '../best_models/CNN_optimise_model_20_1767995697.4717474_0.5165333333333333prct.pt'

def get_empty_board():
    return np.zeros((8,8))

"""
Enregistrer les parties jouées dans des fichiers h5
"""
def create_h5_file(file_path, boards, moves, key_name='data'):
    import h5py
    with h5py.File(file_path, 'w') as h5f:
        stacked = np.stack([boards, moves], axis=0).astype(np.int8)
        h5f.create_dataset(key_name, data=stacked, dtype=np.int8)

def add_file_to_generated_dataset(filename):
    with open('../train_generated.txt', 'a') as f:
        f.write(filename + '\n')

def export_game_to_json(boards, moves, key, outpath):
    # Convert numpy arrays to plain Python lists for JSON serialization
    try:
        boards_serializable = np.asarray(boards).tolist()
    except Exception:
        boards_serializable = boards
    try:
        moves_serializable = np.asarray(moves).tolist()
    except Exception:
        moves_serializable = moves

    out = {'boards': boards_serializable, 'moves': moves_serializable, 'key': key}
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    print(f'Wrote JSON to {outpath}')

if __name__ == "__main__":
    start_time = time()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    for i in range(NB_PARTIES):
        # On va faire une liste des coups joués et des boards
        moves = []
        boards = []


        win = {}
        conf={}

        # Alternate which model starts first
        if i%2 == 0:
            conf['player1']= model_1
            conf['player2']= model_2
        else:
            conf['player2']= model_1
            conf['player1']= model_2

        # Initialize the board
        board_stat=initialze_board()

        moves_log = ""

        
        board_stats_seq=[]
        pass2player=False

        while not np.all(board_stat) and not pass2player:
            NgBlackPsWhith=-1
            board_stats_seq.append(copy.copy(board_stat))

            # Load model for player 1
            if torch.cuda.is_available():
                model = torch.load(conf['player1'],weights_only=False)
            else:
                model = torch.load(conf['player1'],map_location=torch.device('cpu'),weights_only=False)
            model.eval()

            input_seq_boards=input_seq_generator(board_stats_seq,model.len_inpout_seq)
            #if black is the current player the board should be multiplay by -1
            model_input=np.array([input_seq_boards])*-1
            move1_prob = model(torch.tensor(model_input).float().to(device))
            move1_prob=move1_prob.cpu().detach().numpy().reshape(8,8)

            legal_moves=get_legal_moves(board_stat,NgBlackPsWhith)

            if len(legal_moves)>0:
                
                best_move=find_best_move(move1_prob,legal_moves)

                # Je génère le board du coup joué
                empty_board = get_empty_board()
                empty_board[best_move[0], best_move[1]] = 1
                moves.append(empty_board)

                board_stat[best_move[0],best_move[1]]=NgBlackPsWhith
                moves_log+=str(best_move[0]+1)+str(best_move[1]+1)
                
                board_stat=apply_flip(best_move,board_stat,NgBlackPsWhith)

                boards.append(copy.copy(board_stat))

            else:
                # print("Black pass")
                if moves_log[-2:]=="__":
                    pass2player=True
                moves_log+="__"

            NgBlackPsWhith=+1
            board_stats_seq.append(copy.copy(board_stat))

            if torch.cuda.is_available():
                model = torch.load(conf['player2'],weights_only=False)
            else:
                model = torch.load(conf['player2'],map_location=torch.device('cpu'),weights_only=False)
            model.eval()

            input_seq_boards=input_seq_generator(board_stats_seq,model.len_inpout_seq)
            #if white is the current player the board should be multiplay by -1
            model_input=np.array([input_seq_boards])*-1
            move1_prob = model(torch.tensor(model_input).float().to(device))
            move1_prob=move1_prob.cpu().detach().numpy().reshape(8,8)

            legal_moves=get_legal_moves(board_stat,NgBlackPsWhith)


            if len(legal_moves)>0:
                
                best_move = find_best_move(move1_prob,legal_moves)
                # print(f"White: {best_move} < from possible move {legal_moves}")

                # Je génère le board du coup joué
                empty_board = get_empty_board()
                empty_board[best_move[0], best_move[1]] = 1
                moves.append(empty_board)
                
                board_stat[best_move[0],best_move[1]]=NgBlackPsWhith
                moves_log+=str(best_move[0]+1)+str(best_move[1]+1)
                
                board_stat=apply_flip(best_move,board_stat,NgBlackPsWhith)

                boards.append(copy.copy(board_stat))

            else:
                # print("White pass")
                if moves_log[-2:]=="__":
                    pass2player=True
                moves_log+="__"
                
        board_stats_seq.append(copy.copy(board_stat))
        

        # if np.sum(board_stat)<0:
        #     model_name = conf['player1'].replace('\\', '')
        #     if not win.get(model_name):
        #         win[model_name] = 0
        #     win[model_name] += 1

        #     # print(f"Black {conf['player1']} is winner (with {-1*int(np.sum(board_stat))} points)")
        # elif np.sum(board_stat)>0:
        #     model_name = conf['player2'].replace('\\', '')
        #     if not win.get(model_name):
        #         win[model_name] = 0
        #     win[model_name] += 1

        #     # print(f"White {conf['player2']} is winner (with {int(np.sum(board_stat))} points)")
        # else:
        #     print(f"Draw")
        # print(moves)
        # print(boards)

        # Conversion des float en int 
        for m in range(len(moves)):
            moves[m] = moves[m].astype(np.int8)
        for b in range(len(boards)):
            boards[b] = boards[b].astype(np.int8)

        key = random.randint(100000,9999999)
        _time = int(time())
        name = f'game_{_time}_{key}'

        # Si il y a moins de 60 coups, on enregistre pas la partie
        # if len(moves) < 60:
        #     i -= 1
        #     continue

        create_h5_file(f'../generated_dataset/h5/{name}.h5', boards, moves, name)
        export_game_to_json(boards, moves, name, f'../generated_dataset/json/{name}.json')
        add_file_to_generated_dataset(f'{name}.h5')
        print(len(moves))
        print(len(boards))
    print("Génération de", NB_PARTIES, "parties en", time() - start_time, "sc")