#!/usr/bin/env python3
"""Générateur de parties H5 compatible avec le format dataset (shape (2,60,8,8), dtype=int8).

Ce wrapper utilise `play_game` depuis `generate_h5_files_from_games.py` si disponible,
et sauvegarde chaque partie au format attendu.

Usage:
  python generate_h5_cli.py model1.pt model2.pt 10 output_dir --start-id 2000000
"""
import argparse
import os
import sys
import numpy as np
import torch

# ensure parent dir is importable
here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(here, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from generate_h5_files_from_games import play_game
except Exception:
    play_game = None

import h5py


def save_game_h5_compatible(boards, moves, file_path, key_name):
    # boards: list/array shape (61,8,8) ; moves: list/array shape (60,8,8)
    b = np.array(boards, dtype=np.int8)
    m = np.array(moves, dtype=np.int8)

    # Expecting shape (2,60,8,8) — drop final board to get 60 if needed
    if b.shape[0] == 61:
        b60 = b[:60]
    else:
        b60 = b

    stacked = np.stack([b60, m], axis=0).astype(np.int8)

    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset(key_name, data=stacked, dtype=np.int8)


def load_model(path, device):
    if torch.cuda.is_available():
        return torch.load(path, weights_only=False)
    else:
        return torch.load(path, map_location=torch.device('cpu'), weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model1')
    parser.add_argument('model2')
    parser.add_argument('num_games', type=int)
    parser.add_argument('out_dir')
    parser.add_argument('--start-id', type=int, default=1000000)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    print('Loading models...')
    model1 = load_model(args.model1, device)
    model2 = load_model(args.model2, device)

    if play_game is None:
        raise SystemExit('play_game not importable from generate_h5_files_from_games.py')

    for i in range(args.num_games):
        # alternate order
        if i % 2 == 0:
            boards, moves = play_game(model1, model2, device, verbose=False)
            is_model1_black = True
        else:
            boards, moves = play_game(model2, model1, device, verbose=False)
            is_model1_black = False

        game_id = args.start_id + i
        fname = os.path.join(args.out_dir, f"{game_id}.h5")
        save_game_h5_compatible(boards, moves, fname, str(game_id))
        print(f'Saved {fname}')

    print('Done')


if __name__ == '__main__':
    main()
