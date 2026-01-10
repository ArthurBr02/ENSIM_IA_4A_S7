#!/usr/bin/env python3
"""Convertit un fichier .h5 (jeu) en .npz ou .json pour inspection.

Le script détecte la première clé du fichier et exporte:
 - .npz : arrays `boards` et `moves` (dtype preserved)
 - .json : structure convertie en listes (potentiellement volumineux)

Usage:
  python h5_to_game.py dataset/16740.h5 --out-format npz
"""
import argparse
import h5py
import numpy as np
import os
import json
import sys


def read_first_dataset(path):
    with h5py.File(path, 'r') as h5f:
        keys = list(h5f.keys())
        if not keys:
            raise SystemExit('No datasets found in h5 file')
        key = keys[0]
        data = h5f[key][()]
        return key, data


def export_npz(outpath, key, data):
    # Expected shape (2, 60, 8, 8): 0 -> boards, 1 -> moves
    arr = np.asarray(data)
    if arr.ndim == 4 and arr.shape[0] == 2:
        boards = arr[0].astype(np.int8)
        moves = arr[1].astype(np.int8)
    else:
        # try to heuristically split last axis
        raise SystemExit('Unexpected array shape: ' + str(arr.shape))

    np.savez_compressed(outpath, boards=boards, moves=moves)
    print(f'Wrote NPZ to {outpath}')


def export_json(outpath, key, data):
    arr = np.asarray(data)
    if arr.ndim == 4 and arr.shape[0] == 2:
        boards = arr[0].tolist()
        moves = arr[1].tolist()
    else:
        raise SystemExit('Unexpected array shape for json export: ' + str(arr.shape))

    out = {'key': key, 'boards': boards, 'moves': moves}
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print(f'Wrote JSON to {outpath}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5file')
    parser.add_argument('--out-format', choices=['npz', 'json'], default='npz')
    parser.add_argument('--out', default=None, help='Output path (optional)')
    args = parser.parse_args()

    key, data = read_first_dataset(args.h5file)
    base = args.out if args.out else os.path.splitext(os.path.basename(args.h5file))[0]

    if args.out_format == 'npz':
        outpath = base + '.npz' if not args.out else args.out
        export_npz(outpath, key, data)
    else:
        outpath = base + '.json' if not args.out else args.out
        export_json(outpath, key, data)


if __name__ == '__main__':
    main()
