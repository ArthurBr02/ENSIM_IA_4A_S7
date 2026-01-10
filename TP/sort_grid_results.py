#!/usr/bin/env python3
"""Trie un fichier CSV de résultats de gridsearch.

Usage:
  python sort_grid_results.py results/gridsearch_cnn_results_20260108_214135.csv

Le script produit un fichier nommé {input_basename}_sorted.csv

Tri effectué (priorités):
 1) `accuracy_diff_pct` asc
 2) `loss_diff_pct` asc
 3) `dev_accuracy` desc
"""
import argparse
import csv
import math
import os
from typing import List, Dict


def to_float_safe(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def sort_rows(rows: List[Dict[str,str]], fieldnames: List[str]):
    # Prepare numeric conversions with sensible defaults
    for r in rows:
        r['_accuracy_diff_pct'] = to_float_safe(r.get('accuracy_diff_pct', ''), math.inf)
        r['_loss_diff_pct'] = to_float_safe(r.get('loss_diff_pct', ''), math.inf)
        # for dev_accuracy we want DESC, so use -dev for sorting, missing -> -inf (so they go last)
        dev = to_float_safe(r.get('dev_accuracy', ''), -math.inf)
        r['_neg_dev_accuracy'] = -dev

    rows.sort(key=lambda r: (r['_accuracy_diff_pct'], r['_loss_diff_pct'], r['_neg_dev_accuracy']))

    # cleanup helper keys
    for r in rows:
        del r['_accuracy_diff_pct']
        del r['_loss_diff_pct']
        del r['_neg_dev_accuracy']

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='Chemin vers le fichier CSV à trier')
    args = parser.parse_args()

    inp = args.csvfile
    if not os.path.isfile(inp):
        raise SystemExit(f"Fichier introuvable: {inp}")

    base, ext = os.path.splitext(inp)
    out = f"{base}_sorted{ext}"

    with open(inp, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise SystemExit("Fichier CSV sans en-tête détecté")

    sorted_rows = sort_rows(rows, fieldnames)

    # write back preserving original field order
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_rows:
            writer.writerow(r)

    # Print a short summary
    print(f"Tri terminé — {len(sorted_rows)} lignes écrites dans: {out}")
    print("Top 5 après tri (dev_accuracy, accuracy_diff_pct, loss_diff_pct):")
    for r in sorted_rows[:5]:
        da = r.get('dev_accuracy','')
        ad = r.get('accuracy_diff_pct','')
        ld = r.get('loss_diff_pct','')
        print(f"  dev_accuracy={da}, accuracy_diff_pct={ad}, loss_diff_pct={ld}, model={r.get('model','')}")


if __name__ == '__main__':
    main()
