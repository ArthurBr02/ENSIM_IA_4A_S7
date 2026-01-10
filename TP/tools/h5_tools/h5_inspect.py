#!/usr/bin/env python3
"""Inspecteur H5 minimal (copie placée dans tools/h5_tools).

Usage:
  python h5_inspect.py path/to/file.h5 [--max-items N]

Affiche : groupes racine, datasets, shapes, dtypes et un petit extrait des données.
"""
import argparse
import h5py
import numpy as np
import sys


def print_attrs(name, obj):
    typ = 'Group' if isinstance(obj, h5py.Group) else 'Dataset'
    try:
        shape = getattr(obj, 'shape', None)
        dtype = getattr(obj, 'dtype', None)
    except Exception:
        shape = None
        dtype = None
    print(f"{name}: {typ} shape={shape} dtype={dtype}")


def safe_preview(value, max_items=3):
    try:
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                out = []
                for i, v in enumerate(value):
                    if i >= max_items:
                        break
                    out.append(type(v).__name__ + f" len={getattr(v,'shape', getattr(v,'__len__', 'N/A'))}")
                return out
            else:
                flat = value.flatten()
                preview = flat[:max_items].tolist()
                return preview
        else:
            return repr(value)[:200]
    except Exception as e:
        return f"<preview failed: {e}>"


def inspect_file(path, max_items=3):
    print(f"Opening H5 file: {path}\n")
    with h5py.File(path, 'r') as h5f:
        print("Top-level keys:")
        for k in h5f.keys():
            obj = h5f[k]
            print_attrs(k, obj)

        print("\nWalking all items (limited preview):")
        for name, obj in h5f.items():
            print_attrs(name, obj)
            if isinstance(obj, h5py.Dataset):
                try:
                    data = obj[()]
                    print("  -> preview:", safe_preview(data, max_items))
                except Exception as e:
                    print(f"  -> could not read dataset contents: {e}")
            elif isinstance(obj, h5py.Group):
                for subname, subobj in obj.items():
                    print_attrs(f"  {name}/{subname}", subobj)
                    if isinstance(subobj, h5py.Dataset):
                        try:
                            data = subobj[()]
                            print("    -> preview:", safe_preview(data, max_items))
                        except Exception as e:
                            print(f"    -> cannot read: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5file')
    parser.add_argument('--max-items', type=int, default=3)
    args = parser.parse_args()

    try:
        inspect_file(args.h5file, max_items=args.max_items)
    except Exception as e:
        print(f"Error inspecting file: {e}", file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
