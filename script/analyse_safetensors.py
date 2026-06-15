#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from safetensors import safe_open


def dtype_nbytes(dtype: str) -> int:
    """
    safetensors 返回的 dtype 通常是：
    F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL 等
    """
    mapping = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    return mapping.get(dtype, 0)


def numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def analyze_safetensors_file(path: Path):
    rows = []
    total_params = 0
    total_bytes = 0

    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensor_slice = f.get_slice(name)

            shape = tensor_slice.get_shape()
            dtype = str(tensor_slice.get_dtype())

            n_params = numel(shape)
            n_bytes = n_params * dtype_nbytes(dtype)

            rows.append({
                "file": path.name,
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "numel": n_params,
                "bytes": n_bytes,
            })

            total_params += n_params
            total_bytes += n_bytes

    return rows, total_params, total_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tensor shapes and dtypes in safetensors model weights."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory containing .safetensors files",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "file", "size"],
        default="file",
        help="Sort output by tensor name, file name, or tensor size",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only print tensors whose name contains this substring",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Path does not exist: {model_dir}")

    safetensor_files = sorted(model_dir.rglob("*.safetensors"))

    if not safetensor_files:
        print(f"No .safetensors files found under: {model_dir}")
        return

    all_rows = []
    grand_total_params = 0
    grand_total_bytes = 0

    for path in safetensor_files:
        rows, total_params, total_bytes = analyze_safetensors_file(path)
        all_rows.extend(rows)
        grand_total_params += total_params
        grand_total_bytes += total_bytes

    if args.filter:
        all_rows = [
            row for row in all_rows
            if args.filter in row["name"]
        ]

    if args.sort == "name":
        all_rows.sort(key=lambda x: x["name"])
    elif args.sort == "file":
        all_rows.sort(key=lambda x: (x["file"], x["name"]))
    elif args.sort == "size":
        all_rows.sort(key=lambda x: x["bytes"], reverse=True)

    print("=" * 120)
    print(f"{'File':30} {'Tensor Name':60} {'Shape':25} {'DType':8} {'Numel':15} {'Size'}")
    print("=" * 120)

    for row in all_rows:
        print(
            f"{row['file'][:30]:30} "
            f"{row['name'][:60]:60} "
            f"{str(row['shape'])[:25]:25} "
            f"{row['dtype']:8} "
            f"{row['numel']:15,d} "
            f"{format_size(row['bytes'])}"
        )

    print("=" * 120)
    print(f"Safetensors files: {len(safetensor_files)}")
    print(f"Printed tensors:    {len(all_rows)}")
    print(f"Total params:       {grand_total_params:,}")
    print(f"Estimated size:     {format_size(grand_total_bytes)}")
    print("=" * 120)


if __name__ == "__main__":
    main()
