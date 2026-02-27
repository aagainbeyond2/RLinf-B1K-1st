import argparse
import os
import sys
import dill
import dataclasses
import numpy as np
import torch


def _stats(x):
    if torch.is_tensor(x):
        shape = tuple(x.shape)
        dtype = x.dtype
        device = x.device
        if x.numel() == 0:
            return f"shape={shape} dtype={dtype} device={device}"
        x_f = x.detach().to(device="cpu", dtype=torch.float32)
        return (
            f"shape={shape} dtype={dtype} device={device} "
            f"min={float(x_f.min()):.6f} max={float(x_f.max()):.6f} "
            f"mean={float(x_f.mean()):.6f} std={float(x_f.std()):.6f}"
        )
    if isinstance(x, np.ndarray):
        shape = tuple(x.shape)
        dtype = x.dtype
        if x.size == 0:
            return f"shape={shape} dtype={dtype}"
        x_f = x.astype(np.float32, copy=False)
        return (
            f"shape={shape} dtype={dtype} "
            f"min={float(x_f.min()):.6f} max={float(x_f.max()):.6f} "
            f"mean={float(x_f.mean()):.6f} std={float(x_f.std()):.6f}"
        )
    if hasattr(x, "__array__") or hasattr(x, "shape"):
        try:
            arr = np.asarray(x)
            shape = tuple(arr.shape)
            dtype = arr.dtype
            if arr.size == 0:
                return f"shape={shape} dtype={dtype}"
            arr_f = arr.astype(np.float32, copy=False)
            return (
                f"shape={shape} dtype={dtype} "
                f"min={float(arr_f.min()):.6f} max={float(arr_f.max()):.6f} "
                f"mean={float(arr_f.mean()):.6f} std={float(arr_f.std()):.6f}"
            )
        except Exception:
            pass
    return f"type={type(x).__name__}"


def _summarize(obj, prefix, lines, depth, max_depth):
    if depth > max_depth:
        lines.append(f"{prefix} depth_exceeded")
        return
    if dataclasses.is_dataclass(obj):
        lines.append(f"{prefix} type={type(obj).__name__}")
        for field in dataclasses.fields(obj):
            key = f"{prefix}.{field.name}" if prefix else field.name
            _summarize(getattr(obj, field.name), key, lines, depth + 1, max_depth)
        return
    if hasattr(obj, "__array__") or hasattr(obj, "shape"):
        lines.append(f"{prefix} {_stats(obj)}")
        return
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        obj_dict = vars(obj)
        lines.append(f"{prefix} type={type(obj).__name__}")
        if not obj_dict:
            lines.append(f"{prefix} empty_object")
            return
        for k in sorted(obj_dict.keys(), key=lambda x: str(x)):
            key = f"{prefix}.{k}" if prefix else str(k)
            _summarize(obj_dict[k], key, lines, depth + 1, max_depth)
        return
    if isinstance(obj, dict):
        if not obj:
            lines.append(f"{prefix} empty_dict")
            return
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            key = f"{prefix}.{k}" if prefix else str(k)
            _summarize(obj[k], key, lines, depth + 1, max_depth)
        return
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        lines.append(f"{prefix} len={n} type={type(obj).__name__}")
        if n == 0:
            return
        for i in range(n):
            key = f"{prefix}[{i}]"
            _summarize(obj[i], key, lines, depth + 1, max_depth)
        return
    lines.append(f"{prefix} {_stats(obj)}")


def _load_pkl(path):
    with open(path, "rb") as f:
        return dill.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="/data1/heyelin/models/behavior_1st_torch/checkpoint_2",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    b1k_src = os.path.join(repo_root, "behavior_1st_solution_torch", "src")
    openpi_src = os.path.join(
        repo_root, "behavior_1st_solution_torch", "openpi", "src"
    )
    openpi_client_src = os.path.join(
        repo_root,
        "behavior_1st_solution_torch",
        "openpi",
        "packages",
        "openpi-client",
        "src",
    )
    if b1k_src not in sys.path:
        sys.path.insert(0, b1k_src)
    if openpi_src not in sys.path:
        sys.path.insert(0, openpi_src)
    if openpi_client_src not in sys.path:
        sys.path.insert(0, openpi_client_src)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    base_dir = os.path.abspath(os.path.expanduser(args.dir))
    files = [
        "batch_from_jax.pkl",
        "batch_noise_time.pkl",
        "config_from_jax.pkl",
        "data_config_from_jax.pkl",
    ]

    for name in files:
        path = os.path.join(base_dir, name)
        print(f"== {name} ==")
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        obj = _load_pkl(path)
        lines = []
        _summarize(obj, "", lines, 0, args.max_depth)
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
