import argparse
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class _ImageGrid:
    image: np.ndarray
    nrow: int
    ncol: int


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _as_numpy(x: Any) -> Any:
    # torch.Tensor(bfloat16) cannot be converted to numpy directly in many builds.
    # Convert to float32 on CPU first.
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        return x.cpu().numpy()
    return x


def _summarize_tensor(name: str, t: torch.Tensor) -> str:
    shape = tuple(t.shape)
    dtype = str(t.dtype)
    minv = None
    maxv = None
    try:
        if t.numel() > 0 and t.dtype.is_floating_point:
            minv = float(t.min().item())
            maxv = float(t.max().item())
    except Exception:
        pass
    if minv is None:
        return f"{name}: shape={shape} dtype={dtype}"
    return f"{name}: shape={shape} dtype={dtype} min={minv:.6g} max={maxv:.6g}"


def _maybe_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _maybe_import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _to_uint8_rgb(img_chw: np.ndarray) -> np.ndarray:
    # Convert a single image from CHW float/uint8 to HWC uint8 for saving.
    if img_chw.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape={img_chw.shape}")
    c, h, w = img_chw.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels, got c={c} shape={img_chw.shape}")
    x = img_chw
    if np.issubdtype(x.dtype, np.floating):
        # Three common cases:
        # 1) Already in [0, 1] -> scale to [0, 255]
        # 2) Already in [0, 255] -> clamp
        # 3) Normalized (e.g., mean/std) or otherwise out of range -> min-max to [0, 255]
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        if 0.0 - 1e-6 <= x_min and x_max <= 1.0 + 1e-6:
            x = x * 255.0
            x = np.clip(x, 0.0, 255.0).astype(np.uint8)
        elif 0.0 - 1e-6 <= x_min and x_max <= 255.0 + 1e-6:
            x = np.clip(x, 0.0, 255.0).astype(np.uint8)
        else:
            denom = x_max - x_min
            if not np.isfinite(denom) or denom < 1e-12:
                x = np.zeros_like(x, dtype=np.uint8)
            else:
                x = (x - x_min) / denom
                x = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


def _make_grid(images_hwc: list[np.ndarray], ncol: int) -> _ImageGrid:
    # Tile a list of equal-sized HWC images into a single grid image.
    if len(images_hwc) == 0:
        raise ValueError("No images to grid")
    h = images_hwc[0].shape[0]
    w = images_hwc[0].shape[1]
    for img in images_hwc:
        if img.shape[:2] != (h, w):
            raise ValueError("All images must have same H,W")
    n = len(images_hwc)
    ncol = max(1, min(ncol, n))
    nrow = int(math.ceil(n / ncol))
    grid = np.zeros((nrow * h, ncol * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images_hwc):
        r = idx // ncol
        c = idx % ncol
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    return _ImageGrid(image=grid, nrow=nrow, ncol=ncol)


def _parse_floats(s: str) -> list[float]:
    # Accept either comma-separated or whitespace-separated triples.
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def _apply_denorm(
    img_chw: torch.Tensor, mean: list[float], std: list[float], clamp01: bool
) -> torch.Tensor:
    if img_chw.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(img_chw.shape)}")
    if img_chw.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got shape={tuple(img_chw.shape)}")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"mean/std must have 3 values, got mean={mean} std={std}")
    x = img_chw.to(torch.float32)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    x = x * std_t + mean_t
    if clamp01:
        x = x.clamp_(0.0, 1.0)
    return x


def _denorm_preset(preset: str) -> tuple[list[float], list[float]]:
    if preset == "half":
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    if preset == "imagenet":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    raise ValueError(f"Unknown denorm preset: {preset}")


def _extract_pixel_images(
    pixel_values: torch.Tensor,
    t: int,
    b: int,
    denorm: bool,
    mean: list[float] | None,
    std: list[float] | None,
    denorm_clamp01: bool,
    num_images_in_input: int | None,
    transform_index: int,
    show_all_transforms: bool,
    denorm_post_scale: float,
) -> list[np.ndarray]:
    # Handle two common pixel_values layouts:
    # - [T, B, 3*N, H, W] (OpenVLA-style packed multi-image)
    # - [T, B, N, 3, H, W] (explicit N images)
    pv = pixel_values
    if pv.ndim == 5:
        pv_tb = pv[t, b]
        c_total = int(pv_tb.shape[0])
        h = int(pv_tb.shape[1])
        w = int(pv_tb.shape[2])
        if num_images_in_input is not None:
            if num_images_in_input <= 0 or c_total % num_images_in_input != 0:
                return []
            c_per_img = int(c_total // num_images_in_input)
            if c_per_img % 3 != 0:
                return []
            n_tf = int(c_per_img // 3)
            if show_all_transforms:
                pv_imgs = pv_tb.reshape(num_images_in_input, n_tf, 3, h, w)
                img_list: list[torch.Tensor] = [
                    pv_imgs[i, j] for i in range(num_images_in_input) for j in range(n_tf)
                ]
            else:
                if not (0 <= transform_index < n_tf):
                    return []
                pv_imgs = pv_tb.reshape(num_images_in_input, n_tf, 3, h, w)
                img_list = [pv_imgs[i, transform_index] for i in range(num_images_in_input)]
        else:
            if c_total % 3 != 0:
                return []
            n_img = int(c_total // 3)
            pv_imgs = pv_tb.reshape(n_img, 3, h, w)
            img_list = [pv_imgs[i] for i in range(n_img)]
        imgs: list[np.ndarray] = []
        for img in img_list:
            if denorm:
                if mean is None or std is None:
                    raise ValueError("--denorm requires --mean and --std")
                img = _apply_denorm(img, mean=mean, std=std, clamp01=denorm_clamp01)
                if denorm_post_scale != 1.0:
                    img = img.to(torch.float32) * float(denorm_post_scale)
            imgs.append(_to_uint8_rgb(_as_numpy(img)))
        return imgs
    if pv.ndim == 6:
        pv_tb = pv[t, b]
        if pv_tb.ndim != 4:
            return []
        if pv_tb.shape[1] == 3:
            imgs = []
            for i in range(pv_tb.shape[0]):
                img = pv_tb[i]
                if denorm:
                    if mean is None or std is None:
                        raise ValueError("--denorm requires --mean and --std")
                    img = _apply_denorm(img, mean=mean, std=std, clamp01=denorm_clamp01)
                    if denorm_post_scale != 1.0:
                        img = img.to(torch.float32) * float(denorm_post_scale)
                imgs.append(_to_uint8_rgb(_as_numpy(img)))
            return imgs
        return []
    return []


def _plot_series(
    out_path: str, series: np.ndarray, title: str, xlabel: str, ylabel: str
) -> bool:
    # Plot to PNG if matplotlib is available; otherwise caller can fall back to CSV.
    plt = _maybe_import_matplotlib_pyplot()
    if plt is None:
        return False
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(series)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_heatmap(
    out_path: str,
    mat: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "viridis",
) -> bool:
    plt = _maybe_import_matplotlib_pyplot()
    if plt is None:
        return False
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_action_tokens_lines(
    out_path: str,
    series_ta: np.ndarray,
    title: str,
    xlabel: str = "t",
    ylabel: str = "token value",
) -> bool:
    plt = _maybe_import_matplotlib_pyplot()
    if plt is None:
        return False
    if series_ta.ndim != 2:
        raise ValueError(f"Expected [T, action_dim], got shape={series_ta.shape}")
    t_len, action_dim = series_ta.shape
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(t_len)
    for j in range(action_dim):
        ax.plot(x, series_ta[:, j], linewidth=1.0, alpha=0.9, label=f"a{j}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if action_dim <= 32:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            ncol=1,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_action_tokens_all_chunks(
    out_path: str,
    tokens_tca: np.ndarray,
    title: str,
    xlabel: str = "t",
    ylabel: str = "token value",
) -> bool:
    plt = _maybe_import_matplotlib_pyplot()
    if plt is None:
        return False
    if tokens_tca.ndim != 3:
        raise ValueError(f"Expected [T, C, action_dim], got shape={tokens_tca.shape}")
    t_len, n_chunk, action_dim = tokens_tca.shape
    ncol = int(math.ceil(math.sqrt(n_chunk)))
    nrow = int(math.ceil(n_chunk / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrow, ncol)
    x = np.arange(t_len)
    for chunk in range(n_chunk):
        r = chunk // ncol
        c = chunk % ncol
        ax = axes[r, c]
        for j in range(action_dim):
            ax.plot(x, tokens_tca[:, chunk, j], linewidth=0.8, alpha=0.85)
        ax.set_title(f"chunk {chunk}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    for k in range(n_chunk, nrow * ncol):
        r = k // ncol
        c = k % ncol
        axes[r, c].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _save_image(out_path: str, rgb_hwc: np.ndarray) -> None:
    # Use OpenCV to write PNG/JPEG. Requires opencv-python.
    cv2 = _maybe_import_cv2()
    if cv2 is None:
        raise RuntimeError("cv2 not available; install opencv-python to save images")
    bgr = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(out_path, bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def _resolve_data(payload: Any) -> dict[str, Any]:
    # Newer saved format: {"global_step": ..., "rank": ..., "stage_id": ..., "data": {...}}
    # Older/alternative: directly a dict of tensors.
    if (
        isinstance(payload, dict)
        and "data" in payload
        and isinstance(payload["data"], dict)
    ):
        return payload["data"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unexpected payload type: {type(payload)}")


def _torch_load_safely(path: str) -> Any:
    # Prefer weights_only=True to reduce pickle attack surface when possible.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/data1/heyelin/rlinf/RLinf-main/logs/20260118-15:27:49/test_openvla/rollouts/global_step_0/rollout_rank0_stage0.pt",
    )
    parser.add_argument("--t", type=int, default=0)
    parser.add_argument("--b", type=int, default=0)
    parser.add_argument("--ncol", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--denorm", action="store_true", default=False)
    parser.add_argument(
        "--denorm_preset",
        type=str,
        default="none",
        choices=["none", "half", "imagenet"],
    )
    parser.add_argument("--denorm_no_clamp", action="store_true", default=False)
    parser.add_argument("--mean", type=str, default=None)
    parser.add_argument("--std", type=str, default=None)
    parser.add_argument("--denorm_post_scale", type=float, default=1.0)
    parser.add_argument("--num_images_in_input", type=int, default=None)
    parser.add_argument("--transform_index", type=int, default=0)
    parser.add_argument("--show_all_transforms", action="store_true", default=False)
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="action_tokens chunk index to visualize (for shape [T,B,num_chunks,action_dim])",
    )
    args = parser.parse_args()

    mean = _parse_floats(args.mean) if args.mean is not None else None
    std = _parse_floats(args.std) if args.std is not None else None
    if bool(args.denorm) and (mean is None or std is None) and args.denorm_preset != "none":
        mean, std = _denorm_preset(args.denorm_preset)

    payload = _torch_load_safely(args.path)
    data = _resolve_data(payload)
    episode = (
        payload.get("episode", None)
        if isinstance(payload, dict) and isinstance(payload.get("episode", None), dict)
        else None
    )

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(args.path), "viz")
    out_dir = _ensure_dir(out_dir)

    print(f"loaded: {args.path}")
    if isinstance(payload, dict):
        meta_keys = [k for k in payload.keys() if k != "data"]
        if len(meta_keys) > 0:
            meta = {k: payload.get(k) for k in meta_keys}
            print(f"meta: {meta}")

    keys = sorted(list(data.keys()))
    print(f"keys({len(keys)}): {keys}")
    for k in keys:
        v = data[k]
        if isinstance(v, torch.Tensor):
            print(_summarize_tensor(k, v))
        else:
            print(f"{k}: type={type(v)}")
    if episode is not None:
        episode_keys = sorted(list(episode.keys()))
        print(f"episode_keys({len(episode_keys)}): {episode_keys}")
        for k in episode_keys:
            v = episode[k]
            if isinstance(v, torch.Tensor):
                print(_summarize_tensor(f"episode.{k}", v))
            else:
                print(f"episode.{k}: type={type(v)}")

    if "pixel_values" in data and isinstance(data["pixel_values"], torch.Tensor):
        imgs = _extract_pixel_images(
            data["pixel_values"],
            t=args.t,
            b=args.b,
            denorm=bool(args.denorm),
            mean=mean,
            std=std,
            denorm_clamp01=not bool(args.denorm_no_clamp),
            num_images_in_input=args.num_images_in_input,
            transform_index=args.transform_index,
            show_all_transforms=bool(args.show_all_transforms),
            denorm_post_scale=float(args.denorm_post_scale),
        )
        if len(imgs) > 0:
            grid = _make_grid(imgs, ncol=args.ncol)
            img_path = os.path.join(out_dir, f"pixel_values_t{args.t}_b{args.b}.png")
            _save_image(img_path, grid.image)
            print(f"saved image: {img_path} (nrow={grid.nrow} ncol={grid.ncol})")
        else:
            print("pixel_values present but could not be parsed into RGB images")

    if "action_tokens" in data and isinstance(data["action_tokens"], torch.Tensor):
        at = data["action_tokens"]
        if at.ndim == 4:
            if not (0 <= args.chunk < at.shape[2]):
                raise ValueError(
                    f"--chunk={args.chunk} out of range for action_tokens.shape={tuple(at.shape)}"
                )
            n_chunk = int(at.shape[2])
            for chunk_idx in range(n_chunk):
                series_ta = _as_numpy(at[:, args.b, chunk_idx, :]).astype(np.float32)
                chunk_plot_path = os.path.join(
                    out_dir, f"action_tokens_lines_b{args.b}_chunk{chunk_idx}.png"
                )
                ok = _plot_action_tokens_lines(
                    chunk_plot_path,
                    series_ta,
                    title=f"action_tokens lines (b={args.b}, chunk={chunk_idx})",
                )
                if ok:
                    print(f"saved plot: {chunk_plot_path}")
                else:
                    print("matplotlib not available; skipped action_tokens line plots")
                    break

            all_chunks = _as_numpy(at[:, args.b, :, :]).astype(np.float32)
            all_chunks_path = os.path.join(
                out_dir, f"action_tokens_lines_b{args.b}_all_chunks.png"
            )
            ok = _plot_action_tokens_all_chunks(
                all_chunks_path,
                all_chunks,
                title=f"action_tokens lines (b={args.b}, all chunks)",
            )
            if ok:
                print(f"saved plot: {all_chunks_path}")
            else:
                print("matplotlib not available; skipped all-chunks plot")
        else:
            print(f"action_tokens present but unsupported shape={tuple(at.shape)}")

    if "rewards" in data and isinstance(data["rewards"], torch.Tensor):
        r = data["rewards"]
        r_np = _as_numpy(r)
        if r_np.ndim >= 2:
            series = r_np[:, args.b]
            if series.ndim > 1:
                series = series.sum(axis=-1)
            series = series.astype(np.float32).flatten()
            plot_path = os.path.join(out_dir, f"rewards_sum_t_b{args.b}.png")
            ok = _plot_series(
                plot_path,
                series,
                title="rewards (sum over chunk dim if present)",
                xlabel="t",
                ylabel="reward",
            )
            if ok:
                print(f"saved plot: {plot_path}")
            else:
                csv_path = os.path.join(out_dir, f"rewards_sum_b{args.b}.csv")
                np.savetxt(csv_path, series, delimiter=",")
                print(f"saved csv: {csv_path} (matplotlib not available)")

    if "dones" in data and isinstance(data["dones"], torch.Tensor):
        d = data["dones"]
        d_np = _as_numpy(d)
        if d_np.ndim >= 2:
            series = d_np[:, args.b]
            if series.ndim > 1:
                series = series.any(axis=-1)
            series = series.astype(np.int32).flatten()
            plot_path = os.path.join(out_dir, f"dones_any_b{args.b}.png")
            ok = _plot_series(
                plot_path,
                series,
                title="dones (any over chunk dim if present)",
                xlabel="t",
                ylabel="done",
            )
            if ok:
                print(f"saved plot: {plot_path}")
            else:
                csv_path = os.path.join(out_dir, f"dones_any_b{args.b}.csv")
                np.savetxt(csv_path, series, delimiter=",", fmt="%d")
                print(f"saved csv: {csv_path} (matplotlib not available)")

    if episode is not None:
        for k in sorted(list(episode.keys())):
            v = episode[k]
            if not isinstance(v, torch.Tensor):
                continue
            v_np = _as_numpy(v)
            if v_np.ndim < 2:
                continue
            series = v_np[:, args.b]
            if series.ndim > 1:
                flat = series.reshape(series.shape[0], -1)
                if flat.dtype == np.bool_:
                    series = flat.any(axis=-1)
                else:
                    series = flat.mean(axis=-1)
            series = series.astype(np.float32).flatten()
            plot_path = os.path.join(out_dir, f"episode_{k}_b{args.b}.png")
            ok = _plot_series(
                plot_path,
                series,
                title=f"episode.{k}",
                xlabel="t",
                ylabel=k,
            )
            if ok:
                print(f"saved plot: {plot_path}")
            else:
                csv_path = os.path.join(out_dir, f"episode_{k}_b{args.b}.csv")
                np.savetxt(csv_path, series, delimiter=",")
                print(f"saved csv: {csv_path} (matplotlib not available)")


if __name__ == "__main__":
    main()

