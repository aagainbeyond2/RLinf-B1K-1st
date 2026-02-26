import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from dataclasses import dataclass
from typing import Any

import dill
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models_pytorch.pi_behavior_pytorch import PiBehaviorPytorch


def _getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


@dataclass
class ObsAdapter:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None
    token_loss_mask: torch.Tensor | None
    fast_tokens: torch.Tensor | None
    fast_token_mask: torch.Tensor | None

    @staticmethod
    def from_jax_obs(jax_obs: Any, *, device: str, float_dtype: torch.dtype) -> "ObsAdapter":
        images_in = _getattr(jax_obs, "images", None)
        image_masks_in = _getattr(jax_obs, "image_masks", None)
        state_in = _getattr(jax_obs, "state", None)
        tokenized_prompt_in = _getattr(jax_obs, "tokenized_prompt", None)
        tokenized_prompt_mask_in = _getattr(jax_obs, "tokenized_prompt_mask", None)
        token_ar_mask_in = _getattr(jax_obs, "token_ar_mask", None)
        token_loss_mask_in = _getattr(jax_obs, "token_loss_mask", None)
        fast_tokens_in = _getattr(jax_obs, "fast_tokens", None)
        fast_token_mask_in = _getattr(jax_obs, "fast_token_mask", None)

        if isinstance(jax_obs, dict):
            images_in = jax_obs.get("images", images_in)
            image_masks_in = jax_obs.get("image_masks", image_masks_in)
            state_in = jax_obs.get("state", state_in)
            tokenized_prompt_in = jax_obs.get("tokenized_prompt", tokenized_prompt_in)
            tokenized_prompt_mask_in = jax_obs.get(
                "tokenized_prompt_mask", tokenized_prompt_mask_in
            )
            token_ar_mask_in = jax_obs.get("token_ar_mask", token_ar_mask_in)
            token_loss_mask_in = jax_obs.get("token_loss_mask", token_loss_mask_in)
            fast_tokens_in = jax_obs.get("fast_tokens", fast_tokens_in)
            fast_token_mask_in = jax_obs.get("fast_token_mask", fast_token_mask_in)

        images: dict[str, torch.Tensor] = {}
        image_masks: dict[str, torch.Tensor] = {}
        if images_in is None or image_masks_in is None:
            raise ValueError("jax_obs must contain images and image_masks")

        for k, v in dict(images_in).items():
            img_np = _as_numpy(v)
            if img_np.ndim == 4:
                img_np = np.expand_dims(img_np, axis=1)
            if not img_np.flags.writeable:
                img_np = img_np.copy()
            img_pt = (
                torch.from_numpy(img_np)
                .permute(0, 1, 4, 2, 3)
                .to(device=device, dtype=float_dtype)
                .contiguous()
            )
            images[str(k)] = img_pt

        for k, v in dict(image_masks_in).items():
            mask_np = _as_numpy(v)
            if mask_np.ndim == 1:
                mask_np = np.expand_dims(mask_np, axis=1)
            if not mask_np.flags.writeable:
                mask_np = mask_np.copy()
            image_masks[str(k)] = (
                torch.from_numpy(mask_np).to(device=device).contiguous()
            )

        if state_in is None or tokenized_prompt_in is None:
            raise ValueError("jax_obs must contain state and tokenized_prompt")

        state_np = _as_numpy(state_in)
        if not state_np.flags.writeable:
            state_np = state_np.copy()
        state = torch.from_numpy(state_np).to(device=device, dtype=float_dtype)
        tokenized_prompt = torch.from_numpy(_as_numpy(tokenized_prompt_in)).to(
            device=device
        )

        if tokenized_prompt_mask_in is None:
            tokenized_prompt_mask = torch.ones_like(
                tokenized_prompt, dtype=torch.bool, device=device
            )
        else:
            tokenized_prompt_mask = torch.from_numpy(_as_numpy(tokenized_prompt_mask_in)).to(
                device=device
            )

        token_ar_mask = (
            None
            if token_ar_mask_in is None
            else torch.from_numpy(_as_numpy(token_ar_mask_in)).to(device=device)
        )
        token_loss_mask = (
            None
            if token_loss_mask_in is None
            else torch.from_numpy(_as_numpy(token_loss_mask_in)).to(device=device)
        )

        fast_tokens = (
            None
            if fast_tokens_in is None
            else torch.from_numpy(_as_numpy(fast_tokens_in)).to(device=device)
        )
        fast_token_mask = (
            None
            if fast_token_mask_in is None
            else torch.from_numpy(_as_numpy(fast_token_mask_in)).to(device=device)
        )

        return ObsAdapter(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
            fast_tokens=fast_tokens,
            fast_token_mask=fast_token_mask,
        )


def _tensor_stats(x: torch.Tensor) -> str:
    x = x.detach()
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
        f"min={x.min().item():.6f} max={x.max().item():.6f} "
        f"mean={x.mean().item():.6f} std={x.std(unbiased=False).item():.6f}"
    )


def _print_losses(losses: dict[str, torch.Tensor]) -> None:
    keys = [
        "total_loss",
        "action_loss",
        "fast_loss",
        "fast_accuracy",
        "subtask_accuracy",
        "action_loss_base_vel_x",
        "action_loss_base_vel_y",
        "action_loss_base_vel_z",
        "action_loss_trunk_0",
        "action_loss_trunk_1",
        "action_loss_trunk_2",
        "action_loss_trunk_3",
        "action_loss_left_arm_0",
        "action_loss_left_arm_1",
        "action_loss_left_arm_2",
        "action_loss_left_arm_3",
        "action_loss_left_arm_4",
        "action_loss_left_arm_5",
        "action_loss_left_arm_6",
        "action_loss_left_gripper",
        "action_loss_right_arm_0",
        "action_loss_right_arm_1",
        "action_loss_right_arm_2",
        "action_loss_right_arm_3",
        "action_loss_right_arm_4",
        "action_loss_right_arm_5",
        "action_loss_right_arm_6",
        "action_loss_right_gripper",
    ]
    for k in keys:
        v = losses.get(k, None)
        if v is None:
            continue
        if torch.is_tensor(v):
            print(f"{k}: {v.item():.6f}")
        else:
            print(f"{k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduce_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--num_flow_samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    reproduce_dir = os.path.abspath(os.path.expanduser(args.reproduce_dir))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but torch.cuda.is_available() is False")

    device = args.device
    if device == "cuda":
        torch.cuda.set_device(int(args.cuda_device))

    if device == "cpu":
        float_dtype = torch.float32
    else:
        if args.dtype == "fp32":
            float_dtype = torch.float32
        elif args.dtype == "fp16":
            float_dtype = torch.float16
        else:
            float_dtype = torch.bfloat16
    print(f"device={device} dtype={float_dtype} reproduce_dir={reproduce_dir}")

    with open(os.path.join(reproduce_dir, "config_from_jax.pkl"), "rb") as f:
        jax_config = dill.load(f)
    with open(os.path.join(reproduce_dir, "batch_from_jax.pkl"), "rb") as f:
        jax_obs, jax_actions = dill.load(f)

    noise = None
    time = None
    noise_time_path = os.path.join(reproduce_dir, "batch_noise_time.pkl")
    if os.path.exists(noise_time_path):
        with open(noise_time_path, "rb") as f:
            noise_time = dill.load(f)
        noise = noise_time.get("noise", None)
        time = noise_time.get("time", None)

    model_config = _getattr(jax_config, "model", None) or jax_config
    model = PiBehaviorPytorch(model_config)
    model = model.to(device=device, dtype=float_dtype)
    model.eval()

    weights_path = args.weights
    if weights_path is None:
        weights_path = os.path.join(reproduce_dir, "final_b1k_pytorch.pt")
    weights_path = os.path.abspath(os.path.expanduser(weights_path))
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(
        state_dict["state_dict"], dict
    ):
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    obs = ObsAdapter.from_jax_obs(jax_obs, device=device, float_dtype=float_dtype)
    for k in sorted(obs.images.keys()):
        print(f"image[{k}] {_tensor_stats(obs.images[k])}")
    for k in sorted(obs.image_masks.keys()):
        print(f"image_mask[{k}] {_tensor_stats(obs.image_masks[k].to(dtype=torch.float32))}")
    print(f"state {_tensor_stats(obs.state)}")

    actions = torch.from_numpy(_as_numpy(jax_actions)).to(device=device, dtype=float_dtype)
    print(f"actions {_tensor_stats(actions)}")
    if actions.numel() > 0:
        preview = actions[0, 0, : min(12, actions.shape[-1])].detach().cpu().tolist()
        print(f"actions[0,0,:12]={preview}")
    if noise is not None:
        noise = torch.from_numpy(_as_numpy(noise)).to(device=device, dtype=float_dtype)
    if time is not None:
        time = torch.from_numpy(_as_numpy(time)).to(device=device, dtype=float_dtype)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
    )
    with torch.no_grad(), autocast_ctx:
        losses = model.forward_detailed(
            obs,
            actions,
            noise=noise,
            time=time,
            num_flow_samples=int(args.num_flow_samples),
            train=False,
        )

    _print_losses(losses)
    obs_dict: dict[str, Any] = {
        "state": obs.state,
        "image": dict(obs.images),
        "image_mask": dict(obs.image_masks),
        "tokenized_prompt": obs.tokenized_prompt,
        "tokenized_prompt_mask": obs.tokenized_prompt_mask,
    }
    if obs.token_ar_mask is not None:
        obs_dict["token_ar_mask"] = obs.token_ar_mask
    if obs.token_loss_mask is not None:
        obs_dict["token_loss_mask"] = obs.token_loss_mask
    if obs.fast_tokens is not None:
        obs_dict["fast_tokens"] = obs.fast_tokens
    if obs.fast_token_mask is not None:
        obs_dict["fast_token_mask"] = obs.fast_token_mask

    observation = _model.Observation.from_dict(obs_dict)
    with torch.no_grad(), autocast_ctx:
        try:
            pred_actions, _ = model.sample_actions(
                device=device,
                observation=observation,
                num_steps=int(getattr(model.config, "num_steps", 10)),
                noise=noise,
            )
        except Exception as e:
            print(f"sample_actions_with_noise_failed: {e}")
            pred_actions, _ = model.sample_actions(
                device=device,
                observation=observation,
                num_steps=int(getattr(model.config, "num_steps", 10)),
                noise=None,
            )
    print(f"pred_actions {_tensor_stats(pred_actions)}")
    if pred_actions.numel() > 0:
        preview = pred_actions[0, 0, : min(12, pred_actions.shape[-1])].detach().cpu().tolist()
        print(f"pred_actions[0,0,:12]={preview}")


if __name__ == "__main__":
    main()
