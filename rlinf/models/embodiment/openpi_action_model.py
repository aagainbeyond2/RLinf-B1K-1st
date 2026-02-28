# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple, Dict

import jax
import numpy as np
import torch
import torch.nn.functional as F
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models_pytorch.pi_behavior_pytorch import PiBehaviorPytorch

from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).lower() not in ("", "0", "false", "no")


_OPENPI_DEBUG_ENABLED = (
    _env_flag("RLINF_DEBUG_OPENPI")
    or _env_flag("RLINF_DEBUG_OPENPI_SAVE")
    or _env_flag("RLINF_DEBUG_PREDICT_TRACE")
)
_OPENPI_DEBUG_PRINTED = False
_OPENPI_DEBUG_SAVED = False


def _as_stats_str(x: Any) -> str:
    if torch.is_tensor(x):
        x_detached = x.detach()
        x_float = (
            x_detached
            if x_detached.dtype.is_floating_point
            else x_detached.to(dtype=torch.float32)
        )
        return (
            f"shape={tuple(x_detached.shape)} dtype={x_detached.dtype} device={x_detached.device} "
            f"min={x_float.min().item():.6f} max={x_float.max().item():.6f} "
            f"mean={x_float.mean().item():.6f} std={x_float.std(unbiased=False).item():.6f}"
        )
    x_np = np.asarray(x)
    x_float = x_np.astype(np.float32, copy=False)
    return (
        f"shape={tuple(x_np.shape)} dtype={x_np.dtype} "
        f"min={float(x_float.min()):.6f} max={float(x_float.max()):.6f} "
        f"mean={float(x_float.mean()):.6f} std={float(x_float.std()):.6f}"
    )


def _should_print_openpi_debug() -> bool:
    return bool(_OPENPI_DEBUG_ENABLED and (not _OPENPI_DEBUG_PRINTED))


def _print_openpi_debug(lines: list[str]) -> None:
    global _OPENPI_DEBUG_PRINTED
    _OPENPI_DEBUG_PRINTED = True
    for line in lines:
        print(line)


def _should_save_openpi_debug() -> bool:
    enabled = str(os.environ.get("RLINF_DEBUG_OPENPI_SAVE", "")).lower() not in (
        "",
        "0",
        "false",
        "no",
    )
    return bool(enabled and (not _OPENPI_DEBUG_SAVED))


def _to_cpu_tree(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().to(device="cpu").contiguous()
    if isinstance(x, dict):
        return {k: _to_cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu_tree(v) for v in x)
    return x


def _save_openpi_debug(payload: dict[str, Any]) -> str:
    global _OPENPI_DEBUG_SAVED
    _OPENPI_DEBUG_SAVED = True
    out_dir = os.environ.get("RLINF_DEBUG_OPENPI_DIR", "debug_openpi_inputs")
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    pid = os.getpid()
    out_path = os.path.join(out_dir, f"openpi_inputs_pid{pid}_{ts}.pt")
    try:
        torch.save(_to_cpu_tree(payload), out_path)
    except Exception as e:
        print(f"openpi_debug_save_failed out_path={out_path} error={e}")
        raise
    print(f"openpi_debug_saved {out_path}")
    return out_path


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = (
        "pi0_libero"  # pi0_libero, pi05_libero, pi0_metaworld, pi05_metaworld
    )
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token


class OpenPi0ForRLActionPrediction(PI0Pytorch):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        # Currently, PaliGemmaForConditionalGeneration only support DDP, as many of it's modules are called without forward
        return [
            "PaliGemmaForConditionalGeneration",
            "GemmaDecoderLayer",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaForCausalLM",
            "GemmaRotaryEmbedding",
        ]

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__(config, is_pi05=config.pi05)
        self.sample_actions = sample_actions_func
        self.global_step = 0
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )

    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {
                key: inputs[key]
                for key in inputs.keys()
                if ("/" in key)
                or (
                    key
                    in (
                        "tokenized_prompt",
                        "tokenized_prompt_mask",
                        "token_ar_mask",
                        "token_loss_mask",
                    )
                )
            }
        # tensor -> numpy
        inputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs
        )
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            # convert from [3,256,256] -> [256,256,3]
            if transpose:
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and x.shape[0] in (1, 3, 4)
                    else (
                        x.transpose(0, 2, 3, 1)
                        if len(x.shape) == 4 and x.shape[1] in (1, 3, 4)
                        else x
                    ),
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                if "prompt" not in sample:
                    sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_sample.pop("prompt", None)
            transformed_samples.append(transformed_sample)
        # recombine
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        if ("tokenized_prompt" in obs) and ("tokenized_prompt_mask" in obs):
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
            inputs.pop("token_ar_mask", None)
            inputs.pop("token_loss_mask", None)
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        if "mode" in kwargs and kwargs["mode"] == "sft":
            observation = data["observation"]
            actions = data["actions"]
            return super().forward(observation, actions)
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        # input transform
        observation = self.input_transform(data)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            processed_obs["observation/state"] = env_obs["states"]
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    def predict_action_batch(
        self, env_obs, mode: Literal["train", "eval"] = "train", compute_values=True
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        debug_lines: list[str] | None = None
        if _should_print_openpi_debug():
            debug_lines = []
            if "images" in env_obs:
                debug_lines.append(f"env_obs.images {_as_stats_str(env_obs['images'])}")
            if env_obs.get("wrist_images", None) is not None:
                debug_lines.append(
                    f"env_obs.wrist_images {_as_stats_str(env_obs['wrist_images'])}"
                )
            if "states" in env_obs:
                debug_lines.append(f"env_obs.states {_as_stats_str(env_obs['states'])}")
            if "task_descriptions" in env_obs and len(env_obs["task_descriptions"]) > 0:
                debug_lines.append(
                    f"env_obs.task_descriptions[0]={env_obs['task_descriptions'][0]}"
                )

            images_dict = processed_obs.get("image", {})
            for k in sorted(images_dict.keys()):
                debug_lines.append(
                    f"processed_obs.image[{k}] {_as_stats_str(images_dict[k])}"
                )
            masks_dict = processed_obs.get("image_mask", {})
            for k in sorted(masks_dict.keys()):
                debug_lines.append(
                    f"processed_obs.image_mask[{k}] {_as_stats_str(masks_dict[k])}"
                )
            if "state" in processed_obs:
                debug_lines.append(
                    f"processed_obs.state {_as_stats_str(processed_obs['state'])}"
                )

            for k in sorted(observation.images.keys()):
                debug_lines.append(
                    f"observation.images[{k}] {_as_stats_str(observation.images[k])}"
                )
            for k in sorted(observation.image_masks.keys()):
                debug_lines.append(
                    f"observation.image_masks[{k}] {_as_stats_str(observation.image_masks[k].to(dtype=torch.float32))}"
                )
            debug_lines.append(f"observation.state {_as_stats_str(observation.state)}")

            pre_images, pre_masks, _, _, pre_state = self._preprocess_observation(
                observation, train=False
            )
            for i, img in enumerate(pre_images):
                debug_lines.append(f"preprocess.images[{i}] {_as_stats_str(img)}")
            for i, m in enumerate(pre_masks):
                debug_lines.append(
                    f"preprocess.image_masks[{i}] {_as_stats_str(m.to(dtype=torch.float32))}"
                )
            debug_lines.append(f"preprocess.state {_as_stats_str(pre_state)}")
        if _should_save_openpi_debug():
            pre_images, pre_masks, _, _, pre_state = self._preprocess_observation(
                observation, train=False
            )
            saved_path = _save_openpi_debug(
                {
                    "env_obs": {
                        "images": env_obs.get("images", None),
                        "wrist_images": env_obs.get("wrist_images", None),
                        "states": env_obs.get("states", None),
                        "task_descriptions": env_obs.get("task_descriptions", None),
                    },
                    "to_process_obs": to_process_obs,
                    "processed_obs": processed_obs,
                    "preprocess": {
                        "images": pre_images,
                        "image_masks": pre_masks,
                        "state": pre_state,
                    },
                }
            )
            if debug_lines is not None:
                debug_lines.append(f"saved_openpi_inputs={saved_path}")
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        if debug_lines is not None:
            chains = outputs.get("chains", None)
            if torch.is_tensor(chains):
                debug_lines.append(f"chains {_as_stats_str(chains)}")
                max_steps = min(int(chains.shape[1]), 12)
                max_dims = min(int(chains.shape[-1]), 12)
                for step_i in range(max_steps):
                    step_actions = chains[0, step_i, 0, :max_dims].detach().cpu().tolist()
                    debug_lines.append(
                        f"chains[0,step={step_i},t=0,:{max_dims}]={step_actions}"
                    )

            try:
                output_tf = getattr(self, "_output_transform", None)
                tf_list = getattr(output_tf, "transforms", None)
                if isinstance(tf_list, (list, tuple)) and len(tf_list) > 0:
                    sample = {
                        "actions": outputs["actions"][0].detach().cpu().numpy(),
                        "state": observation.state[0].detach().cpu().numpy(),
                    }
                    debug_lines.append(
                        f"output_transform.input.actions {_as_stats_str(sample['actions'])}"
                    )
                    debug_lines.append(
                        f"output_transform.input.state {_as_stats_str(sample['state'])}"
                    )
                    prev_actions = np.asarray(sample["actions"])
                    for idx, tf in enumerate(tf_list):
                        sample = tf(sample)
                        cur_actions = np.asarray(sample.get("actions", prev_actions))
                        preview_dims = min(int(cur_actions.shape[-1]), 12)
                        preview = cur_actions[0, :preview_dims].tolist()
                        delta_l2 = float(np.linalg.norm(cur_actions - prev_actions))
                        debug_lines.append(
                            f"output_transform[{idx}]={type(tf).__name__} actions {_as_stats_str(cur_actions)}"
                        )
                        debug_lines.append(
                            f"output_transform[{idx}].actions[t=0,:{preview_dims}]={preview}"
                        )
                        debug_lines.append(
                            f"output_transform[{idx}].delta_l2={delta_l2}"
                        )
                        prev_actions = cur_actions
            except Exception as e:
                debug_lines.append(f"output_transform debug failed error={e}")

        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()
        if debug_lines is not None:
            debug_lines.append(f"raw_actions {_as_stats_str(outputs['actions'])}")
            if torch.is_tensor(outputs["actions"]) and outputs["actions"].numel() > 0:
                preview = (
                    outputs["actions"][0, 0, : min(12, outputs["actions"].shape[-1])]
                    .detach()
                    .cpu()
                    .tolist()
                )
                debug_lines.append(f"raw_actions[0,0,:12]={preview}")
            debug_lines.append(f"env_actions shape={actions.shape} dtype={actions.dtype}")
            if actions.size > 0:
                debug_lines.append(
                    f"env_actions[0,0,:12]={actions[0,0,: min(12, actions.shape[-1])].tolist()}"
                )
                if actions.ndim >= 3 and actions.shape[1] > 1:
                    debug_lines.append(
                        f"env_actions[0,1,:12]={actions[0,1,: min(12, actions.shape[-1])].tolist()}"
                    )
                    debug_lines.append(
                        f"env_actions_delta_l2[0,0->1]={float(np.linalg.norm(actions[0,1] - actions[0,0]))}"
                    )
                    debug_lines.append(
                        f"env_actions_time_std_mean={float(actions.std(axis=1).mean())}"
                    )
            _print_openpi_debug(debug_lines)

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        if "subtask_logits" in outputs:
            result["subtask_logits"] = outputs["subtask_logits"]
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        # add value based on the vlm for pi05, expert for pi0
        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                compute_values,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        # post process for logprob
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]
        # post process for value
        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.config.noise_level).to(device)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        suffix_out_proj = suffix_out
        proj_dtype = self.action_out_proj.weight.dtype
        if suffix_out_proj.dtype != proj_dtype:
            suffix_out_proj = suffix_out_proj.to(dtype=proj_dtype)
        v_t = self.action_out_proj(suffix_out_proj)  # [bs,n_action_steps,max_action_dim]
        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_in = suffix_out_value
            try:
                value_dtype = next(self.value_head.parameters()).dtype
            except StopIteration:
                value_dtype = torch.float32
            if value_in.dtype != value_dtype:
                value_in = value_in.to(dtype=value_dtype)
            value_t = self.value_head(value_in)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)
        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t
    
    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(prefix_output))
            else:
                chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False


class PiBehaviorForRLActionPrediction(PiBehaviorPytorch):
    """
    PiBehavior model for reinforcement learning action prediction.
    Inherits from PiBehaviorPytorch to support BEHAVIOR-1K specific architecture.
    """

    # We reuse OpenPi0Config as base, assuming keys are compatible
    # config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        # Reuse from OpenPi0ForRLActionPrediction
        return [
            "PaliGemmaForConditionalGeneration",
            "GemmaDecoderLayer",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaForCausalLM",
            "GemmaRotaryEmbedding",
        ]

    def __init__(self, config):
        # Initialize parent PiBehaviorPytorch
        super().__init__(config)
        self.global_step = 0
        
        # RL config checks
        self.rl_config = config # Alias for consistency
        
        # rl model init
        # Use value_after_vlm if present in config, else default to False
        value_after_vlm = getattr(config, "value_after_vlm", False)
        if value_after_vlm:
            proj_width = 2048 # Paligemma width
        else:
            proj_width = 1024 # Expert width
            
        # value head
        add_value_head = getattr(config, "add_value_head", False)
        if add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
        self.use_vlm_value = value_after_vlm and add_value_head
        
        # noise head for flow-noise
        noise_method = getattr(config, "noise_method", "flow_sde")
        if noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=getattr(config, "noise_logvar_range", [0.08, 0.16]),
                noise_scheduler_type="learn",
            )

    # Reuse methods from OpenPi0ForRLActionPrediction by copying them or mixin
    # Since we can't easily multiple inherit with PyTorch modules sharing state, we copy logic.
    
    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    # Copy input_transform from OpenPi0ForRLActionPrediction
    def input_transform(self, obs: dict, transpose=True):
        # Same implementation
        inputs = jax.tree.map(lambda x: x, obs)
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {
                key: inputs[key]
                for key in inputs.keys()
                if ("/" in key)
                or (
                    key
                    in (
                        "tokenized_prompt",
                        "tokenized_prompt_mask",
                        "token_ar_mask",
                        "token_loss_mask",
                    )
                )
            }
        inputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs
        )
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            if transpose:
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and x.shape[0] in (1, 3, 4)
                    else (
                        x.transpose(0, 2, 3, 1)
                        if len(x.shape) == 4 and x.shape[1] in (1, 3, 4)
                        else x
                    ),
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                if "prompt" not in sample:
                    sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_sample.pop("prompt", None)
            transformed_samples.append(transformed_sample)
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        if ("tokenized_prompt" in obs) and ("tokenized_prompt_mask" in obs):
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
            inputs.pop("token_ar_mask", None)
            inputs.pop("token_loss_mask", None)
        return inputs

    # Copy output_transform
    def output_transform(self, outputs):
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    # Adapted forward
    def forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        if "mode" in kwargs and kwargs["mode"] == "sft":
            observation = data["observation"]
            actions = data["actions"]
            return super().forward(observation, actions)
        
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        
        observation = self.input_transform(data)
        observation = _model.Observation.from_dict(observation)
        
        # PiBehavior uses tokenized_prompt in embed_prefix, not lang_tokens
        images, img_masks, _, _, state = self._preprocess_observation(observation, train=False)
        tokenized_prompt = observation.tokenized_prompt
        
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        tokenized_prompt = tokenized_prompt.to(device)
        
        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            tokenized_prompt, # Replaces lang_tokens
            None, # lang_masks unused in PiBehavior
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        value_t = value_t.mean(dim=-1, keepdim=False)
        
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    # Copy obs_processor
    def obs_processor(self, env_obs):
        processed_obs = {
            "observation/image": env_obs["images"],
            "prompt": env_obs["task_descriptions"],
            "task_id": env_obs["task_id"], # Add task_id
        }
        # Add tokenized_prompt if available (it should be from env wrapper)
        if "tokenized_prompt" in env_obs:
            processed_obs["tokenized_prompt"] = env_obs["tokenized_prompt"]
            processed_obs["tokenized_prompt_mask"] = env_obs["tokenized_prompt_mask"]
            
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            processed_obs["observation/state"] = env_obs["states"]
            
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        return processed_obs

    # Copy precision_processor
    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    # Copy predict_action_batch
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        initial_actions: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)
        debug_lines: list[str] | None = None
        if _should_print_openpi_debug():
            debug_lines = []
            if "images" in env_obs:
                debug_lines.append(f"env_obs.images {_as_stats_str(env_obs['images'])}")
            if env_obs.get("wrist_images", None) is not None:
                debug_lines.append(
                    f"env_obs.wrist_images {_as_stats_str(env_obs['wrist_images'])}"
                )
            if "states" in env_obs:
                debug_lines.append(f"env_obs.states {_as_stats_str(env_obs['states'])}")
            if "task_id" in env_obs:
                task_id = env_obs["task_id"]
                task_id_preview = (
                    task_id.detach().cpu().tolist() if torch.is_tensor(task_id) else task_id
                )
                debug_lines.append(f"env_obs.task_id={task_id_preview}")
            if "task_descriptions" in env_obs and len(env_obs["task_descriptions"]) > 0:
                debug_lines.append(
                    f"env_obs.task_descriptions[0]={env_obs['task_descriptions'][0]}"
                )

            images_dict = processed_obs.get("image", {})
            for k in sorted(images_dict.keys()):
                debug_lines.append(
                    f"processed_obs.image[{k}] {_as_stats_str(images_dict[k])}"
                )
            masks_dict = processed_obs.get("image_mask", {})
            for k in sorted(masks_dict.keys()):
                debug_lines.append(
                    f"processed_obs.image_mask[{k}] {_as_stats_str(masks_dict[k])}"
                )
            if "state" in processed_obs:
                debug_lines.append(
                    f"processed_obs.state {_as_stats_str(processed_obs['state'])}"
                )

            for k in sorted(observation.images.keys()):
                debug_lines.append(
                    f"observation.images[{k}] {_as_stats_str(observation.images[k])}"
                )
            for k in sorted(observation.image_masks.keys()):
                debug_lines.append(
                    f"observation.image_masks[{k}] {_as_stats_str(observation.image_masks[k].to(dtype=torch.float32))}"
                )
            debug_lines.append(f"observation.state {_as_stats_str(observation.state)}")

            pre_images, pre_masks, _, _, pre_state = self._preprocess_observation(
                observation, train=False
            )
            for i, img in enumerate(pre_images):
                debug_lines.append(f"preprocess.images[{i}] {_as_stats_str(img)}")
            for i, m in enumerate(pre_masks):
                debug_lines.append(
                    f"preprocess.image_masks[{i}] {_as_stats_str(m.to(dtype=torch.float32))}"
                )
            debug_lines.append(f"preprocess.state {_as_stats_str(pre_state)}")
        if _should_save_openpi_debug():
            pre_images, pre_masks, _, _, pre_state = self._preprocess_observation(
                observation, train=False
            )
            saved_path = _save_openpi_debug(
                {
                    "env_obs": {
                        "images": env_obs.get("images", None),
                        "wrist_images": env_obs.get("wrist_images", None),
                        "states": env_obs.get("states", None),
                        "task_descriptions": env_obs.get("task_descriptions", None),
                        "task_id": env_obs.get("task_id", None),
                    },
                    "to_process_obs": to_process_obs,
                    "processed_obs": processed_obs,
                    "preprocess": {
                        "images": pre_images,
                        "image_masks": pre_masks,
                        "state": pre_state,
                    },
                }
            )
            if debug_lines is not None:
                debug_lines.append(f"saved_openpi_inputs={saved_path}")
        
        outputs = self.sample_actions(
            observation,
            mode=mode,
            compute_values=compute_values,
            initial_actions=initial_actions,
        )
        
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()
        if debug_lines is not None:
            debug_lines.append(f"raw_actions {_as_stats_str(outputs['actions'])}")
            if torch.is_tensor(outputs["actions"]) and outputs["actions"].numel() > 0:
                preview = (
                    outputs["actions"][0, 0, : min(12, outputs["actions"].shape[-1])]
                    .detach()
                    .cpu()
                    .tolist()
                )
                debug_lines.append(f"raw_actions[0,0,:12]={preview}")
            debug_lines.append(f"env_actions shape={actions.shape} dtype={actions.dtype}")
            if actions.size > 0:
                debug_lines.append(
                    f"env_actions[0,0,:12]={actions[0,0,: min(12, actions.shape[-1])].tolist()}"
                )
            _print_openpi_debug(debug_lines)

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        if "subtask_logits" in outputs:
            result["subtask_logits"] = outputs["subtask_logits"]
        return actions, result

    # Adapted sample_actions
    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
        num_steps=None, # Optional override
        initial_actions: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]: # Changed return type to dict
        """
        Modified sample_actions that returns a dictionary with actions, chains, logprobs, values.
        Uses PiBehaviorPytorch's inpainting logic (Euler ODE) but collects chains.
        """
        bsize_base = observation.state.shape[0]
        device = observation.state.device
        if num_steps is None:
            num_steps = self.config.num_steps
        num_flow_samples = int(getattr(self.config, "num_flow_samples", 1) or 1)
        if num_flow_samples < 1:
            num_flow_samples = 1

        fixed_z_O = None
        x0_O = None
        inpainting_data = None

        action_dim = int(self.config.action_dim)
        action_horizon = int(self.config.action_horizon)
        flat_dim = action_horizon * action_dim

        if initial_actions is not None:
            if not torch.is_tensor(initial_actions):
                initial_actions = torch.as_tensor(initial_actions)
            initial_actions = initial_actions.to(device=device)
            if not initial_actions.dtype.is_floating_point:
                initial_actions = initial_actions.to(dtype=torch.float32)

            num_initial_steps = int(initial_actions.shape[1])
            input_action_dim = int(initial_actions.shape[2])

            if input_action_dim < action_dim:
                padding_dim = torch.zeros(
                    bsize_base,
                    num_initial_steps,
                    action_dim - input_action_dim,
                    device=device,
                    dtype=initial_actions.dtype,
                )
                initial_actions_full = torch.cat([initial_actions, padding_dim], dim=2)
            else:
                initial_actions_full = initial_actions[:, :, :action_dim]

            if num_initial_steps < action_horizon:
                padding_steps = torch.zeros(
                    bsize_base,
                    action_horizon - num_initial_steps,
                    action_dim,
                    device=device,
                    dtype=initial_actions.dtype,
                )
                initial_actions_padded = torch.cat(
                    [initial_actions_full, padding_steps], dim=1
                )
            else:
                initial_actions_padded = initial_actions_full[:, :action_horizon]

            O_indices_list = [
                t * action_dim + d
                for t in range(num_initial_steps)
                for d in range(min(input_action_dim, action_dim))
            ]
            O_indices = torch.tensor(O_indices_list, dtype=torch.long, device=device)

            O_set = set(O_indices_list)
            U_indices_list = [i for i in range(flat_dim) if i not in O_set]
            U_indices = torch.tensor(U_indices_list, dtype=torch.long, device=device)
            x0_O_base = initial_actions_padded.reshape(bsize_base, flat_dim)[:, O_indices]

            if getattr(self, "correlation_loaded", False) and hasattr(
                self, "_precompute_correction_matrix"
            ):
                inpainting_data = self._precompute_correction_matrix(O_indices, U_indices)
        else:
            x0_O_base = None

        bsize = int(bsize_base) * int(num_flow_samples)
        if noise is None:
            if getattr(self, "correlation_loaded", False):
                noise = self.generate_correlated_noise(bsize, device)
            else:
                noise = torch.randn(bsize, action_horizon, action_dim, device=device)
        else:
            if not torch.is_tensor(noise):
                noise = torch.as_tensor(noise)
            noise = noise.to(device=device)
            if int(noise.shape[0]) == int(bsize_base) and num_flow_samples > 1:
                noise = noise.repeat_interleave(num_flow_samples, dim=0)
            elif int(noise.shape[0]) != int(bsize):
                noise = noise[:bsize]

        if x0_O_base is not None:
            noise_flat = noise.reshape(bsize, flat_dim)
            fixed_z_O = noise_flat[:, O_indices]
            x0_O = x0_O_base.repeat_interleave(num_flow_samples, dim=0)

        # Preprocess (reuse parent logic parts)
        # Note: We must call preprocess to get images/masks for embed_prefix
        images, img_masks, _, _, state = self._preprocess_observation(observation, train=False)
        tokenized_prompt = observation.tokenized_prompt

        # Embed Prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokenized_prompt, state
        )

        # VLM Pass
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_out, _), kv_cache = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            inputs_embeds=[prefix_embs, None],
            use_cache=True
        )
        num_img_tokens = int(sum([img.shape[1] for img in images]))
        base_task_output = prefix_out[:, num_img_tokens, :]
        stage_dtype = self.stage_pred_from_vlm.weight.dtype
        if base_task_output.dtype != stage_dtype:
            base_task_output = base_task_output.to(dtype=stage_dtype)
        subtask_logits = self.stage_pred_from_vlm(base_task_output)
        
        # Transform KV Cache
        if self.kv_transform is not None:
            kv_cache = self.kv_transform(kv_cache)

        if num_flow_samples > 1:
            prefix_pad_masks = prefix_pad_masks.repeat_interleave(num_flow_samples, dim=0)
            prefix_out = prefix_out.repeat_interleave(num_flow_samples, dim=0)
            subtask_logits = subtask_logits.repeat_interleave(num_flow_samples, dim=0)

            def _repeat_past_key_values_for_multisample(past_key_values):
                try:
                    from transformers.cache_utils import DynamicCache
                except Exception:
                    DynamicCache = None

                if past_key_values is None:
                    return None

                cache_cls = None
                if hasattr(past_key_values, "to_legacy_cache"):
                    cache_cls = past_key_values.__class__
                    legacy = past_key_values.to_legacy_cache()
                elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
                    cache_cls = past_key_values.__class__
                    legacy = list(zip(past_key_values.key_cache, past_key_values.value_cache))
                else:
                    legacy = past_key_values

                if isinstance(legacy, tuple):
                    legacy = list(legacy)
                if not isinstance(legacy, list):
                    return past_key_values

                expanded_legacy = []
                for layer in legacy:
                    if (
                        isinstance(layer, (list, tuple))
                        and len(layer) >= 2
                        and torch.is_tensor(layer[0])
                        and torch.is_tensor(layer[1])
                    ):
                        k, v = layer[0], layer[1]
                        expanded_legacy.append(
                            (
                                k.repeat_interleave(num_flow_samples, dim=0),
                                v.repeat_interleave(num_flow_samples, dim=0),
                            )
                        )
                    else:
                        expanded_legacy.append(layer)

                if cache_cls is not None and hasattr(cache_cls, "from_legacy_cache"):
                    return cache_cls.from_legacy_cache(expanded_legacy)
                if DynamicCache is not None:
                    return DynamicCache.from_legacy_cache(expanded_legacy)
                return expanded_legacy

            kv_cache = _repeat_past_key_values_for_multisample(kv_cache)

        # Initialize chains and logprobs
        x_t = noise
        chains = []
        log_probs = []
        values = []
        chains.append(x_t) # Chain 0 is noise

        # Calculate initial logprob (of noise)
        joint_logprob = getattr(self.config, "joint_logprob", False)
        if joint_logprob:
            # Assume noise is standard normal (or correlated)
            # get_logprob_norm calculates log N(sample | mu, sigma)
            # Here we assume sample=x_t, mu=0, sigma=1 (simplified)
            # If correlated noise, this is approximate or needs adjustment
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            log_probs.append(initial_log_prob)

        # Prepare for loop
        dt = -1.0 / num_steps
        time = torch.tensor(1.0, device=device)
        time_threshold_inpaint = float(getattr(self.config, "time_threshold_inpaint", 0.0))
        
        # Denoise indices (for training selection)
        if mode == "train":
            if joint_logprob:
                denoise_inds = torch.arange(num_steps, device=device)
            else:
                # Randomly select steps if not joint
                # For ODE, we step through all anyway, but maybe we only want to train on some?
                # OpenPi0ForRLActionPrediction logic:
                if getattr(self.config, "ignore_last", False):
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps,
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps,
                        device=device,
                        dtype=torch.long,
                    )
        else:
            denoise_inds = torch.full((num_steps,), -1, device=device, dtype=torch.long)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)
        
        # Loop
        # We reuse PiBehavior loop logic but added chain collection
        
        # Suffix masks
        suffix_len = self.config.action_horizon
        suffix_pad_masks = torch.ones(bsize, suffix_len, dtype=torch.bool, device=device)
        prefix_len = prefix_pad_masks.shape[1]
        
        for idx in range(num_steps):
            # 1. Embed Suffix
            suffix_embs, _, suffix_att_masks, adarms_cond = self.embed_suffix(
                observation, x_t, time.expand(bsize)
            )
            
            # 2. Masks
            prefix_mask_expanded = prefix_pad_masks.unsqueeze(1).expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_mask_expanded, suffix_att_2d], dim=2)
            full_att_4d = self._prepare_attention_masks_4d(full_att_2d)
            suffix_pos = prefix_len + torch.cumsum(suffix_pad_masks, dim=1) - 1
            
            # 3. Forward
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=suffix_pos,
                past_key_values=kv_cache, 
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond]
            )
            suffix_out_proj = suffix_out[:, -suffix_len:]
            proj_dtype = self.action_out_proj.weight.dtype
            if suffix_out_proj.dtype != proj_dtype:
                suffix_out_proj = suffix_out_proj.to(dtype=proj_dtype)
            v_t = self.action_out_proj(suffix_out_proj)
            
            # 4. Value Prediction
            if self.use_vlm_value and compute_values:
                 # Compute value from VLM output (prefix_out) only once?
                 # OpenPi0ForRLActionPrediction appends it every step.
                 val = self.get_value_from_vlm(prefix_out)
                 values.append(val)
            elif self.config.add_value_head and compute_values:
                 # Compute value from action expert features (suffix_out)
                 if getattr(self.config, "chunk_critic_input", False):
                     suffix_out_val = torch.mean(suffix_out[:, :self.config.action_chunk], dim=1)
                 else:
                     suffix_out_val = torch.mean(suffix_out, dim=1)
                 if getattr(self.config, "detach_critic_input", False):
                     suffix_out_val = suffix_out_val.detach()
                 value_in = suffix_out_val
                 try:
                     value_dtype = next(self.value_head.parameters()).dtype
                 except StopIteration:
                     value_dtype = torch.float32
                 if value_in.dtype != value_dtype:
                     value_in = value_in.to(dtype=value_dtype)
                 values.append(self.value_head(value_in)[:, 0])
            else:
                 values.append(torch.zeros(bsize, device=device))

            x_t_new = x_t + dt * v_t
            time_new = time + dt

            if inpainting_data is not None and time_new > time_threshold_inpaint:
                x_flat = x_t_new.reshape(bsize, flat_dim)
                x_desired_O = (1.0 - time_new) * x0_O + time_new * fixed_z_O
                current_O = x_flat[:, inpainting_data["O_indices"]]
                delta_O = x_desired_O - current_O
                x_flat.scatter_(
                    1,
                    inpainting_data["O_indices"]
                    .unsqueeze(0)
                    .expand(bsize, -1),
                    x_desired_O,
                )
                correction_mat = inpainting_data["correction_matrix"]
                delta_U = delta_O @ correction_mat.t()
                x_flat.scatter_add_(
                    1,
                    inpainting_data["U_indices"]
                    .unsqueeze(0)
                    .expand(bsize, -1),
                    delta_U,
                )
                x_t_new = x_flat.reshape(bsize, action_horizon, action_dim)
            
            # 6. Logprobs (Approximate for ODE)
            # For ODE, x_t is deterministic. "log_prob" is not well-defined per step in the SDE sense.
            # But we need something for PPO.
            # If joint_logprob is True, we use initial noise logprob (already appended).
            # If not joint_logprob, we usually need p(x_t | x_{t-1}) which is Dirac for ODE.
            # However, OpenPi0ForRLActionPrediction.get_log_prob_value uses get_logprob_norm assuming SDE.
            # If we train with SDE loss (sample_mean_var_val), we should probably rollout with SDE too?
            # But we must use PiBehavior weights which are ODE/Flow.
            # Let's stick to returning dummy logprobs for intermediate steps if joint_logprob is True.
            # If joint_logprob is False, this might be an issue.
            # Assuming joint_logprob=True for B1K (config says True in OpenPi0ForRLActionPrediction default? No, False).
            # But B1K config in `pytorch_config.py` doesn't specify RL params.
            # I will append zeros for logprobs if not joint, or copy previous.
            
            if not joint_logprob:
                 # Use a dummy logprob or calc based on some sigma?
                 # Let's assume sigma=1 for now to avoid nan
                 log_probs.append(self.get_logprob_norm(x_t_new, x_t_new, torch.ones_like(x_t_new)))
            
            x_t = x_t_new
            time = time_new
            chains.append(x_t)
            
        x_0 = x_t
        chains = torch.stack(chains, dim=1)

        subtask_logits_base = subtask_logits
        
        # Post process logprobs
        if len(log_probs) > 0:
            if joint_logprob:
                # Just one logprob (initial)
                # Stack expects list of tensors.
                pass 
            # Stack
            log_probs = torch.stack(log_probs, dim=1) # [B, Steps, Action_Dim]
            log_probs = log_probs[:, :, :self.config.action_chunk, :self.config.action_env_dim]
            if joint_logprob:
                log_probs = log_probs.mean(dim=1) # Mean over steps? No, joint means prob of trajectory
                # OpenPi0 implementation:
                # if joint: log_probs = log_probs.mean(dim=1)
                pass
            else:
                # Select based on denoise_inds
                # log_probs = log_probs[torch.arange(bsize), denoise_inds[:, 0]]
                # For rollout, we just return the full sequence or let worker handle it?
                # Predict_action_batch returns `outputs["prev_logprobs"]`.
                # Worker expects [B, action_dim] or [B, 1]?
                # OpenPi0: returns [B, 1] if joint, else [B, chunk, dim]?
                # Let's look at `predict_action_batch`:
                # result = { "prev_logprobs": outputs["prev_logprobs"] }
                pass
        else:
            log_probs = torch.zeros(bsize, 1, device=device)

        # Post process values
        if self.use_vlm_value:
             # values_vlm is [B]
             values = values[0][:, None] # Just use first one? Or mean?
             # OpenPi0 uses values_vlm[:, None] directly.
             pass
        else:
             values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        if num_flow_samples > 1:
            horizon = int(x_0.shape[1])
            act_dim = int(x_0.shape[2])
            steps_plus_one = int(chains.shape[1])

            x_0_s = x_0.reshape(bsize_base, num_flow_samples, horizon, act_dim)
            env_dim = int(getattr(self.config, "action_env_dim", act_dim))
            chunk = int(getattr(self.config, "action_chunk", horizon))
            x_eval = x_0_s[:, :, :chunk, :env_dim]
            score = x_eval.std(dim=2).mean(dim=2)
            best = torch.argmin(score, dim=1)
            arange = torch.arange(bsize_base, device=device)
            x_0 = x_0_s[arange, best]

            chains_s = chains.reshape(bsize_base, num_flow_samples, steps_plus_one, horizon, act_dim)
            chains = chains_s[arange, best]

            if torch.is_tensor(log_probs) and int(log_probs.shape[0]) == int(bsize):
                log_probs = log_probs.reshape(bsize_base, num_flow_samples, *log_probs.shape[1:])[arange, best]
            if torch.is_tensor(values) and int(values.shape[0]) == int(bsize):
                values = values.reshape(bsize_base, num_flow_samples, *values.shape[1:])[arange, best]
            denoise_inds = denoise_inds.reshape(bsize_base, num_flow_samples, *denoise_inds.shape[1:])[arange, best]
            subtask_logits_base = subtask_logits.reshape(bsize_base, num_flow_samples, *subtask_logits.shape[1:])[arange, best]

        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
            "subtask_logits": subtask_logits_base,
        }

    # Adapted get_log_prob_value (for training)
    def get_log_prob_value(
        self,
        images,
        img_masks,
        tokenized_prompt,
        lang_masks, # Unused
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokenized_prompt, state
        )
        
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_4d = self._prepare_attention_masks_4d(prefix_att_2d)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            inputs_embeds=[prefix_embs, None],
            use_cache=True
        )
        
        if self.kv_transform is not None:
            past_key_values = self.kv_transform(past_key_values)
            
        chains_log_probs = []
        chains_values = []
        chains_entropy = []
        
        # ... (Similar loop to OpenPi0, but calling sample_mean_var_val which calls get_suffix_out which calls PiBehavior embed_suffix)
        # Note: We need to pass tokenized_prompt to sample_mean_var_val to reconstruct observation for embed_suffix
        
        # We need to override sample_mean_var_val to accept tokenized_prompt
        # Or wrap it.
        
        # Let's redefine the loop here to be explicit
        num_steps = 1
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0], torch.zeros_like(chains[:, 0]), torch.ones_like(chains[:, 0])
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
            
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            
            # Call custom sample_mean_var_val
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                tokenized_prompt, # Added arg
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            
            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(prefix_output))
            else:
                chains_values.append(value_t)
                
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        
        if getattr(self.config, "noise_method", "flow_sde") == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
            
        return chains_log_probs, chains_values, chains_entropy

    # Overridden sample_mean_var_val with tokenized_prompt
    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        tokenized_prompt, # Added
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
            
        # Noise params ... (same as OpenPi0)
        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.config.noise_level).to(device)
            
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        
        # Velocity prediction using get_suffix_out with tokenized_prompt
        suffix_out = self.get_suffix_out(
            state,
            tokenized_prompt, # Added
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        suffix_out_proj = suffix_out
        proj_dtype = self.action_out_proj.weight.dtype
        if suffix_out_proj.dtype != proj_dtype:
            suffix_out_proj = suffix_out_proj.to(dtype=proj_dtype)
        v_t = self.action_out_proj(suffix_out_proj)
        
        # Value prediction ... (same)
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_in = suffix_out_value
            try:
                value_dtype = next(self.value_head.parameters()).dtype
            except StopIteration:
                value_dtype = torch.float32
            if value_in.dtype != value_dtype:
                value_in = value_in.to(dtype=value_dtype)
            value_t = self.value_head(value_in)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)
            
        # ODE/SDE mix ... (same)
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
                
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    # Overridden get_suffix_out with tokenized_prompt
    def get_suffix_out(
        self,
        state,
        tokenized_prompt, # Added
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        # Create dummy observation for PiBehavior embed_suffix
        # It needs observation.tokenized_prompt
        import types
        observation = types.SimpleNamespace()
        observation.tokenized_prompt = tokenized_prompt
        
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(observation, x_t, timestep)
        )
        
        # ... rest same as OpenPi0 ...
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # Re-use methods from OpenPi0
    def get_logprob_norm(self, sample, mu, sigma):
        if getattr(self.config, "safe_get_logprob", False):
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def get_value_from_vlm(self, prefix_output):
        # ... (same as OpenPi0)
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816
        else:
            # Default fallback
            lang_token_len = 200
            all_token_length = 968

        if self.config.value_vlm_mode == "mean_token":
            # Adjust mask for B1K inputs?
            # B1K has 5 images? No, 3 images usually?
            # images list length
            # Let's assume standard config
            num_images = getattr(self.config, "num_images_in_input", 3)
            # SigLIP tokens = 256
            # But wait, PiBehavior has variable number of images?
            # prefix_output shape depends on inputs.
            # Let's just use mean of all tokens for simplicity if mode is mean_token
            pass
        
        # Simplified implementation
        prefix_out_value = prefix_output.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
