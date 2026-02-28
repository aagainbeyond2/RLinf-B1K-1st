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

import copy
import gc
import os

from collections import deque
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement


_B1K_OPEN_THRESHOLD = 0.90
_B1K_CLOSED_THRESHOLD = -0.98
_B1K_GRIPPER_VARIATION_THRESHOLD = 0.2

_B1K_LEFT_GRIPPER_IDX = 14
_B1K_RIGHT_GRIPPER_IDX = 22

_B1K_ALWAYS_OPEN_LEFT_GRIPPER_TASKS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    42,
    43,
    44,
    45,
    47,
    48,
}
_B1K_ALWAYS_OPEN_RIGHT_GRIPPER_TASKS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    34,
    35,
    36,
    37,
    42,
    43,
    44,
    47,
    48,
    49,
}
_B1K_MIN_STAGE_FOR_CLOSURE: dict[int, dict[str, int]] = {
    0: {"left": 2, "right": 2},
    30: {"right": 6},
    31: {"right": 8},
    32: {"right": 5},
    33: {"right": 11},
    40: {"right": 4},
    41: {"left": 14, "right": 14},
    45: {"right": 10},
    46: {"left": 8, "right": 8},
    49: {"left": 14},
}
_B1K_RIGHT_GRIPPER_ALWAYS_ALLOWED = {38, 39}
_B1K_TASK_NUM_STAGES = (
    5,
    6,
    15,
    15,
    14,
    12,
    9,
    15,
    10,
    15,
    7,
    13,
    10,
    15,
    15,
    15,
    15,
    11,
    13,
    12,
    14,
    15,
    9,
    15,
    15,
    15,
    15,
    15,
    15,
    15,
    11,
    10,
    10,
    13,
    5,
    5,
    14,
    6,
    8,
    10,
    5,
    15,
    8,
    15,
    12,
    11,
    9,
    14,
    15,
    15,
)


def _b1k_task0_stage4_reset_to_stage2(
    task_id: int,
    stage: int,
    state: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    if task_id != 0 or stage < 2:
        return None

    corrected_stage = 2 if stage == 4 else stage

    left_gripper = float(state[_B1K_LEFT_GRIPPER_IDX])
    right_gripper = float(state[_B1K_RIGHT_GRIPPER_IDX])

    left_closed = left_gripper < _B1K_CLOSED_THRESHOLD
    right_closed = right_gripper < _B1K_CLOSED_THRESHOLD
    left_open = left_gripper > _B1K_OPEN_THRESHOLD
    right_open = right_gripper > _B1K_OPEN_THRESHOLD

    left_middle = not (left_open or left_closed)
    right_middle = not (right_open or right_closed)

    corrected_actions = np.tile(state, (actions.shape[0], 1))
    changed = False

    if left_closed and not right_middle:
        corrected_actions[:, _B1K_LEFT_GRIPPER_IDX] = 1.0
        changed = True
    if right_closed and not left_middle:
        corrected_actions[:, _B1K_RIGHT_GRIPPER_IDX] = 1.0
        changed = True

    if not changed:
        if corrected_stage == stage:
            return None
        corrected_actions = actions

    return corrected_actions, corrected_stage


def _b1k_general_gripper_correction(
    task_id: int,
    stage: int,
    state: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    left_gripper = float(state[_B1K_LEFT_GRIPPER_IDX])
    right_gripper = float(state[_B1K_RIGHT_GRIPPER_IDX])

    left_closed = left_gripper < _B1K_CLOSED_THRESHOLD
    right_closed = right_gripper < _B1K_CLOSED_THRESHOLD

    left_needs_opening = False
    right_needs_opening = False

    if left_closed:
        if task_id in _B1K_ALWAYS_OPEN_LEFT_GRIPPER_TASKS:
            left_needs_opening = True
        else:
            min_stage_left = _B1K_MIN_STAGE_FOR_CLOSURE.get(task_id, {}).get("left")
            if min_stage_left is not None and stage < int(min_stage_left):
                left_needs_opening = True

    if right_closed:
        if task_id in _B1K_RIGHT_GRIPPER_ALWAYS_ALLOWED:
            right_needs_opening = False
        elif task_id in _B1K_ALWAYS_OPEN_RIGHT_GRIPPER_TASKS:
            right_needs_opening = True
        else:
            min_stage_right = _B1K_MIN_STAGE_FOR_CLOSURE.get(task_id, {}).get("right")
            if min_stage_right is not None and stage < int(min_stage_right):
                right_needs_opening = True

    if not (left_needs_opening or right_needs_opening):
        return None

    corrected_actions = np.tile(state, (actions.shape[0], 1))
    if left_needs_opening:
        corrected_actions[:, _B1K_LEFT_GRIPPER_IDX] = 1.0
    if right_needs_opening:
        corrected_actions[:, _B1K_RIGHT_GRIPPER_IDX] = 1.0

    return corrected_actions, stage


def _b1k_apply_correction_rules(
    task_id: int,
    stage: int,
    state: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, int]:
    if task_id == 0:
        result = _b1k_task0_stage4_reset_to_stage2(task_id, stage, state, actions)
        if result is not None:
            return result

    result = _b1k_general_gripper_correction(task_id, stage, state, actions)
    if result is not None:
        return result

    return actions, stage


def _b1k_check_gripper_variation(
    actions: np.ndarray, num_actions_to_check: int
) -> tuple[bool, float, float]:
    actions_to_check = actions[:num_actions_to_check]
    left_gripper_actions = actions_to_check[:, _B1K_LEFT_GRIPPER_IDX]
    right_gripper_actions = actions_to_check[:, _B1K_RIGHT_GRIPPER_IDX]
    left_variation = float(np.max(left_gripper_actions) - np.min(left_gripper_actions))
    right_variation = float(
        np.max(right_gripper_actions) - np.min(right_gripper_actions)
    )
    has_high_variation = (
        left_variation > _B1K_GRIPPER_VARIATION_THRESHOLD
        or right_variation > _B1K_GRIPPER_VARIATION_THRESHOLD
    )
    return has_high_variation, left_variation, right_variation


def _b1k_interpolate_actions(actions: np.ndarray, target_steps: int) -> np.ndarray:
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    if actions.ndim == 2:
        actions_b = actions[None, :, :]
    elif actions.ndim == 3:
        actions_b = actions
    else:
        raise ValueError("actions must have shape [T, D] or [B, T, D]")

    bsz, t_steps, act_dim = actions_b.shape
    if target_steps == t_steps:
        out = actions_b
    elif target_steps == 1:
        out = actions_b[:, :1, :]
    else:
        x = np.linspace(0.0, float(t_steps - 1), t_steps, dtype=np.float32)
        x_new = np.linspace(0.0, float(t_steps - 1), target_steps, dtype=np.float32)
        out = np.empty((bsz, target_steps, act_dim), dtype=actions_b.dtype)
        try:
            from scipy.interpolate import interp1d

            for b in range(bsz):
                for d in range(act_dim):
                    f = interp1d(x, actions_b[b, :, d], kind="cubic")
                    out[b, :, d] = f(x_new).astype(actions_b.dtype, copy=False)
        except Exception:
            for b in range(bsz):
                for d in range(act_dim):
                    out[b, :, d] = np.interp(x_new, x, actions_b[b, :, d]).astype(
                        actions_b.dtype, copy=False
                    )

    if actions.ndim == 2:
        return out[0]
    return out


def _slice_env_obs_batch(env_obs: dict[str, object], indices: list[int]) -> dict[str, object]:
    if len(indices) == 0:
        return {}
    out: dict[str, object] = {}
    for k, v in env_obs.items():
        if torch.is_tensor(v):
            out[k] = v[torch.as_tensor(indices, device=v.device, dtype=torch.long)]
        elif isinstance(v, np.ndarray):
            idx_np = np.asarray(indices, dtype=np.int64)
            out[k] = v[idx_np]
        elif isinstance(v, list):
            out[k] = [v[i] for i in indices]
        else:
            out[k] = v
    return out


def _scatter_batched_value(
    out_value: object,
    out_indices: list[int],
    src_value: object,
) -> object:
    if src_value is None:
        return out_value
    if out_value is None:
        return None
    if torch.is_tensor(out_value) and torch.is_tensor(src_value):
        out_value[torch.as_tensor(out_indices, device=out_value.device, dtype=torch.long)] = src_value
        return out_value
    if isinstance(out_value, np.ndarray) and isinstance(src_value, np.ndarray):
        out_value[out_indices] = src_value
        return out_value
    if isinstance(out_value, list) and isinstance(src_value, list):
        for dst_i, src_i in enumerate(out_indices):
            out_value[src_i] = src_value[dst_i]
        return out_value
    return out_value


def _alloc_like_batched(src_value: object, batch_size: int) -> object:
    if src_value is None:
        return None
    if torch.is_tensor(src_value):
        shape = (batch_size, *src_value.shape[1:])
        return torch.empty(shape, device=src_value.device, dtype=src_value.dtype)
    if isinstance(src_value, np.ndarray):
        shape = (batch_size, *src_value.shape[1:])
        return np.empty(shape, dtype=src_value.dtype)
    if isinstance(src_value, list):
        return [None for _ in range(batch_size)]
    return src_value


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        # v3
        self.global_step = 0

        self.cfg = cfg

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name

        self.channel = self.connect_channel(cfg.rollout.channel.name)
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())
        self._b1k_tricks_cfg = None
        self._b1k_stage = None
        self._b1k_task_id = None
        self._b1k_prediction_history = None
        self._b1k_next_initial_actions = None

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            model_config, input_processor = get_vla_model_config_and_processor(
                self.cfg.actor
            )
            self.hf_model.setup_config_and_processor(
                model_config, self.cfg, input_processor
            )

        self.hf_model.eval()

        self.setup_sample_params()
        self._setup_b1k_tricks()
        if self.enable_offload:
            self.offload_model()

    def load_checkpoint(self, load_path):
        model_dict = torch.load(load_path)
        self.hf_model.load_state_dict(model_dict)

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def _setup_b1k_tricks(self):
        self._b1k_tricks_cfg = OmegaConf.select(
            self.cfg, "actor.model.policy_setup.b1k_tricks", default=None
        )
        self._b1k_stage = None
        self._b1k_task_id = None
        self._b1k_prediction_history = None
        self._b1k_next_initial_actions = None
        if (
            self._b1k_tricks_cfg is not None
            and hasattr(self, "hf_model")
            and hasattr(self.hf_model, "config")
        ):
            t = self._b1k_tricks_cfg.get("time_threshold_inpaint", None)
            if t is not None:
                setattr(self.hf_model.config, "time_threshold_inpaint", float(t))
            if hasattr(self.hf_model.config, "num_flow_samples"):
                setattr(self.hf_model.config, "num_flow_samples", 1)

    def _b1k_tricks_enabled(self) -> bool:
        if self._b1k_tricks_cfg is None:
            return False
        enabled = self._b1k_tricks_cfg.get("enabled", False)
        return bool(enabled)

    def _ensure_b1k_state(self, batch_size: int):
        if (
            self._b1k_stage is None
            or self._b1k_task_id is None
            or self._b1k_prediction_history is None
            or self._b1k_next_initial_actions is None
            or len(self._b1k_stage) != batch_size
            or len(self._b1k_task_id) != batch_size
            or len(self._b1k_prediction_history) != batch_size
        ):
            history_len = int(self._b1k_tricks_cfg.get("history_len", 3))
            self._b1k_stage = np.zeros(batch_size, dtype=np.int32)
            self._b1k_task_id = np.full(batch_size, -1, dtype=np.int32)
            self._b1k_prediction_history = [
                deque([], maxlen=history_len) for _ in range(batch_size)
            ]
            self._b1k_next_initial_actions = [None for _ in range(batch_size)]

    def _prepare_b1k_tokenized_prompt(self, env_obs: dict[str, object]):
        if not self._b1k_tricks_enabled():
            return
        task_id_tensor = env_obs.get("task_id", None)
        if task_id_tensor is None or not torch.is_tensor(task_id_tensor):
            return
        batch_size = int(task_id_tensor.shape[0])
        self._ensure_b1k_state(batch_size)
        task_ids = task_id_tensor.to(torch.int32).detach().cpu().numpy()
        changed = task_ids != self._b1k_task_id
        if np.any(changed):
            self._b1k_stage[changed] = 0
            self._b1k_task_id = task_ids.astype(np.int32)
            for i, is_changed in enumerate(changed.tolist()):
                if is_changed:
                    self._b1k_prediction_history[i].clear()
                    if self._b1k_next_initial_actions is not None:
                        self._b1k_next_initial_actions[i] = None
        device = task_id_tensor.device
        stage_tensor = torch.from_numpy(self._b1k_stage).to(
            device=device, dtype=torch.int32
        )
        env_obs["tokenized_prompt"] = torch.stack(
            [task_id_tensor.to(torch.int32), stage_tensor], dim=1
        )
        env_obs["tokenized_prompt_mask"] = torch.ones(
            batch_size, 2, dtype=torch.bool, device=device
        )

    def _update_b1k_stage_from_logits(self, result: dict[str, object]):
        if not self._b1k_tricks_enabled():
            return
        if self._b1k_tricks_cfg is None or self._b1k_stage is None or self._b1k_task_id is None:
            return
        subtask_logits = result.get("subtask_logits", None)
        if subtask_logits is None:
            return
        if torch.is_tensor(subtask_logits):
            logits_np = subtask_logits.detach().cpu().numpy()
        else:
            logits_np = np.asarray(subtask_logits)
        if logits_np.ndim != 2:
            return

        history_len = int(self._b1k_tricks_cfg.get("history_len", 3))
        votes_to_promote = int(self._b1k_tricks_cfg.get("votes_to_promote", 2))
        if history_len <= 0:
            return

        bsz = int(logits_np.shape[0])
        self._ensure_b1k_state(bsz)

        predicted = np.argmax(logits_np, axis=-1).astype(np.int32)
        for i in range(bsz):
            task_id = int(self._b1k_task_id[i])
            if task_id < 0 or task_id >= len(_B1K_TASK_NUM_STAGES):
                continue
            max_stage = int(_B1K_TASK_NUM_STAGES[task_id]) - 1
            if max_stage < 0:
                continue
            pred_stage = int(predicted[i])
            if pred_stage > max_stage:
                pred_stage = max_stage

            self._b1k_prediction_history[i].append(pred_stage)
            if len(self._b1k_prediction_history[i]) != history_len:
                continue

            cur_stage = int(self._b1k_stage[i])
            next_stage = cur_stage + 1
            if next_stage > max_stage:
                self._b1k_prediction_history[i].clear()
                continue

            votes_for_next = sum(
                1 for p in self._b1k_prediction_history[i] if p == next_stage
            )
            votes_to_skip = sum(
                1 for p in self._b1k_prediction_history[i] if p == next_stage + 1
            )
            votes_to_go_back = sum(
                1 for p in self._b1k_prediction_history[i] if p == cur_stage - 1
            )

            if votes_for_next >= votes_to_promote:
                self._b1k_stage[i] = next_stage
                self._b1k_prediction_history[i].clear()
            elif votes_to_skip == history_len:
                self._b1k_stage[i] = next_stage
                self._b1k_prediction_history[i].clear()
            elif votes_to_go_back == history_len and cur_stage > 0:
                self._b1k_stage[i] = cur_stage - 1
                self._b1k_prediction_history[i].clear()

    def _b1k_build_initial_actions(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self._b1k_tricks_enabled():
            return None
        if self._b1k_tricks_cfg is None or self._b1k_next_initial_actions is None:
            return None
        if any(a is None for a in self._b1k_next_initial_actions):
            return None
        initial_np = np.stack(self._b1k_next_initial_actions, axis=0)
        return torch.from_numpy(initial_np).to(device=device, dtype=torch.float32)

    def _b1k_collect_initial_actions(
        self, device: torch.device
    ) -> tuple[Optional[torch.Tensor], list[int]]:
        if not self._b1k_tricks_enabled():
            return None, []
        if self._b1k_tricks_cfg is None or self._b1k_next_initial_actions is None:
            return None, []
        indices = [i for i, a in enumerate(self._b1k_next_initial_actions) if a is not None]
        if len(indices) == 0:
            return None, []
        initial_np = np.stack([self._b1k_next_initial_actions[i] for i in indices], axis=0)
        return torch.from_numpy(initial_np).to(device=device, dtype=torch.float32), indices

    def _apply_b1k_tricks_to_actions(
        self, env_obs: dict[str, object], actions: np.ndarray
    ) -> np.ndarray:
        if not self._b1k_tricks_enabled():
            return actions
        apply_eval_tricks = bool(self._b1k_tricks_cfg.get("apply_eval_tricks", True))

        task_id_tensor = env_obs.get("task_id", None)
        state_tensor = env_obs.get("states", None)
        if (
            task_id_tensor is None
            or state_tensor is None
            or not torch.is_tensor(task_id_tensor)
            or not torch.is_tensor(state_tensor)
        ):
            return actions

        batch_size = int(task_id_tensor.shape[0])
        self._ensure_b1k_state(batch_size)

        task_ids = task_id_tensor.detach().cpu().numpy().astype(np.int32)
        states = state_tensor.detach().cpu().numpy()

        actions_out = np.asarray(actions).copy()
        if apply_eval_tricks:
            for i in range(batch_size):
                task_id = int(task_ids[i])
                if task_id < 0:
                    continue
                stage = int(self._b1k_stage[i])
                corrected_actions, corrected_stage = _b1k_apply_correction_rules(
                    task_id=task_id,
                    stage=stage,
                    state=np.asarray(states[i]),
                    actions=np.asarray(actions_out[i]),
                )
                actions_out[i] = corrected_actions
                self._b1k_stage[i] = int(corrected_stage)

        execute_in_n_steps = int(self._b1k_tricks_cfg.get("execute_in_n_steps", 20))
        actions_to_execute_cfg = int(self._b1k_tricks_cfg.get("actions_to_execute", 26))
        actions_to_keep = int(self._b1k_tricks_cfg.get("actions_to_keep", 4))
        if (
            execute_in_n_steps <= 0
            or actions_out.ndim != 3
            or actions_out.shape[1] < actions_to_execute_cfg
        ):
            return actions_out

        compress_enabled = bool(self._b1k_tricks_cfg.get("compress", True))
        base_should_compress = bool(compress_enabled and execute_in_n_steps < actions_to_execute_cfg)

        actions_exec = np.empty(
            (batch_size, execute_in_n_steps, actions_out.shape[2]), dtype=actions_out.dtype
        )
        for i in range(batch_size):
            should_compress = base_should_compress
            if apply_eval_tricks and should_compress:
                has_high_var, _, _ = _b1k_check_gripper_variation(
                    actions_out[i], actions_to_execute_cfg
                )
                if has_high_var:
                    should_compress = False

            actions_to_execute = (
                actions_to_execute_cfg if should_compress else execute_in_n_steps
            )

            if (
                actions_to_keep > 0
                and self._b1k_next_initial_actions is not None
                and len(self._b1k_next_initial_actions) == batch_size
            ):
                inpaint_end = actions_to_execute + actions_to_keep
                if actions_out.shape[1] >= inpaint_end:
                    self._b1k_next_initial_actions[i] = np.asarray(
                        actions_out[i, actions_to_execute:inpaint_end, :]
                    ).copy()
                else:
                    self._b1k_next_initial_actions[i] = None

            src = actions_out[i, :actions_to_execute, :]
            if should_compress:
                compressed = _b1k_interpolate_actions(src, execute_in_n_steps)
                compression_factor = float(actions_to_execute) / float(execute_in_n_steps)
                compressed[:, :3] *= compression_factor
                actions_exec[i] = compressed
            else:
                actions_exec[i] = src[:execute_in_n_steps]

        return actions_exec

    def predict(self, env_obs, mode="train"):
        if not hasattr(self, "_debug_predict_trace_remaining"):
            enabled = str(os.environ.get("RLINF_DEBUG_PREDICT_TRACE", "")).lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            raw_n = os.environ.get("RLINF_DEBUG_PREDICT_TRACE_N", "1")
            try:
                n = int(raw_n)
            except Exception:
                n = 1
            self._debug_predict_trace_remaining = max(0, n) if enabled else 0

        def _to_np(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            if isinstance(x, np.ndarray):
                return x
            return np.asarray(x)

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
                    f"mean={float(x_f.mean()):.6f} std={float(x_f.std(unbiased=False)):.6f}"
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
                    arr = _to_np(x)
                    shape = tuple(arr.shape)
                    dtype = arr.dtype
                    if getattr(arr, "size", 0) == 0:
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

        def _summarize(obj, prefix, lines, limit):
            if len(lines) >= limit:
                return
            if isinstance(obj, dict):
                for k in sorted(obj.keys(), key=lambda z: str(z)):
                    _summarize(
                        obj[k],
                        f"{prefix}.{k}" if prefix else str(k),
                        lines,
                        limit,
                    )
                return
            if isinstance(obj, (list, tuple)):
                lines.append(f"{prefix} len={len(obj)} type={type(obj).__name__}")
                for i in range(len(obj)):
                    _summarize(obj[i], f"{prefix}[{i}]", lines, limit)
                return
            lines.append(f"{prefix} {_stats(obj)}")

        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
        ]:
            kwargs = {"mode": mode}

        if isinstance(env_obs, dict):
            if self._debug_predict_trace_remaining > 0:
                lines = []
                lines.append(
                    f"[predict.trace] enter mode={mode} pid={os.getpid()} rank={getattr(self, '_rank', None)} "
                    f"hf_model={type(self.hf_model).__name__}"
                )
                _summarize(env_obs, "env_obs", lines, 400)
                print("\n".join(lines), flush=True)

            self._prepare_b1k_tokenized_prompt(env_obs)

            if self._debug_predict_trace_remaining > 0:
                lines = []
                lines.append(
                    "[predict.trace] after _prepare_b1k_tokenized_prompt"
                )
                for k in ("tokenized_prompt", "tokenized_prompt_mask", "task_id"):
                    if k in env_obs:
                        lines.append(f"env_obs.{k} {_stats(env_obs[k])}")
                print("\n".join(lines), flush=True)

        with torch.no_grad():
            if isinstance(env_obs, dict) and self._b1k_tricks_enabled():
                if self._debug_predict_trace_remaining > 0:
                    print("[predict.trace] branch: b1k_tricks_enabled", flush=True)
                device = next(self.hf_model.parameters()).device
                initial_actions, indices = self._b1k_collect_initial_actions(device=device)
                if self._debug_predict_trace_remaining > 0:
                    print(
                        f"[predict.trace] _b1k_collect_initial_actions "
                        f"initial_actions={_stats(initial_actions) if initial_actions is not None else 'None'} "
                        f"indices_len={len(indices)}",
                        flush=True,
                    )
                if initial_actions is not None and len(indices) > 0:
                    batch_size = int(env_obs["task_id"].shape[0]) if torch.is_tensor(env_obs.get("task_id", None)) else int(initial_actions.shape[0])
                    if len(indices) == batch_size:
                        kwargs = dict(kwargs)
                        kwargs["initial_actions"] = initial_actions
                        if self._debug_predict_trace_remaining > 0:
                            print(
                                f"[predict.trace] calling hf_model.predict_action_batch(all) "
                                f"env_obs.task_id={_stats(env_obs.get('task_id', None))} "
                                f"initial_actions={_stats(initial_actions)}",
                                flush=True,
                            )
                        actions, result = self.hf_model.predict_action_batch(
                            env_obs=env_obs,
                            **kwargs,
                        )
                    else:
                        indices_set = set(indices)
                        indices_no = [i for i in range(batch_size) if i not in indices_set]

                        env_obs_yes = _slice_env_obs_batch(env_obs, indices)
                        env_obs_no = _slice_env_obs_batch(env_obs, indices_no)

                        kwargs_yes = dict(kwargs)
                        kwargs_yes["initial_actions"] = initial_actions

                        if self._debug_predict_trace_remaining > 0:
                            print(
                                f"[predict.trace] calling hf_model.predict_action_batch(split) "
                                f"yes={len(indices)} no={len(indices_no)} "
                                f"initial_actions={_stats(initial_actions)}",
                                flush=True,
                            )
                        actions_yes, result_yes = self.hf_model.predict_action_batch(
                            env_obs=env_obs_yes, **kwargs_yes
                        )
                        actions_no, result_no = self.hf_model.predict_action_batch(
                            env_obs=env_obs_no, **kwargs
                        )

                        actions_out = np.empty(
                            (batch_size, *actions_yes.shape[1:]), dtype=actions_yes.dtype
                        )
                        actions_out[indices] = actions_yes
                        actions_out[indices_no] = actions_no

                        result_out: dict[str, object] = {}
                        for key in set(result_yes.keys()) | set(result_no.keys()):
                            if key == "forward_inputs":
                                fi_yes = result_yes.get("forward_inputs", None)
                                fi_no = result_no.get("forward_inputs", None)
                                fi_out: dict[str, object] = {}
                                if isinstance(fi_yes, dict):
                                    for k, v in fi_yes.items():
                                        fi_out[k] = _alloc_like_batched(v, batch_size)
                                    for k, v in fi_yes.items():
                                        fi_out[k] = _scatter_batched_value(fi_out[k], indices, v)
                                if isinstance(fi_no, dict):
                                    for k, v in fi_no.items():
                                        if k not in fi_out:
                                            fi_out[k] = _alloc_like_batched(v, batch_size)
                                        fi_out[k] = _scatter_batched_value(fi_out[k], indices_no, v)
                                result_out[key] = fi_out
                                continue

                            v_yes = result_yes.get(key, None)
                            v_no = result_no.get(key, None)
                            base = v_yes if v_yes is not None else v_no
                            out_v = _alloc_like_batched(base, batch_size)
                            out_v = _scatter_batched_value(out_v, indices, v_yes)
                            out_v = _scatter_batched_value(out_v, indices_no, v_no)
                            result_out[key] = out_v

                        actions, result = actions_out, result_out
                else:
                    if self._debug_predict_trace_remaining > 0:
                        print(
                            "[predict.trace] calling hf_model.predict_action_batch(no_initial_actions)",
                            flush=True,
                        )
                    actions, result = self.hf_model.predict_action_batch(
                        env_obs=env_obs,
                        **kwargs,
                    )
            else:
                if self._debug_predict_trace_remaining > 0:
                    print(
                        f"[predict.trace] calling hf_model.predict_action_batch(baseline) "
                        f"env_obs_type={type(env_obs).__name__}",
                        flush=True,
                    )
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )

        if self._debug_predict_trace_remaining > 0:
            lines = []
            lines.append("[predict.trace] after hf_model.predict_action_batch")
            lines.append(f"actions {_stats(actions)}")
            if isinstance(result, dict):
                lines.append(f"result.keys={sorted(list(result.keys()))}")
                for k in ("prev_logprobs", "prev_values", "subtask_logits"):
                    if k in result:
                        lines.append(f"result.{k} {_stats(result[k])}")
                fi = result.get("forward_inputs", None)
                if isinstance(fi, dict):
                    for k in sorted(fi.keys()):
                        v = fi[k]
                        if torch.is_tensor(v) or isinstance(v, np.ndarray) or hasattr(v, "shape"):
                            lines.append(f"result.forward_inputs.{k} {_stats(v)}")
            print("\n".join(lines), flush=True)

        if isinstance(env_obs, dict) and isinstance(actions, np.ndarray):
            if isinstance(result, dict):
                self._update_b1k_stage_from_logits(result)
            if self._debug_predict_trace_remaining > 0:
                before = np.asarray(actions)
                actions = self._apply_b1k_tricks_to_actions(env_obs, actions)
                after = np.asarray(actions)
                try:
                    delta = float(np.linalg.norm(after - before))
                except Exception:
                    delta = float("nan")
                print(
                    f"[predict.trace] after _apply_b1k_tricks_to_actions "
                    f"actions { _stats(after) } delta_l2={delta}",
                    flush=True,
                )
            else:
                actions = self._apply_b1k_tricks_to_actions(env_obs, actions)

        if self._debug_predict_trace_remaining > 0:
            self._debug_predict_trace_remaining -= 1

        return actions, result

    def get_dones_and_rewards(
        self, env_output: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards) tensors.
        """
        # First step: no rewards yet, only dones
        if env_output["rewards"] is None:
            return env_output["dones"].bool().cpu().contiguous(), None

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    actions, result = self.predict(final_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                # Add bootstrap value to the last step of done episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        return dones, rewards

    def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = self.recv(self.actor_group_name, src_rank=self._rank)

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self):
        if self.enable_offload:
            self.reload_model()

        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        debug_enabled = str(
            os.environ.get("RLINF_DEBUG_ROLLOUT_STEP", "")
        ).lower() in ("1", "true", "yes", "on")

        def _to_np(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            if isinstance(x, np.ndarray):
                return x
            return np.asarray(x)

        def _stats(x):
            if torch.is_tensor(x):
                x_np = _to_np(x)
                device = x.device
                if x_np.size == 0:
                    return f"shape={tuple(x_np.shape)} dtype={x_np.dtype} device={device}"
                x_f = x_np.astype(np.float32, copy=False)
                return (
                    f"shape={tuple(x_np.shape)} dtype={x_np.dtype} device={device} "
                    f"min={float(x_f.min()):.6f} max={float(x_f.max()):.6f} "
                    f"mean={float(x_f.mean()):.6f} std={float(x_f.std()):.6f}"
                )
            if isinstance(x, np.ndarray):
                if x.size == 0:
                    return f"shape={tuple(x.shape)} dtype={x.dtype}"
                x_f = x.astype(np.float32, copy=False)
                return (
                    f"shape={tuple(x.shape)} dtype={x.dtype} "
                    f"min={float(x_f.min()):.6f} max={float(x_f.max()):.6f} "
                    f"mean={float(x_f.mean()):.6f} std={float(x_f.std()):.6f}"
                )
            return f"type={type(x).__name__} value={x}"

        def _summarize(obj, prefix, lines, limit):
            if len(lines) >= limit:
                return
            if isinstance(obj, dict):
                for k in sorted(obj.keys()):
                    _summarize(obj[k], f"{prefix}.{k}" if prefix else str(k), lines, limit)
                return
            if isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    lines.append(f"{prefix} len=0")
                else:
                    head = obj[0]
                    if isinstance(head, str):
                        lines.append(f"{prefix} len={len(obj)} head={head}")
                    else:
                        lines.append(f"{prefix} len={len(obj)} head_type={type(head).__name__}")
                return
            lines.append(f"{prefix} {_stats(obj)}")

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = self.recv_env_output()

                    dones, rewards = self.get_dones_and_rewards(env_output)
                    actions, result = self.predict(env_output["obs"])
                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        forward_inputs=result["forward_inputs"],
                        episode=env_output.get("episode", None),
                    )
                    if debug_enabled and not getattr(self, "_debug_rollout_step_done", False):
                        self._debug_rollout_step_done = True
                        lines = []
                        lines.append("RLINF_DEBUG_ROLLOUT_STEP enabled")
                        lines.append(
                            f"n_chunk_steps={n_chunk_steps} "
                            f"(max_steps_per_rollout_epoch={self.cfg.env.train.max_steps_per_rollout_epoch}, "
                            f"num_action_chunks={self.cfg.actor.model.num_action_chunks})"
                        )
                        lines.append(
                            f"num_pipeline_stages={self.num_pipeline_stages} "
                            f"(cfg.rollout.pipeline_stage_num)"
                        )
                        lines.append(
                            f"stage_id={stage_id} (pipeline stage index in [0, {self.num_pipeline_stages - 1}])"
                        )
                        _summarize(env_output, "env_output", lines, 200)
                        if isinstance(result, dict):
                            for k in sorted(result.keys()):
                                _summarize(result[k], f"result.{k}", lines, 200)
                        lines.append(f"actions {_stats(actions)}")
                        lines.append(
                            f"ChunkStepResult prev_logprobs={_stats(chunk_step_result.prev_logprobs)} "
                            f"prev_values={_stats(chunk_step_result.prev_values)} "
                            f"dones={_stats(chunk_step_result.dones)} "
                            f"rewards={_stats(chunk_step_result.rewards)}"
                        )
                        lines.append(
                            f"buffer_list size={len(self.buffer_list)} "
                            f"buffer_list[{stage_id}]={type(self.buffer_list[stage_id]).__name__}"
                        )
                        if self._rank == 0:
                            print("\n".join(lines), flush=True)
                        os._exit(0)
                    self.buffer_list[stage_id].append_result(chunk_step_result)

                    self.send_chunk_actions(actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = self.recv_env_output()

                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards = self.get_dones_and_rewards(env_output)
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].rewards.append(rewards)

                with self.worker_timer():
                    actions, result = self.predict(env_output["obs"])
                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )

        for i in range(self.num_pipeline_stages):
            self.send_rollout_batch(i)

        if self.enable_offload:
            self.offload_model()

    def evaluate(self):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = self.recv_env_output()
                    actions, _ = self.predict(env_output["obs"], mode="eval")
                    self.send_chunk_actions(actions)

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    def recv_env_output(self):
        env_output = self.channel.get(
            key=f"{self._obs_queue_name}_{self._rank}",
        )
        return env_output

    def send_chunk_actions(self, chunk_actions):
        self.channel.put(
            item=chunk_actions,
            key=f"{self._action_queue_name}_{self._rank}",
        )

    # v1
    # def send_rollout_batch(self, stage_id):
    #     # send rollout_batch to actor
    #     send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
    #     recv_num = self.placement.get_world_size("actor")
    #     split_num = compute_split_num(recv_num, send_num)
    #     splited_rollout_result = self.buffer_list[stage_id].to_splited_dict(split_num)
    #     for i in range(split_num):
    #         self.channel.put(
    #             item=splited_rollout_result[i],
    #             key=self._replay_buffer_name,
    #         )


    # v2
    # def send_rollout_batch(self, stage_id):
    #     import os
    #
    #     send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
    #     recv_num = self.placement.get_world_size("actor")
    #     split_num = compute_split_num(recv_num, send_num)
    #     if os.environ.get("RLINF_DEBUG_SHAPES", "0") == "1":
    #         msg = (
    #             f"[RLinf][debug_shapes] send_rollout_batch stage_id={stage_id} "
    #             f"send_num={send_num} recv_num={recv_num} split_num={split_num}"
    #         )
    #         log_info = getattr(self, "log_info", None)
    #         if callable(log_info):
    #             log_info(msg)
    #         else:
    #             print(msg, flush=True)
    #     splited_rollout_result = self.buffer_list[stage_id].to_splited_dict(split_num)
    #     for i in range(split_num):
    #         self.channel.put(
    #             item=splited_rollout_result[i],
    #             key=self._replay_buffer_name,
    #         )
    #
    # def set_global_step(self, global_step):
    #     if hasattr(self.hf_model, "set_global_step"):
    #         self.hf_model.set_global_step(global_step)


    # v3
    def send_rollout_batch(self, stage_id):
        import os

        save_cfg = self.cfg.rollout.get("save_rollout", None)
        if save_cfg is not None and save_cfg.get("enabled", False):
            every_n_steps = int(save_cfg.get("every_n_steps", 1))
            should_save = every_n_steps <= 0 or (self.global_step % every_n_steps == 0)
            if should_save:
                base_dir = save_cfg.get("dir", None)
                if base_dir is None:
                    logger_cfg = self.cfg.runner.logger
                    base_dir = os.path.join(
                        logger_cfg.get("log_path", "logs"),
                        logger_cfg.get("experiment_name", "default"),
                        "rollouts",
                    )
                out_dir = os.path.join(base_dir, f"global_step_{self.global_step}")
                os.makedirs(out_dir, exist_ok=True)
                payload = {
                    "global_step": int(self.global_step),
                    "rank": int(self._rank),
                    "stage_id": int(stage_id),
                    "data": self.buffer_list[stage_id].to_dict(),
                }
                episode = self.buffer_list[stage_id].episode_to_dict()
                if episode is not None:
                    payload["episode"] = episode
                torch.save(
                    payload,
                    os.path.join(
                        out_dir, f"rollout_rank{self._rank}_stage{stage_id}.pt"
                    ),
                )
        if self.cfg.rollout.get("save_rollout_only", False):
            return
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        if os.environ.get("RLINF_DEBUG_SHAPES", "0") == "1":
            msg = (
                f"[RLinf][debug_shapes] send_rollout_batch stage_id={stage_id} "
                f"send_num={send_num} recv_num={recv_num} split_num={split_num}"
            )
            log_info = getattr(self, "log_info", None)
            if callable(log_info):
                log_info(msg)
            else:
                print(msg, flush=True)
        splited_rollout_result = self.buffer_list[stage_id].to_splited_dict(split_num)
        for i in range(split_num):
            self.channel.put(
                item=splited_rollout_result[i],
                key=self._replay_buffer_name,
            )

    def set_global_step(self, global_step):
        self.global_step = int(global_step)
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
