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

import json
import os
import sys
from typing import Callable

import cv2
import gymnasium as gym
import numpy as np
import torch
from av.container import Container
from av.stream import Stream
from omegaconf import OmegaConf, open_dict
from omnigibson.envs import VectorEnvironment
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_INDICES_TO_NAMES,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
from omnigibson.macros import gm

from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger

# Make sure object states are enabled
gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

__all__ = ["BehaviorEnv"]

_OMNI_STREAM_FILTER_INSTALLED = False


class _LineFilterStream:
    def __init__(self, wrapped, should_drop: Callable[[str], bool]):
        self._wrapped = wrapped
        self._should_drop = should_drop
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        if isinstance(s, bytes):
            s = s.decode(errors="replace")
        elif not isinstance(s, str):
            s = str(s)
        self._buffer += s
        parts = self._buffer.splitlines(keepends=True)
        if parts and not (parts[-1].endswith("\n") or parts[-1].endswith("\r")):
            self._buffer = parts.pop()
        else:
            self._buffer = ""
        written = 0
        for part in parts:
            if self._should_drop(part):
                written += len(part)
                continue
            res = self._wrapped.write(part)
            written += len(part) if res is None else int(res)
        return written

    def flush(self) -> None:
        if self._buffer:
            if not self._should_drop(self._buffer):
                self._wrapped.write(self._buffer)
            self._buffer = ""
        if hasattr(self._wrapped, "flush"):
            self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _install_omniverse_warning_filter() -> None:
    global _OMNI_STREAM_FILTER_INSTALLED
    if _OMNI_STREAM_FILTER_INSTALLED:
        return

    def should_drop(line: str) -> bool:
        return (
            "omni.fabric.plugin" in line
            and "removePath called on non-existent path" in line
            and "HydraTextures/Replicator/PostRender/SDGPipeline" in line
        )

    sys.stdout = _LineFilterStream(sys.stdout, should_drop)
    sys.stderr = _LineFilterStream(sys.stderr, should_drop)
    _OMNI_STREAM_FILTER_INSTALLED = True


class BehaviorEnv(gym.Env):
    def __init__(
        self, cfg, num_envs, seed_offset, total_num_processes, record_metrics=True
    ):
        _install_omniverse_warning_filter()
        self.cfg = cfg

        self.num_envs = num_envs
        self.ignore_terminations = cfg.ignore_terminations
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.seed = self.cfg.seed + seed_offset
        self.record_metrics = record_metrics
        self._is_start = True

        self.logger = get_logger()

        self.auto_reset = cfg.auto_reset
        self._debug_trunc_remaining = 0
        if str(os.environ.get("RLINF_DEBUG_BEHAVIOR_TRUNC", "")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            raw_n = os.environ.get("RLINF_DEBUG_BEHAVIOR_TRUNC_N", "10")
            try:
                self._debug_trunc_remaining = int(raw_n)
            except Exception:
                self._debug_trunc_remaining = 10
            self.logger.info(
                "RLINF_DEBUG_BEHAVIOR_TRUNC is enabled "
                f"value={os.environ.get('RLINF_DEBUG_BEHAVIOR_TRUNC')} "
                f"n={self._debug_trunc_remaining}"
            )
        if self.record_metrics:
            self._init_metrics()

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0

        self._init_env()

        # manually reset environment episode number
        self._video_writer = None
        self.video_cnt = 0
        self._video_fps = self._infer_video_fps()
        if self.cfg.video_cfg.save_video:
            os.makedirs(str(self.cfg.video_cfg.video_base_dir), exist_ok=True)
            self._ensure_video_writer()

    def _infer_video_fps(self) -> float:
        cfg_fps = None
        try:
            cfg_fps = self.cfg.video_cfg.get("fps", None)
        except Exception:
            cfg_fps = None
        if cfg_fps is not None:
            try:
                return float(cfg_fps)
            except Exception:
                pass

        env0 = None
        try:
            envs = getattr(self.env, "envs", None)
            if isinstance(envs, (list, tuple)) and len(envs) > 0:
                env0 = envs[0]
        except Exception:
            env0 = None

        def get_attr(obj, name):
            if obj is None:
                return None
            try:
                return getattr(obj, name)
            except Exception:
                return None

        for name in (
            "control_freq",
            "control_frequency",
            "action_frequency",
            "action_freq",
        ):
            v = get_attr(env0, name)
            if v is not None:
                try:
                    v = float(v)
                    if v > 0:
                        return v
                except Exception:
                    pass

        for name in (
            "control_timestep",
            "action_timestep",
            "dt",
            "physics_dt",
            "sim_dt",
        ):
            v = get_attr(env0, name)
            if v is not None:
                try:
                    v = float(v)
                    if v > 0:
                        return 1.0 / v
                except Exception:
                    pass

        sim = get_attr(env0, "sim")
        for name in ("dt", "physics_dt", "render_dt"):
            v = get_attr(sim, name)
            if v is not None:
                try:
                    v = float(v)
                    if v > 0:
                        return 1.0 / v
                except Exception:
                    pass

        return 10.0

    def _ensure_video_writer(self) -> None:
        if self._video_writer is not None:
            return
        base_dir = str(self.cfg.video_cfg.video_base_dir)
        output_dir = os.path.join(base_dir, f"seed_{self.seed}")
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, f"{self.video_cnt}.mp4")
        fps = getattr(self, "_video_fps", 10.0)
        try:
            self.video_writer = create_video_writer(
                fpath=video_path, resolution=(448, 672), fps=fps
            )
        except TypeError:
            self.video_writer = create_video_writer(
                fpath=video_path, resolution=(448, 672)
            )

    def _load_tasks_cfg(self):
        with open_dict(self.cfg):
            self.cfg.omnigibson_cfg["task"]["activity_name"] = TASK_INDICES_TO_NAMES[
                self.cfg.task_idx
            ]

        # Read task description
        task_description_path = os.path.join(
            os.path.dirname(__file__), "behavior_task.jsonl"
        )
        with open(task_description_path, "r") as f:
            text = f.read()
            task_description = [json.loads(x) for x in text.strip().split("\n") if x]
        task_description_map = {
            task_description[i]["task_name"]: task_description[i]["task"]
            for i in range(len(task_description))
        }
        self.task_description = task_description_map[
            self.cfg.omnigibson_cfg["task"]["activity_name"]
        ]

    def _init_env(self):
        self._load_tasks_cfg()

        self.env = VectorEnvironment(
            self.num_envs,
            OmegaConf.to_container(self.cfg.omnigibson_cfg, resolve=True),
        )

    # def _permute_and_norm(self, x):
    #     return x.to(torch.uint8)[..., :3].permute(2, 0, 1) / 255.0

    def _permute_and_norm(self, x):
        return x.to(torch.uint8)[..., :3].permute(2, 0, 1) / 255.0

    def _extract_state_from_proprio(self, proprio_data):
        if not torch.is_tensor(proprio_data):
            proprio_data = torch.as_tensor(proprio_data, device=self.device)
        proprio_data = proprio_data.to(torch.float32)

        indices = PROPRIOCEPTION_INDICES["R1Pro"]
        base_qvel = proprio_data[..., indices["base_qvel"]]
        trunk_qpos = proprio_data[..., indices["trunk_qpos"]]
        arm_left_qpos = proprio_data[..., indices["arm_left_qpos"]]
        arm_right_qpos = proprio_data[..., indices["arm_right_qpos"]]

        left_gripper_raw = proprio_data[..., indices["gripper_left_qpos"]].sum(
            dim=-1, keepdim=True
        )
        right_gripper_raw = proprio_data[..., indices["gripper_right_qpos"]].sum(
            dim=-1, keepdim=True
        )

        max_gripper_width = 0.1
        left_gripper_width = 2.0 * (left_gripper_raw / max_gripper_width) - 1.0
        right_gripper_width = 2.0 * (right_gripper_raw / max_gripper_width) - 1.0

        return torch.cat(
            [
                base_qvel,
                trunk_qpos,
                arm_left_qpos,
                left_gripper_width,
                arm_right_qpos,
                right_gripper_width,
            ],
            dim=-1,
        )

    def _required_proprio_dim(self) -> int:
        indices = PROPRIOCEPTION_INDICES["R1Pro"]
        max_index = -1
        for v in indices.values():
            if isinstance(v, slice):
                stop = 0 if v.stop is None else int(v.stop)
                max_index = max(max_index, stop - 1)
            else:
                arr = np.asarray(v)
                if arr.size:
                    max_index = max(max_index, int(arr.max()))
        return max_index + 1

    def _summarize_raw_obs(self, raw_obs, limit: int = 200) -> list[str]:
        out = []

        def _shape_str(x):
            if torch.is_tensor(x):
                return f"torch{tuple(x.shape)}"
            if isinstance(x, np.ndarray):
                return f"np{tuple(x.shape)}"
            return type(x).__name__

        def _walk(obj, prefix: str):
            if len(out) >= limit:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _walk(v, f"{prefix}/{k}" if prefix else str(k))
                return
            out.append(f"{prefix}: {_shape_str(obj)}")

        _walk(raw_obs, "")
        return out

    def _extract_obs_image(self, raw_obs):
        permute_and_norm = self._permute_and_norm
        left_image = None
        right_image = None
        zed_image = None
        proprio_data = None
        required_dim = self._required_proprio_dim()
        for sensor_data in raw_obs.values():
            assert isinstance(sensor_data, dict)
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_image = permute_and_norm(v["rgb"])
                elif "right_realsense_link:Camera:0" in k:
                    right_image = permute_and_norm(v["rgb"])
                elif "zed_link:Camera:0" in k:
                    zed_image = permute_and_norm(v["rgb"])
                else:
                    key_l = str(k).lower()
                    if "proprio" in key_l or "proprioception" in key_l:
                        if isinstance(v, dict):
                            if "proprio" in v:
                                proprio_data = v["proprio"]
                            elif "data" in v:
                                proprio_data = v["data"]
                        else:
                            proprio_data = v
                    if proprio_data is not None:
                        try:
                            cand = (
                                proprio_data
                                if torch.is_tensor(proprio_data)
                                else np.asarray(proprio_data)
                            )
                            last_dim = int(cand.shape[-1]) if hasattr(cand, "shape") else -1
                            if last_dim < required_dim:
                                proprio_data = None
                        except Exception:
                            proprio_data = None

        if zed_image is None or left_image is None or right_image is None:
            if not hasattr(self, "_debug_saved_obs_schema"):
                self._debug_saved_obs_schema = False
            if not self._debug_saved_obs_schema:
                try:
                    schema = self._summarize_raw_obs(raw_obs)
                    self.logger.warning(
                        "Missing expected camera observations in BEHAVIOR env output. "
                        + " | ".join(schema)
                    )
                except Exception as e:
                    self.logger.warning(f"failed to summarize BEHAVIOR raw_obs: {e}")
                self._debug_saved_obs_schema = True

        if proprio_data is None:
            if not hasattr(self, "_debug_saved_proprio_schema"):
                self._debug_saved_proprio_schema = False
            if not self._debug_saved_proprio_schema:
                try:
                    schema = self._summarize_raw_obs(raw_obs)
                    self.logger.warning(
                        "Missing proprio observation in BEHAVIOR env output; using zeros. "
                        + " | ".join(schema)
                    )
                except Exception as e:
                    self.logger.warning(f"failed to summarize BEHAVIOR raw_obs: {e}")
                self._debug_saved_proprio_schema = True
            proprio_data = torch.zeros(required_dim, dtype=torch.float32, device=self.device)

        return {
            "images": zed_image,  # [C, H, W]
            "wrist_images": torch.stack(
                [left_image, right_image], axis=0
            ),  # [N_IMG, C, H, W]
            "states": self._extract_state_from_proprio(proprio_data),
        }

    def _wrap_obs(self, obs_list):
        extracted_obs_list = []
        for obs in obs_list:
            extracted_obs = self._extract_obs_image(obs)
            extracted_obs_list.append(extracted_obs)

        obs = {
            "images": torch.stack(
                [obs["images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, C, H, W]
            "wrist_images": torch.stack(
                [obs["wrist_images"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, N_IMG, C, H, W]
            "states": torch.stack(
                [obs["states"] for obs in extracted_obs_list], axis=0
            ),  # [N_ENV, state_dim]
            "task_descriptions": [self.task_description for i in range(self.num_envs)],
            "task_id": torch.full(
                (self.num_envs,),
                int(self.cfg.task_idx),
                dtype=torch.int64,
                device=self.device,
            ),
        }
        task_ids = obs["task_id"].to(torch.int32)
        subtask_states = torch.zeros_like(task_ids)
        obs["tokenized_prompt"] = torch.stack([task_ids, subtask_states], dim=1)
        obs["tokenized_prompt_mask"] = torch.ones(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        return obs

    def reset(self):
        raw_obs, infos = self.env.reset()
        obs = self._wrap_obs(raw_obs)
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        infos = self._record_metrics(rewards, infos)
        self._reset_metrics()
        return obs, infos

    def step(
        self, actions=None
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        raw_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        if self.cfg.video_cfg.save_video:
            self._write_video(raw_obs)
        obs = self._wrap_obs(raw_obs)
        infos = self._record_metrics(rewards, infos)
        if self.ignore_terminations:
            terminations[:] = False

        return (
            obs,
            to_tensor(rewards),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_rewards, terminations, truncations, infos = self.step(
                actions
            )
            chunk_rewards.append(step_rewards)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)

        past_dones = torch.logical_or(past_terminations, past_truncations)
        if self._debug_trunc_remaining > 0 and past_dones.any():
            try:
                env0_term = raw_chunk_terminations[0].detach().cpu().to(torch.int32)
                env0_trunc = raw_chunk_truncations[0].detach().cpu().to(torch.int32)
                first_term = int(env0_term.nonzero(as_tuple=False)[0].item()) if env0_term.any() else -1
                first_trunc = int(env0_trunc.nonzero(as_tuple=False)[0].item()) if env0_trunc.any() else -1
                self.logger.info(
                    f"behavior_chunk_done env0 first_term={first_term} first_trunc={first_trunc} "
                    f"term_seq={env0_term.tolist()} trunc_seq={env0_trunc.tolist()}"
                )
                if isinstance(infos, dict):
                    self.logger.info(
                        f"behavior_chunk_done infos_keys={sorted(list(infos.keys()))}"
                    )
            except Exception as e:
                self.logger.warning(f"behavior_trunc_debug_failed error={e}")
            self._debug_trunc_remaining -= 1

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, infos
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    @property
    def device(self):
        return "cuda"

    @property
    def elapsed_steps(self):
        return torch.tensor(self.cfg.max_episode_steps)

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def video_writer(self) -> tuple[Container, Stream]:
        """
        Returns the video writer for the current evaluation step.
        """
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()
        self._video_writer = video_writer

    def flush_video(self) -> None:
        """
        Flush the video writer.
        """
        if self.cfg.video_cfg.save_video:
            self.video_writer = None
            self.video_cnt += 1

    def _write_video(self, raw_obs) -> None:
        """
        Write the current robot observations to video.
        """
        self._ensure_video_writer()
        for sensor_data in raw_obs[0].values():
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_wrist_rgb = cv2.resize(v["rgb"].numpy(), (224, 224))
                elif "right_realsense_link:Camera:0" in k:
                    right_wrist_rgb = cv2.resize(v["rgb"].numpy(), (224, 224))
                elif "zed_link:Camera:0" in k:
                    head_rgb = cv2.resize(v["rgb"].numpy(), (448, 448))

        write_video(
            np.expand_dims(
                np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0
            ),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.prev_step_reward = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
        else:
            mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.prev_step_reward[mask] = 0.0
        if self.record_metrics:
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0

    def _record_metrics(self, rewards, infos):
        info_lists = []
        for env_idx, (reward, info) in enumerate(zip(rewards, infos)):
            episode_info = {
                "success": info.get("done", {}).get("success", False),
                "episode_length": info.get("episode_length", 0),
            }
            self.returns[env_idx] += reward
            if "success" in info:
                self.success_once[env_idx] = (
                    self.success_once[env_idx] | info["success"]
                )
                episode_info["success_once"] = self.success_once[env_idx].clone()
            if "fail" in info:
                self.fail_once[env_idx] = self.fail_once[env_idx] | info["fail"]
                episode_info["fail_once"] = self.fail_once[env_idx].clone()
            episode_info["return"] = self.returns[env_idx].clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = (
                episode_info["return"] / episode_info["episode_len"]
            )
            if self.ignore_terminations:
                episode_info["success_at_end"] = info["success"]

            info_lists.append(episode_info)

        infos = {"episode": to_tensor(list_of_dict_to_dict_of_list(info_lists))}
        return infos

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = infos.copy()
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset()
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def update_reset_state_ids(self):
        # use for multi task training
        pass
