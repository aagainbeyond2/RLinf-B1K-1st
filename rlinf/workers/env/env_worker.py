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

from collections import defaultdict
from typing import Any

import numpy as np
import os
import torch
from omegaconf import DictConfig

from rlinf.data.io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0

        self.simulator_list = []
        self.last_obs_list = []
        self.last_dones_list = []
        self.eval_simulator_list = []
        self._debug_chunk_actions_remaining = 0

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        assert (
            self._component_placement.get_world_size("rollout")
            % self._component_placement.get_world_size("env")
            == 0
        )
        # gather_num: number of rollout for each env process
        self.gather_num = self._component_placement.get_world_size(
            "rollout"
        ) // self._component_placement.get_world_size("env")
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        # only need rank0 to create channel
        if self._rank == 0:
            self.channel = self.create_channel(cfg.env.channel.name)
        else:
            self.channel = self.connect_channel(cfg.env.channel.name)

        # Env configurations
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )

    def init_worker(self):
        enable_offload = self.cfg.env.enable_offload
        if str(os.environ.get("RLINF_DEBUG_CHUNK_REWARDS", "")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            self.log_info(
                f"RLINF_DEBUG_CHUNK_REWARDS is enabled value={os.environ.get('RLINF_DEBUG_CHUNK_REWARDS')}"
            )
        if str(os.environ.get("RLINF_DEBUG_CHUNK_ACTIONS", "")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            raw_n = os.environ.get("RLINF_DEBUG_CHUNK_ACTIONS_N", "3")
            try:
                self._debug_chunk_actions_remaining = int(raw_n)
            except Exception:
                self._debug_chunk_actions_remaining = 3
            self.log_info(
                "RLINF_DEBUG_CHUNK_ACTIONS is enabled "
                f"value={os.environ.get('RLINF_DEBUG_CHUNK_ACTIONS')} "
                f"n={self._debug_chunk_actions_remaining}"
            )

        train_env_cls = get_env_cls(
            self.cfg.env.train.simulator_type, self.cfg.env.train
        )
        eval_env_cls = get_env_cls(self.cfg.env.eval.simulator_type, self.cfg.env.eval)

        if not self.only_eval:
            for stage_id in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.cfg.env.train,
                        rank=self._rank,
                        num_envs=self.train_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=train_env_cls,
                        enable_offload=enable_offload,
                    )
                )
        if self.enable_eval:
            for stage_id in range(self.stage_num):
                self.eval_simulator_list.append(
                    EnvManager(
                        self.cfg.env.eval,
                        rank=self._rank,
                        num_envs=self.eval_num_envs_per_stage,
                        seed_offset=self._rank * self.stage_num + stage_id,
                        total_num_processes=self._world_size * self.stage_num,
                        env_cls=eval_env_cls,
                        enable_offload=enable_offload,
                    )
                )

        if not self.only_eval:
            self._init_simulator()

    def _init_simulator(self):
        if self.cfg.env.train.auto_reset:
            for i in range(self.stage_num):
                self.simulator_list[i].start_simulator()
                extracted_obs, _ = self.simulator_list[i].reset()
                dones = (
                    torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                    .unsqueeze(1)
                    .repeat(1, self.cfg.actor.model.num_action_chunks)
                )
                self.last_obs_list.append(extracted_obs)
                self.last_dones_list.append(dones)
                self.simulator_list[i].stop_simulator()

    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any] | None]:
        """
        This function is used to interact with the environment.
        """
        raw_chunk_actions = chunk_actions
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_chunk_actions,
            simulator_type=self.cfg.env.train.simulator_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        if self._debug_chunk_actions_remaining > 0:
            def _to_np(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().numpy()
                return np.asarray(x)

            def _stats(x):
                x_np = _to_np(x)
                try:
                    x_f = x_np.astype(np.float32, copy=False)
                    return (
                        f"shape={tuple(x_np.shape)} dtype={x_np.dtype} "
                        f"min={float(x_f.min()):.6f} max={float(x_f.max()):.6f} "
                        f"mean={float(x_f.mean()):.6f} std={float(x_f.std()):.6f}"
                    )
                except Exception:
                    return f"shape={tuple(x_np.shape)} dtype={x_np.dtype}"

            try:
                self.log_info(
                    f"raw_chunk_actions {_stats(raw_chunk_actions)} pid={os.getpid()}"
                )
                self.log_info(
                    f"prepared_chunk_actions {_stats(chunk_actions)} pid={os.getpid()}"
                )
                ca = _to_np(chunk_actions)
                if ca.ndim == 3 and ca.shape[0] > 0 and ca.shape[1] > 0:
                    preview_dims = min(int(ca.shape[-1]), 12)
                    self.log_info(
                        f"chunk_actions[0,0,:{preview_dims}]={ca[0,0,:preview_dims].tolist()} pid={os.getpid()}"
                    )
                    if ca.shape[1] > 1:
                        self.log_info(
                            f"chunk_actions[0,1,:{preview_dims}]={ca[0,1,:preview_dims].tolist()} pid={os.getpid()}"
                        )
                        self.log_info(
                            f"chunk_actions_delta_l2[0,0->1]={float(np.linalg.norm(ca[0,1]-ca[0,0]))} pid={os.getpid()}"
                        )
                        self.log_info(
                            f"chunk_actions_time_std_mean={float(ca.std(axis=1).mean())} pid={os.getpid()}"
                        )
            except Exception as e:
                self.log_warning(
                    f"chunk_actions debug failed error={e} pid={os.getpid()}"
                )
            self._debug_chunk_actions_remaining -= 1

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.simulator_list[stage_id].chunk_step(chunk_actions)
        )
        if str(os.environ.get("RLINF_DEBUG_CHUNK_REWARDS", "")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            try:
                shape = (
                    tuple(chunk_rewards.shape) if hasattr(chunk_rewards, "shape") else None
                )
                nonzero = (
                    int((chunk_rewards != 0).sum().item())
                    if torch.is_tensor(chunk_rewards)
                    else None
                )
                r_sum = (
                    float(chunk_rewards.sum().item()) if torch.is_tensor(chunk_rewards) else None
                )
                r_absmax = (
                    float(chunk_rewards.abs().max().item())
                    if torch.is_tensor(chunk_rewards) and chunk_rewards.numel() > 0
                    else None
                )
                self.log_info(
                    f"chunk_rewards shape={shape} nonzero={nonzero} sum={r_sum} absmax={r_absmax} pid={os.getpid()}"
                )
                if torch.is_tensor(chunk_terminations) and torch.is_tensor(chunk_truncations):
                    term_last = int(chunk_terminations[:, -1].sum().item())
                    trunc_last = int(chunk_truncations[:, -1].sum().item())
                    self.log_info(
                        f"chunk_dones_last term={term_last} trunc={trunc_last} pid={os.getpid()}"
                    )
            except Exception as e:
                self.log_warning(
                    f"chunk_rewards debug failed error={e} pid={os.getpid()}"
                )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
        )
        episode = infos.get("episode", None) if isinstance(infos, dict) else None
        return env_output, env_info, episode

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any], dict[str, Any] | None]:
        """
        This function is used to evaluate the environment.
        """
        raw_chunk_actions = raw_actions
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            simulator_type=self.cfg.env.train.simulator_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
        )
        env_info = {}

        if self._debug_chunk_actions_remaining > 0:
            def _to_np(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().numpy()
                return np.asarray(x)

            def _stats(x):
                x_np = _to_np(x)
                try:
                    x_f = x_np.astype(np.float32, copy=False)
                    return (
                        f"shape={tuple(x_np.shape)} dtype={x_np.dtype} "
                        f"min={float(x_f.min()):.6f} max={float(x_f.max()):.6f} "
                        f"mean={float(x_f.mean()):.6f} std={float(x_f.std()):.6f}"
                    )
                except Exception:
                    return f"shape={tuple(x_np.shape)} dtype={x_np.dtype}"

            try:
                self.log_info(
                    f"[eval] raw_chunk_actions {_stats(raw_chunk_actions)} pid={os.getpid()}"
                )
                self.log_info(
                    f"[eval] prepared_chunk_actions {_stats(chunk_actions)} pid={os.getpid()}"
                )
                ca = _to_np(chunk_actions)
                if ca.ndim == 3 and ca.shape[0] > 0 and ca.shape[1] > 0:
                    preview_dims = min(int(ca.shape[-1]), 12)
                    self.log_info(
                        f"[eval] chunk_actions[0,0,:{preview_dims}]={ca[0,0,:preview_dims].tolist()} pid={os.getpid()}"
                    )
                    if ca.shape[1] > 1:
                        self.log_info(
                            f"[eval] chunk_actions[0,1,:{preview_dims}]={ca[0,1,:preview_dims].tolist()} pid={os.getpid()}"
                        )
                        self.log_info(
                            f"[eval] chunk_actions_delta_l2[0,0->1]={float(np.linalg.norm(ca[0,1]-ca[0,0]))} pid={os.getpid()}"
                        )
                        self.log_info(
                            f"[eval] chunk_actions_time_std_mean={float(ca.std(axis=1).mean())} pid={os.getpid()}"
                        )
            except Exception as e:
                self.log_warning(
                    f"[eval] chunk_actions debug failed error={e} pid={os.getpid()}"
                )
            self._debug_chunk_actions_remaining -= 1

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
            self.eval_simulator_list[stage_id].chunk_step(chunk_actions)
        )
        if str(os.environ.get("RLINF_DEBUG_CHUNK_REWARDS", "")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            try:
                shape = (
                    tuple(chunk_rewards.shape) if hasattr(chunk_rewards, "shape") else None
                )
                nonzero = (
                    int((chunk_rewards != 0).sum().item())
                    if torch.is_tensor(chunk_rewards)
                    else None
                )
                self.log_info(
                    f"[eval] chunk_rewards shape={shape} nonzero={nonzero} pid={os.getpid()}"
                )
            except Exception as e:
                self.log_warning(
                    f"[eval] chunk_rewards debug failed error={e} pid={os.getpid()}"
                )
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
        )
        episode = infos.get("episode", None) if isinstance(infos, dict) else None
        return env_output, env_info, episode

    def recv_chunk_actions(self):
        chunk_action = []
        for gather_id in range(self.gather_num):
            chunk_action.append(
                self.channel.get(
                    key=f"{self._action_queue_name}_{gather_id + self._rank * self.gather_num}",
                )
            )
        chunk_action = np.concatenate(chunk_action, axis=0)
        return chunk_action

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            if self.cfg.env.train.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.simulator_list[i].flush_video()
            for i in range(self.stage_num):
                self.simulator_list[i].update_reset_state_ids()
        elif mode == "eval":
            if self.cfg.env.eval.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.eval_simulator_list[i].flush_video()
            if not self.cfg.env.eval.auto_reset:
                for i in range(self.stage_num):
                    self.eval_simulator_list[i].update_reset_state_ids()

    def split_env_batch(self, env_batch, gather_id, mode):
        env_batch_i = {}
        for key, value in env_batch.items():
            if isinstance(value, torch.Tensor):
                env_batch_i[key] = value.chunk(self.gather_num, dim=0)[
                    gather_id
                ].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.train_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.train_num_envs_per_stage} "
                        f"(train_num_envs_per_stage), got {length}"
                    )
                elif mode == "eval":
                    assert length == self.eval_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.eval_num_envs_per_stage} "
                        f"(eval_num_envs_per_stage), got {length}"
                    )
                env_batch_i[key] = value[
                    gather_id * length // self.gather_num : (gather_id + 1)
                    * length
                    // self.gather_num
                ]
            elif isinstance(value, dict):
                env_batch_i[key] = self.split_env_batch(value, gather_id, mode)
            else:
                env_batch_i[key] = value
        return env_batch_i

    def send_env_batch(self, env_batch, mode="train"):
        # split env_batch into num_processes chunks, each chunk contains gather_num env_batch
        for gather_id in range(self.gather_num):
            env_batch_i = self.split_env_batch(env_batch, gather_id, mode)
            self.channel.put(
                item=env_batch_i,
                key=f"{self._obs_queue_name}_{gather_id + self._rank * self.gather_num}",
            )

    def interact(self):
        for simulator in self.simulator_list:
            simulator.start_simulator()

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        env_metrics = defaultdict(list)
        for epoch in range(self.cfg.algorithm.rollout_epoch):
            env_output_list = []
            env_batch_list = []
            if not self.cfg.env.train.auto_reset:
                for stage_id in range(self.stage_num):
                    self.simulator_list[stage_id].is_start = True
                    extracted_obs, infos = self.simulator_list[stage_id].reset()
                    dones = (
                        torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                        .unsqueeze(1)
                        .repeat(1, self.cfg.actor.model.num_action_chunks)
                    )
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        dones=dones,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    env_output_list.append(env_output)
                    env_batch = env_output.to_dict()
                    if isinstance(infos, dict) and "episode" in infos:
                        env_batch["episode"] = infos["episode"]
                    env_batch_list.append(env_batch)
            else:
                self.num_done_envs = 0
                self.num_succ_envs = 0
                for stage_id in range(self.stage_num):
                    env_output = EnvOutput(
                        obs=self.last_obs_list[stage_id],
                        rewards=None,
                        dones=self.last_dones_list[stage_id],
                    )
                    env_output_list.append(env_output)
                    env_batch_list.append(env_output.to_dict())

            for stage_id in range(self.stage_num):
                self.send_env_batch(env_batch_list[stage_id])

            for _ in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions()
                    env_output, env_info, episode = self.env_interact_step(
                        raw_chunk_actions, stage_id
                    )
                    env_batch = env_output.to_dict()
                    if episode is not None:
                        env_batch["episode"] = episode
                    self.send_env_batch(env_batch)
                    env_output_list[stage_id] = env_output
                    for key, value in env_info.items():
                        if (
                            not self.cfg.env.train.auto_reset
                            and not self.cfg.env.train.ignore_terminations
                        ):
                            if key in env_metrics and len(env_metrics[key]) > epoch:
                                env_metrics[key][epoch] = value
                            else:
                                env_metrics[key].append(value)
                        else:
                            env_metrics[key].append(value)

            self.last_obs_list = [env_output.obs for env_output in env_output_list]
            self.last_dones_list = [env_output.dones for env_output in env_output_list]
            self.finish_rollout()

        for simulator in self.simulator_list:
            simulator.stop_simulator()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    def evaluate(self):
        eval_metrics = defaultdict(list)

        for stage_id in range(self.stage_num):
            self.eval_simulator_list[stage_id].start_simulator()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            for stage_id in range(self.stage_num):
                self.eval_simulator_list[stage_id].is_start = True
                extracted_obs, infos = self.eval_simulator_list[stage_id].reset()
                env_output = EnvOutput(
                    obs=extracted_obs,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                )
                env_batch = env_output.to_dict()
                if isinstance(infos, dict) and "episode" in infos:
                    env_batch["episode"] = infos["episode"]
                self.send_env_batch(env_batch, mode="eval")

            for eval_step in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions()
                    env_output, env_info, episode = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)
                    if eval_step == n_chunk_steps - 1:
                        continue
                    env_batch = env_output.to_dict()
                    if episode is not None:
                        env_batch["episode"] = episode
                    self.send_env_batch(env_batch, mode="eval")

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            self.eval_simulator_list[stage_id].stop_simulator()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics
