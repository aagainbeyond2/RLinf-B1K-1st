import glob
import math
import os
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement


class OfflineRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg

        self.global_step = 0
        self.num_pipeline_stages = int(cfg.rollout.pipeline_stage_num)

        self._replay_buffer_name = cfg.actor.channel.queue_name
        if self._rank == 0:
            self.channel = self.create_channel(cfg.rollout.channel.name)
        else:
            self.channel = self.connect_channel(cfg.rollout.channel.name)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        self._base_dir = cfg.rollout.offline.get("dir", None)
        if self._base_dir is None:
            raise ValueError("rollout.offline.dir must be set for offline rollout backend")
        self._base_dir = str(self._base_dir)

        self._loop = bool(cfg.rollout.offline.get("loop", True))
        self._strict = bool(cfg.rollout.offline.get("strict", True))

        self._all_steps: list[int] = []
        self._step_to_files: dict[int, dict[int, str]] = {}

    def init_worker(self):
        self._index_rollouts()

    def set_global_step(self, global_step):
        self.global_step = int(global_step)

    def sync_model_from_actor(self):
        return None

    def evaluate(self):
        return None

    def _torch_load_safely(self, path: str) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _resolve_data(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
            return payload["data"]
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Unexpected payload type: {type(payload)}")

    def _split_rollout_dict(self, base_dict: dict[str, Any], split_size: int) -> list[dict[str, Any]]:
        if split_size <= 0:
            raise ValueError(f"split_size must be > 0, got {split_size}")
        result_list: list[dict[str, Any]] = []
        for i in range(split_size):
            split_dict: dict[str, Any] = {}
            for k, v in base_dict.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() < 2:
                        split_dict[k] = v
                        continue
                    if v.size(1) < split_size:
                        target_batch = int(math.ceil(v.size(1) / split_size) * split_size)
                        repeat = int(math.ceil(target_batch / v.size(1)))
                        repeat_dims = [1] * v.dim()
                        repeat_dims[1] = repeat
                        v = v.repeat(*repeat_dims)[:, :target_batch].contiguous()
                    chunks = torch.chunk(v, split_size, dim=1)
                    split_dict[k] = chunks[i].contiguous()
                else:
                    split_dict[k] = v
            result_list.append(split_dict)
        return result_list

    def _index_rollouts(self) -> None:
        base = self._base_dir
        if os.path.isfile(base):
            payload = self._torch_load_safely(base)
            meta = payload if isinstance(payload, dict) else {}
            step = int(meta.get("global_step", 0))
            stage_id = int(meta.get("stage_id", 0))
            self._all_steps = [step]
            self._step_to_files = {step: {stage_id: base}}
            return

        pattern = os.path.join(base, "global_step_*", "rollout_rank*_stage*.pt")
        files = sorted(glob.glob(pattern))
        step_to_files: dict[int, dict[int, str]] = {}
        steps_set = set()
        for path in files:
            step_dir = os.path.basename(os.path.dirname(path))
            if not step_dir.startswith("global_step_"):
                continue
            try:
                step = int(step_dir.split("global_step_")[-1])
            except Exception:
                continue
            fname = os.path.basename(path)
            stage_id = None
            if "_stage" in fname and fname.endswith(".pt"):
                try:
                    stage_id = int(fname.split("_stage")[-1].split(".pt")[0])
                except Exception:
                    stage_id = None
            if stage_id is None:
                continue
            steps_set.add(step)
            if step not in step_to_files:
                step_to_files[step] = {}
            if stage_id not in step_to_files[step]:
                step_to_files[step][stage_id] = path

        self._all_steps = sorted(list(steps_set))
        self._step_to_files = step_to_files

        if self._strict:
            if len(self._all_steps) == 0:
                raise ValueError(f"No offline rollouts found under: {base}")
            for step in self._all_steps:
                stage_map = self._step_to_files.get(step, {})
                for stage_id in range(self.num_pipeline_stages):
                    if stage_id not in stage_map:
                        raise ValueError(
                            f"Missing offline rollout file for step={step} stage_id={stage_id} "
                            f"under {base}"
                        )

    def _pick_step(self) -> int:
        if len(self._all_steps) == 0:
            raise ValueError("No offline rollouts indexed")
        if self._loop:
            idx = int(self.global_step) % len(self._all_steps)
            return self._all_steps[idx]
        if int(self.global_step) >= len(self._all_steps):
            return self._all_steps[-1]
        return self._all_steps[int(self.global_step)]

    def generate(self):
        step = self._pick_step()

        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)

        stage_to_file = self._step_to_files.get(step, {})
        for stage_id in range(self.num_pipeline_stages):
            path = stage_to_file.get(stage_id, None)
            if path is None:
                if self._strict:
                    raise ValueError(f"Missing offline rollout for step={step} stage_id={stage_id}")
                continue
            payload = self._torch_load_safely(path)
            base_dict = self._resolve_data(payload)
            split_dicts = self._split_rollout_dict(base_dict, split_num)
            for item in split_dicts:
                self.channel.put(item=item, key=self._replay_buffer_name)
