import os

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.envs import get_env_cls
from rlinf.envs.env_manager import EnvManager
from rlinf.scheduler import Cluster, PackedPlacementStrategy, Worker

os.environ.setdefault("EMBODIED_PATH", os.path.abspath(os.path.dirname(__file__)))

mp.set_start_method("spawn", force=True)


class BehaviorRandomSimWorker(Worker):
    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        action_dim: int,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.seed = seed
        self._rng = None
        self._env_mgr = None

    def init_worker(self):
        env_cfg = self.cfg.env.eval
        with open_dict(env_cfg):
            env_cfg.auto_reset = True

        env_cls = get_env_cls(env_cfg.simulator_type, env_cfg)

        self._env_mgr = EnvManager(
            cfg=env_cfg,
            rank=self._rank,
            num_envs=self.num_envs,
            seed_offset=self.seed + self._rank,
            total_num_processes=self._world_size,
            env_cls=env_cls,
            enable_offload=False,
        )

        self._rng = np.random.default_rng(self.seed + self._rank)
        obs, _ = self._env_mgr.reset()
        return {
            "rank": self._rank,
            "num_envs": self.num_envs,
            "obs_keys": list(obs.keys()),
        }

    def rollout_random(self, num_chunks: int, chunk_size: int):
        total_reward = 0.0
        finished_episodes = 0
        for _ in range(num_chunks):
            chunk_actions = self._rng.uniform(
                low=-1.0,
                high=1.0,
                size=(self.num_envs, chunk_size, self.action_dim),
            ).astype(np.float32)
            _, chunk_rewards, chunk_terminations, chunk_truncations, infos = (
                self._env_mgr.chunk_step(chunk_actions)
            )
            total_reward += float(chunk_rewards.float().sum().item())
            chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
            if infos is not None and isinstance(infos, dict) and "_final_info" in infos:
                finished_episodes += int(infos["_final_info"].sum().item())
            elif chunk_dones.any():
                finished_episodes += int(chunk_dones[:, -1].sum().item())
        return {
            "rank": self._rank,
            "total_reward": total_reward,
            "finished_episodes": finished_episodes,
        }

    def shutdown(self):
        if self._env_mgr is None:
            return
        if self._env_mgr.env is not None and hasattr(self._env_mgr.env, "close"):
            self._env_mgr.env.close()
        if self._env_mgr.process is not None:
            self._env_mgr.stop_simulator()
        self._env_mgr = None


@hydra.main(version_base="1.1", config_path="config", config_name="behavior_ppo_openvlaoft")
def main(cfg: DictConfig):
    cfg = validate_cfg(cfg)

    num_workers = 2
    total_num_envs = int(cfg.env.eval.total_num_envs)
    if total_num_envs % num_workers != 0:
        raise ValueError(
            f"env.eval.total_num_envs ({total_num_envs}) must be divisible by num_workers ({num_workers})"
        )
    num_envs_per_worker = total_num_envs // num_workers

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement_strategy = PackedPlacementStrategy(0, num_workers - 1)

    env_cfg = cfg.env.eval
    with open_dict(env_cfg):
        env_cfg.total_num_envs = total_num_envs
        env_cfg.auto_reset = True

    action_dim = int(cfg.actor.model.action_dim)
    group = BehaviorRandomSimWorker.create_group(
        cfg=cfg, num_envs=num_envs_per_worker, action_dim=action_dim, seed=0
    ).launch(
        cluster=cluster,
        name="BehaviorRandomSimGroup",
        placement_strategy=placement_strategy,
    )

    init_infos = group.init_worker().wait()
    results = group.rollout_random(num_chunks=4, chunk_size=8).wait()
    # group.shutdown().wait()

    # print(OmegaConf.to_yaml(OmegaConf.create(init_infos)))
    # print(OmegaConf.to_yaml(OmegaConf.create(results)))


if __name__ == "__main__":
    main()

# python examples/embodiment/rollout_behavior_random_worker_group.py


