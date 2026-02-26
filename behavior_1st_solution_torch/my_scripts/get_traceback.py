import os
import sys
import jax
import torch
import numpy as np
import flax.nnx as nnx
from flax.traverse_util import flatten_dict
import logging
import etils.epath as epath

# === 确保项目根目录在 PYTHONPATH 中 ===
sys.path.append(os.getcwd())

from b1k.training import config as _config
from b1k.training import checkpoints as _checkpoints
from b1k.training import data_loader as _data_loader
from openpi.training import sharding
import openpi.training.utils as training_utils
from b1k.models.pi_behavior import PiBehavior

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jax_model_strict(config):
    # 1. 设置 Mesh
    mesh = sharding.make_mesh(jax.devices())

    # 2. 初始化 Data Loader (这里最可能报错)
    logger.info(">>> Initializing Data Loader...")

    # 注意：这里我们不做任何修改，保留原始逻辑，让它自然报错
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = _data_loader.create_behavior_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        world_size=1,
    )

    # 如果上面没报错，继续后面的流程...
    data_config = data_loader.data_config()
    norm_stats = data_config.norm_stats

    logger.info(">>> Creating Model...")
    init_rng = jax.random.key(0)
    rng, model_rng = jax.random.split(init_rng)
    model = config.model.create(model_rng)

    if isinstance(model, PiBehavior) and norm_stats is not None:
        model.load_correlation_matrix(norm_stats)

    params = nnx.state(model)
    return params

def main():
    # 解析命令行参数
    config = _config.cli()

    # 强制 CPU (但这不会修复逻辑错误，只会改变运行设备)
    jax.config.update('jax_platform_name', 'cpu')

    logger.info(">>> Starting strict load to reveal traceback...")

    # === 关键：没有任何 try-except 保护 ===
    # 程序将在这里直接崩溃，并打印出你需要的完整 Traceback
    load_jax_model_strict(config)

if __name__ == "__main__":
    main()