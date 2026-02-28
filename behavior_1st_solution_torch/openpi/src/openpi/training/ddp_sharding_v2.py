import contextlib
import logging
import jax
import numpy as np
from typing import Any

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# DATA_AXIS 在 DDP 模式下会自动降级为仅在 batch 轴切分
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)

class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None

def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    """
    创建 Mesh。
    对于 DDP 模式，传入 num_fsdp_devices=1。
    结果 mesh 形状为 (总卡数, 1)。
    """
    num_devices = jax.device_count()
    if num_devices % num_fsdp_devices != 0:
        raise ValueError(f"设备数 {num_devices} 必须能被 {num_fsdp_devices} 整除")
    mesh_shape = (num_devices // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))

@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None

def activation_sharding_constraint(pytree):
    """
    在 DDP 模式下（FSDP=1），DATA_AXIS 实际上只在 batch 轴生效。
    这确保了激活值在 8 张卡间切分，实现数据并行。
    """
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )

def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,
    log: bool = False,
):
    """
    针对 DDP 模式优化的分片逻辑。
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # 【DDP 核心分支】
        # 如果 FSDP 轴长度为 1，说明用户选择了 DDP 模式。
        # 此时显式返回空的 PartitionSpec()，强制全量复制模型参数。
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # --- 以下为 FSDP 模式（num_fsdp_devices > 1）逻辑 ---
        if not hasattr(array, "shape") or len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # 过滤小参数
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # 尝试寻找可切分的轴
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(f"Sharding {jax.tree_util.keystr(kp)} along axis {i}")
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
