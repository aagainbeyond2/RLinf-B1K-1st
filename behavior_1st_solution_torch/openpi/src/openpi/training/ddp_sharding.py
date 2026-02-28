import contextlib
import logging

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# In FSDP, we shard the data across both the batch and FSDP axes.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


# === CHANGED ===
# 1) make_mesh 允许不传 num_fsdp_devices（默认 1，等价“不开 FSDP 参数分片”）
# 2) 增加一些更明确的合法性检查
def make_mesh(num_fsdp_devices: int | None = None) -> jax.sharding.Mesh:
    """
    Create a 2D mesh (BATCH_AXIS, FSDP_AXIS).
    - num_fsdp_devices = 1: "DDP-like" (no parameter sharding; FSDP axis size=1)
    - num_fsdp_devices > 1: enables FSDP-style parameter sharding along FSDP axis

    Note: DATA sharding is typically across DATA_AXIS=(batch, fsdp), so the *batch dimension*
    is sharded across mesh.shape[batch] * mesh.shape[fsdp] devices.
    """
    if num_fsdp_devices is None:
        num_fsdp_devices = 1  # default: no FSDP param sharding
    if num_fsdp_devices <= 0:
        raise ValueError(f"num_fsdp_devices must be >= 1, got {num_fsdp_devices}.")

    ndev = jax.device_count()
    if ndev % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {ndev} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )

    mesh_shape = (ndev // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


# === CHANGED ===
# 计算 DATA_AXIS 的分片数（batch 会被切成这么多份；fsdp=1 时就是 device_count）
def data_axis_size(mesh: jax.sharding.Mesh) -> int:
    return int(np.prod([mesh.shape[a] for a in DATA_AXIS]))


# === CHANGED ===
# 常用 sharding helper：data-parallel input sharding（按 DATA_AXIS 切 batch）
def data_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
    # NamedSharding = Mesh + PartitionSpec :contentReference[oaicite:5]{index=5}
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(DATA_AXIS))


# === CHANGED ===
# 常用 sharding helper：replicated（参数/状态全复制）
def replicated_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
    # PartitionSpec() => all axes replicated (unspecified axes are replicated) :contentReference[oaicite:6]{index=6}
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


# === CHANGED ===
# 把一个 pytree 的每个 leaf 都标成 replicated —— 你要的“纯 DDP train_state”就用这个
def replicated_tree_sharding(pytree, mesh: jax.sharding.Mesh, *, log: bool = False):
    s = replicated_sharding(mesh)

    def _to_replica(_kp, _leaf):
        return s

    out = jax.tree_util.tree_map_with_path(_to_replica, pytree)

    if log:
        logging.info("Replicated train_state sharding enabled: all leaves use PartitionSpec().")
    return out


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """Plumbing the mesh deep into the module tree is extremeley cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below."""
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # if fsdp is not actually going to be used, replicate everything to avoid extraneous logging
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate small arrays
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # replicate if no valid sharding was found
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)