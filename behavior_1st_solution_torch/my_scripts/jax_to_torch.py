import os
import torch
import jax
import numpy as np
import orbax.checkpoint as ocp
from flax.traverse_util import flatten_dict
import re

# ==========================================
# 1. 这里引入你写好的 PyTorch 模型类
# ==========================================
# 例如：from my_model_file import MyPyTorchModel
# 假设你的模型定义在 b1k/models/pi_behavior_pytorch.py 中
try:
    import sys
    sys.path.append("/ms/PA/yejiaquan/code/behavior-1k-solution-main/openpi/src/openpi/models_pytorch")
    from pi_behavior_pytorch import PiBehaviorPytorch
except ImportError:
    print("【注意】请在脚本开头修改 import，引入你的 PyTorch 模型类")
    PiBehaviorPyTorch = None

def robust_load_jax_checkpoint(ckpt_path):
    """
    强力加载 JAX Checkpoint，自动处理 Sharding 报错问题
    """
    print(f"正在加载 JAX 权重: {ckpt_path}")

    # 强制 CPU
    jax.config.update('jax_platform_name', 'cpu')

    # 构造虚拟 Mesh (骗过 Orbax 的多设备检查)
    mesh = jax.sharding.Mesh(jax.devices()[:1], ('data',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpointer = ocp.PyTreeCheckpointer()

    # 1. 先尝试读取结构
    try:
        # Orbax 新旧版本 API 兼容
        if hasattr(checkpointer, 'metadata'):
            structure = checkpointer.metadata(ckpt_path)
        else:
            structure = checkpointer.structure(ckpt_path)
    except Exception:
        # 如果读不到结构，可能是纯参数保存，尝试直接 restore
        print("无法读取元数据，尝试直接加载...")
        return checkpointer.restore(ckpt_path, item=None)

    # 2. 构造 restore_args，强制所有参数加载到 CPU
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=sharding),
        structure
    )

    # 3. 执行加载
    raw_state = checkpointer.restore(ckpt_path, restore_args=restore_args)

    # 4. 提取参数 (兼容 TrainState 包装或纯 Params)
    if 'params' in raw_state:
        return raw_state['params']
    return raw_state

def jax_params_to_torch_state_dict(jax_params, pytorch_model=None):
    """
    将 JAX 参数字典转换为 PyTorch state_dict
    """
    print("正在转换参数格式 (JAX -> PyTorch)...")

    # 1. 扁平化 JAX 参数 (nested dict -> flat dict with dot separator)
    # 处理可能的 nnx.State 包装
    if hasattr(jax_params, 'to_pure_dict'):
        jax_params = jax_params.to_pure_dict()

    flat_jax = flatten_dict(jax_params, sep='.')
    torch_dict = {}

    for key, value in flat_jax.items():
        # 跳过非数据项
        if not isinstance(value, (jax.Array, np.ndarray)):
            continue

        # 转为 PyTorch Tensor
        tensor = torch.from_numpy(np.array(value))

        # === 命名规则转换 ===
        new_key = key
        new_key = new_key.replace('kernel', 'weight')  # Linear/Conv 权重
        new_key = new_key.replace('scale', 'weight')   # Norm 层权重
        # new_key = new_key.replace('bias', 'bias')    # Bias 不变

        # === 形状规则转换 (最关键的部分) ===

        # 规则 A: Linear 层 (2D) -> 需要转置
        # JAX: (Input, Output) vs PyTorch: (Output, Input)
        if 'weight' in new_key and len(tensor.shape) == 2:
            # 排除 Embedding (Input, Dim) -> 不需要转置
            if 'embed' not in new_key and 'position' not in new_key:
                tensor = tensor.t()

        # 规则 B: Conv2d 层 (4D) -> 需要 Permute
        # JAX: (H, W, In, Out) vs PyTorch: (Out, In, H, W)
        if 'conv' in new_key and len(tensor.shape) == 4:
            tensor = tensor.permute(3, 2, 0, 1)

        torch_dict[new_key] = tensor

    # === (可选) 如果传入了 PyTorch 模型，打印匹配率 ===
    if pytorch_model is not None:
        pt_keys = set(pytorch_model.state_dict().keys())
        jax_keys = set(torch_dict.keys())
        missing = pt_keys - jax_keys
        unexpected = jax_keys - pt_keys
        print(f"\n转换统计:")
        print(f"PyTorch 模型参数量: {len(pt_keys)}")
        print(f"JAX 提取参数量: {len(jax_keys)}")
        print(f"成功匹配: {len(pt_keys - missing)}")
        if len(missing) > 0:
            print(f"PyTorch 中缺失的键 (前5个): {list(missing)[:5]}")

    return torch_dict

def main():
    # ================= 配置区域 =================
    # 1. JAX 权重文件夹路径
    JAX_CKPT_PATH = "/data1/heyelin/models/behavior_1st/checkpoint_2/params"

    # 2. 输出路径
    OUTPUT_PATH = "/data1/heyelin/models/behavior_1st_torch/checkpoint_2/random_state.pth"
    # ===========================================

    # 1. 实例化你的 PyTorch 模型 (用于结构验证，可选)
    # model_pt = PiBehaviorPyTorch(...)
    model_pt = None # 如果你还没写好初始化代码，这里设为 None 也可以跑

    # 2. 加载 JAX
    jax_params = robust_load_jax_checkpoint(JAX_CKPT_PATH)

    # 3. 转换
    state_dict = jax_params_to_torch_state_dict(jax_params, model_pt)

    # 4. 保存
    torch.save(state_dict, OUTPUT_PATH)
    print(f"\n✅ 转换完成！文件已保存至: {OUTPUT_PATH}")
    print("使用方法: model.load_state_dict(torch.load('...'), strict=False)")

if __name__ == "__main__":
    main()
