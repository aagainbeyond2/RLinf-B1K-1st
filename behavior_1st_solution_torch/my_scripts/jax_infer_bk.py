import dill
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import os
import logging
from openpi.training import sharding
from openpi.training import utils as training_utils
from b1k.training import checkpoints as _checkpoints
from b1k.models.pi_behavior import PiBehavior
import optax
import dataclasses
from etils import epath
from b1k.training import weight_loaders
import numpy as np
import builtins

# 设置日志
logging.basicConfig(level=logging.INFO)

def load_and_infer(
    reproduce_dir="/ms/PA/yejiaquan/b1k_models_pytorch/reproduce/",
    checkpoint_dir="/root/.cache/openpi/openpi-assets/checkpoints/pi05_behavior_50t/params/snapshots/114902410a6439ad1a0b58a112b84159d3ab3f65/params", # 如果config里有，可以填None；否则手动指定
    step=None # 指定恢复的step，None为自动查找最新
):
    print(f"Loading files from {reproduce_dir}...")
    
    # 1. 加载保存的中间变量
    # -------------------------------------------------------------------------
    with open(os.path.join(reproduce_dir, "config_from_jax.pkl"), "rb") as f:
        config = dill.load(f)
    
    with open(os.path.join(reproduce_dir, "batch_from_jax.pkl"), "rb") as f:
        batch = dill.load(f)
        observation, actions = batch # 解包 batch
        
    with open(os.path.join(reproduce_dir, "data_config_from_jax.pkl"), "rb") as f:
        data_config = dill.load(f)
        
    print("Files loaded successfully.")

    # 2. 初始化 Mesh 和 RNG (参考 train.py main 函数)
    # -------------------------------------------------------------------------
    # 覆盖 checkpoint_dir 如果手动指定了
    #if checkpoint_dir:
        #config.checkpoint_dir = checkpoint_dir
        #config = dataclasses.replace(config, checkpoint_dir=epath.Path(checkpoint_dir))
    #    object.__setattr__(config, 'checkpoint_dir', epath.Path(checkpoint_dir))

    rng = jax.random.key(config.seed if config.seed else 0)
    init_rng, infer_rng = jax.random.split(rng)
    
    # 创建 Mesh (用于参数分片，必须与训练时一致或兼容)
    mesh = sharding.make_mesh(config.fsdp_devices)
    
    # 3. 初始化 TrainState 的 Shape (参考 init_train_state)
    # -------------------------------------------------------------------------
    # 我们需要构建一个“空”的 TrainState 来告诉 checkpoint loader 参数的形状
    print("Initializing model shape...")
    
    # 获取 norm_stats 用于 correlation matrix (PiBehavior 特有逻辑)
    norm_stats = data_config.norm_stats
    if norm_stats is None:
        raise ValueError("norm_stats is missing from data_config!")

    # 定义初始化函数 (逻辑复用 train.py 中的 init_train_state)
    def init_model_fn(rng):
        # 创建模型
        model = config.model.create(rng)
        
        # 加载 Correlation Matrix
        if isinstance(model, PiBehavior) and norm_stats is not None:
            model.load_correlation_matrix(norm_stats)
            print("Loaded correlation matrix into model structure.")
            
        params = nnx.state(model)
        
        # [修复开始] ------------------------------------------------
        # 错误写法: params = params.replace(params.value.astype(jnp.bfloat16))
        # 正确写法: 使用 jax.tree.map 遍历每个参数进行转换
        def cast_to_bf16(leaf):
            # 确保只转换浮点类型的参数 (避免转换步数、整数掩码等)
            if hasattr(leaf, 'value') and jnp.issubdtype(leaf.value.dtype, jnp.floating):
                return leaf.replace(leaf.value.astype(jnp.bfloat16))
            return leaf

        params = jax.tree.map(cast_to_bf16, params)
        # [修复结束] ------------------------------------------------
        dummy_tx = optax.identity()
        
        # 返回 TrainState
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=dummy_tx,
            opt_state={},
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # 计算形状 (使用 jax.eval_shape 避免实际分配内存)
    train_state_shape = jax.eval_shape(init_model_fn, init_rng)
    
    # 定义 Sharding Spec
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)

    # 4. 加载 Checkpoint (使用 B1K 专用的 PiBehaviorWeightLoader)
    # -------------------------------------------------------------------------
    print(f"Restoring weights using PiBehaviorWeightLoader from {checkpoint_dir}...")
    
    ckpt_path = epath.Path(checkpoint_dir)
    
    try:
        # 初始化 Loader
        # 根据你的提示，使用 PiBehaviorWeightLoader
        loader = weight_loaders.PiBehaviorWeightLoader(str(ckpt_path))
        
        # 准备目标参数结构 (纯字典)
        # 这一步很重要，Loader 需要知道预期的形状来做校验或过滤
        target_params_dict = train_state_shape.params.to_pure_dict()
        
        # 加载参数
        # 注意：PiBehaviorWeightLoader 内部通常会自动处理 msgpack/orbax 的判断
        loaded_params_dict = loader.load(target_params_dict)
        print("Weights loaded successfully.")

        # 将加载的参数合并回模型
        # 1. 临时重建模型
        temp_model = nnx.merge(train_state_shape.model_def, train_state_shape.params)

        # 2. 更新参数 (nnx.update 能够智能匹配字典结构)
        nnx.update(temp_model, loaded_params_dict)
        
        # 3. 提取新 State
        new_params = nnx.state(temp_model)
        
        # 4. 替换 TrainState
        restored_state = dataclasses.replace(
            train_state_shape,
            params=new_params,
            step=jnp.array(0, dtype=jnp.int32) 
        )
        print("Model state successfully restored via PiBehaviorWeightLoader.")

    except Exception as e:
        print(f"PiBehaviorWeightLoader failed: {e}")
        # 如果这个特定的 Loader 失败，打印详细堆栈以便调试
        import traceback
        traceback.print_exc()
        raise RuntimeError("Failed to load weights.")

    print(f"Model state restored.")

    # 5. 合并模型并运行推理 (Forward Pass)
    # -------------------------------------------------------------------------
    print("Running inference...")
    
    # 将加载的参数 (params) 合并回 图定义 (model_def)
    model = nnx.merge(restored_state.model_def, restored_state.params)

    # 必须设为 eval 模式
    model.eval()

    with open(os.path.join(reproduce_dir, "batch_noise_time.pkl"), "rb") as f:
        debug_data = dill.load(f)
        noise = debug_data["noise"]
        time = debug_data["time"]

    # [修复]: 使用 @nnx.jit 替代 @jax.jit
    @nnx.jit
    def forward_fn(model, rng, obs, act):
        # 注意：nnx.jit 会自动处理 model 的状态拆分与合并
        #return model.compute_detailed_loss(
        return model.compute_detailed_loss_debug(
            rng,
            obs, 
            act, 
            train=False,
            #num_flow_samples=config.num_flow_samples
            num_flow_samples=1, # for debug
            noise=noise,
            time=time,
        )

    # 执行推理
    loss_dict = forward_fn(model, infer_rng, observation, actions)

    # # save noise and time to disk
    # debug_data = {
    #     "noise": np.array(loss_dict['debug_noise']),
    #     "time": np.array(loss_dict['debug_time']),
    # }
    # with open(os.path.join(reproduce_dir, "batch_noise_time.pkl"), "wb") as f:
    #     dill.dump(debug_data, f)


    # 打印结果
    print("-" * 40)
    print("Inference Results (compute_detailed_loss):")
    for k, v in loss_dict.items():
        val = jnp.mean(v)
        print(f"{k}: {val:.6f}")
    print("-" * 40)
    
    return loss_dict

if __name__ == "__main__":
    # 运行
    load_and_infer()