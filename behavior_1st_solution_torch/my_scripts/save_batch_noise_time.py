import os

import builtins
import dill
import jax
import numpy as np

from b1k.models.pi_behavior import PiBehavior


def save_batch_noise_time(
    reproduce_dir="/data1/heyelin/models/behavior_1st_torch/checkpoint_2/",
):
    print(f"Loading JAX artifacts from {reproduce_dir}...")
    with open(os.path.join(reproduce_dir, "config_from_jax.pkl"), "rb") as f:
        config = dill.load(f)

    with open(os.path.join(reproduce_dir, "batch_from_jax.pkl"), "rb") as f:
        batch = dill.load(f)
        _, actions = batch

    with open(os.path.join(reproduce_dir, "data_config_from_jax.pkl"), "rb") as f:
        data_config = dill.load(f)

    print("Artifacts loaded. Initializing model for noise generation...")
    norm_stats = data_config.norm_stats
    if norm_stats is None:
        raise ValueError("norm_stats is missing from data_config!")

    debug_noise_time_path = os.path.join(reproduce_dir, "batch_noise_time.pkl")
    seed = int(getattr(config, "seed", 0) or 0)
    rng = jax.random.key(seed)
    model_rng, noise_rng, time_rng = jax.random.split(rng, 3)

    builtins.ENABLE_SIGLIP_DEBUG = False
    model = config.model.create(model_rng)
    builtins.ENABLE_SIGLIP_DEBUG = True

    if isinstance(model, PiBehavior):
        model.load_correlation_matrix(norm_stats)

    batch_size = int(np.array(actions).shape[0])
    noise = model.generate_correlated_noise(noise_rng, batch_size)
    time = jax.random.beta(time_rng, 1.5, 1.0, (batch_size,)) * 0.999 + 0.001

    with open(debug_noise_time_path, "wb") as f:
        dill.dump({"noise": np.array(noise), "time": np.array(time)}, f)

    print(f"Saved batch_noise_time.pkl to: {debug_noise_time_path}")
    print(f"noise shape: {np.array(noise).shape}, time shape: {np.array(time).shape}")
    return {"noise": noise, "time": time}


if __name__ == "__main__":
    save_batch_noise_time()

