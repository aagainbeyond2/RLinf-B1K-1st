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
import logging
import os
import sys

import torch
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)

from rlinf.config import SupportedModel, get_supported_model, torch_dtype_from_precision

logger = logging.getLogger(__name__)


def _maybe_add_openpi_to_sys_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate_paths = [
        os.path.join(repo_root, "behavior_1st_solution_torch", "src"),
        os.path.join(repo_root, "behavior_1st_solution_torch", "openpi", "src"),
        os.path.join(
            repo_root,
            "behavior_1st_solution_torch",
            "openpi",
            "packages",
            "openpi-client",
            "src",
        ),
    ]
    for path in candidate_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def get_vla_model_config_and_processor(cfg: DictConfig):
    model_type = get_supported_model(cfg.model.model_type)
    if model_type == SupportedModel.OPENVLA:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

        from .embodiment.prismatic.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

        model_config = AutoConfig.from_pretrained(cfg.tokenizer.tokenizer_model)

        dataset_statistics_path = os.path.join(
            cfg.tokenizer.tokenizer_model, "dataset_statistics.json"
        )
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(model_config, "norm_stats", norm_stats)
        image_processor = PrismaticImageProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True, padding_side="left"
        )
        input_processor = PrismaticProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )
    elif model_type == SupportedModel.OPENVLA_OFT:
        from prismatic.extern.hf.configuration_prismatic import (
            OpenVLAConfig as OpenVLAOFTConfig,
        )

        from .embodiment.prismatic.processing_prismatic import (
            MultiInputPrismaticProcessor as PrismaticProcessorOFT,
        )
        from .embodiment.prismatic.processing_prismatic import PrismaticImageProcessor

        AutoConfig.register("openvla", OpenVLAOFTConfig)
        AutoImageProcessor.register(OpenVLAOFTConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAOFTConfig, PrismaticProcessorOFT)

        model_config = OpenVLAOFTConfig.from_pretrained(
            cfg.tokenizer.tokenizer_model, center_crop=cfg.model.center_crop
        )
        image_processor = PrismaticImageProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True, padding_side="left"
        )
        input_processor = PrismaticProcessorOFT.from_pretrained(
            cfg.tokenizer.tokenizer_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )

    return model_config, input_processor


def get_model(cfg: DictConfig, override_config_kwargs=None):
    model_path = cfg.model_path
    torch_dtype = torch_dtype_from_precision(cfg.precision)
    model_type = get_supported_model(cfg.model_type)
    if model_type == SupportedModel.OPENVLA:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

        actor_model_config = OpenVLAConfig.from_pretrained(
            model_path, trust_remote_code=cfg.trust_remote_code
        )

        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(actor_model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(actor_model_config, "norm_stats", norm_stats)

        from .embodiment.openvla_action_model import OpenVLAForRLActionPrediction

        model = OpenVLAForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            hidden_size=cfg.hidden_size,
            unnorm_key=cfg.unnorm_key,
            config=actor_model_config,
            add_value_head=cfg.add_value_head,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            attn_implementation=cfg.attn_implementation,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
            trust_remote_code=cfg.trust_remote_code,
        )

        model.to(torch_dtype)

    elif model_type == SupportedModel.OPENVLA_OFT:
        from prismatic.extern.hf.configuration_prismatic import (
            OpenVLAConfig as OpenVLAOFTConfig,
        )

        from .embodiment.openvla_oft_action_model import OpenVLAOFTForRLActionPrediction

        AutoConfig.register("openvla", OpenVLAOFTConfig)
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=cfg.trust_remote_code
        )

        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(actor_model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(actor_model_config, "norm_stats", norm_stats)

        override_config_kwargs = cfg
        if override_config_kwargs is not None:
            for key, val in override_config_kwargs.items():
                setattr(actor_model_config, key, val)

        model = OpenVLAOFTForRLActionPrediction.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            # attn_implementation="flash_attention_2",
            config=actor_model_config,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            trust_remote_code=True,
            add_value_head=cfg.add_value_head,
        )

        # oft add
        model.vision_backbone.set_num_images_in_input(cfg.get("num_images_in_input", 1))

        model.to(torch_dtype)

    elif model_type == SupportedModel.OPENPI:
        import glob

        _maybe_add_openpi_to_sys_path()
        try:
            import openpi.shared.download as download
            import openpi.transforms as transforms
            import safetensors
            from openpi.training import checkpoints as _checkpoints
        except ModuleNotFoundError as e:
            missing_root = (getattr(e, "name", "") or "").split(".")[0]
            if missing_root != "openpi":
                raise
            _maybe_add_openpi_to_sys_path()
            try:
                import openpi.shared.download as download
                import openpi.transforms as transforms
                import safetensors
                from openpi.training import checkpoints as _checkpoints
            except ModuleNotFoundError as e2:
                missing_root2 = (getattr(e2, "name", "") or "").split(".")[0]
                if missing_root2 != "openpi":
                    raise
                raise ModuleNotFoundError(
                    "No module named 'openpi'. "
                    "Expected OpenPI sources at "
                    "<repo>/behavior_1st_solution_torch/openpi/src or install it into the env."
                ) from e2

        from .embodiment.openpi import get_openpi_config
        from .embodiment.openpi_action_model import (
            OpenPi0Config,
            OpenPi0ForRLActionPrediction,
        )

        def _unwrap_state_dict(maybe_state_dict):
            if isinstance(maybe_state_dict, dict):
                for k in ("state_dict", "model_state_dict", "model", "params"):
                    v = maybe_state_dict.get(k, None)
                    if isinstance(v, dict):
                        return v
            return maybe_state_dict

        def _load_pytorch_weights(model, weight_file: str, *, strict_load: bool):
            try:
                state = torch.load(weight_file, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(weight_file, map_location="cpu")
            except Exception:
                state = torch.load(weight_file, map_location="cpu")
            state = _unwrap_state_dict(state)
            if strict_load:
                incompat = model.load_state_dict(state, strict=False)
                missing = list(getattr(incompat, "missing_keys", []) or [])
                unexpected = list(getattr(incompat, "unexpected_keys", []) or [])
                if missing or unexpected:
                    raise RuntimeError(
                        "OpenPI checkpoint keys mismatch "
                        f"(missing={len(missing)} unexpected={len(unexpected)}). "
                        f"missing[:20]={missing[:20]} unexpected[:20]={unexpected[:20]} "
                        f"weight_file={weight_file}"
                    )
                return
            model.load_state_dict(state, strict=False)

        def _load_norm_stats_from_known_locations(checkpoint_dir: str, asset_id: str):
            candidate_roots = [
                checkpoint_dir,
                os.path.join(checkpoint_dir, "assets"),
                *sorted(glob.glob(os.path.join(checkpoint_dir, "outputs", "assets", "*"))),
            ]
            loaders = []
            if is_pi_behavior:
                try:
                    from b1k.training import checkpoints as _b1k_checkpoints

                    loaders.append(_b1k_checkpoints.load_norm_stats)
                except Exception:
                    pass
            loaders.append(_checkpoints.load_norm_stats)
            for assets_root in candidate_roots:
                for loader in loaders:
                    try:
                        return loader(assets_root, asset_id)
                    except FileNotFoundError:
                        continue
                    except Exception:
                        continue
            return None

        # config
        config_name = getattr(cfg.openpi, "config_name", None)
        actor_train_config = get_openpi_config(config_name, model_path=model_path)
        actor_model_config = actor_train_config.model
        is_pi_behavior = bool(
            isinstance(config_name, str)
            and config_name.startswith("pi_behavior")
            or getattr(actor_model_config, "model_type", None) == "pi_behavior"
        )
        if is_pi_behavior:
            import types

            from .embodiment.openpi_action_model import PiBehaviorForRLActionPrediction

            actor_model_config_dict = dict(getattr(actor_model_config, "__dict__", {}) or {})
            actor_model_config_dict.setdefault("config_name", config_name)
            if "model_type" not in actor_model_config_dict:
                import openpi.models.model as _model

                actor_model_config_dict["model_type"] = _model.ModelType.PI0
            actor_model_config_dict["add_value_head"] = bool(getattr(cfg, "add_value_head", False))

            override_config_kwargs = cfg.openpi
            if override_config_kwargs is not None:
                for key, val in override_config_kwargs.items():
                    actor_model_config_dict[key] = val

            actor_model_config = types.SimpleNamespace(**actor_model_config_dict)
        else:
            actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
        strict_load = getattr(cfg.openpi, "strict_load", None)
        if strict_load is None:
            strict_load = bool(
                isinstance(config_name, str)
                and (config_name.startswith("pi_behavior") or "b1k" in config_name)
            )
        strict_load = bool(strict_load)
        if not is_pi_behavior:
            override_config_kwargs = cfg.openpi
            if override_config_kwargs is not None:
                for key, val in override_config_kwargs.items():
                    actor_model_config.__dict__[key] = val
            actor_model_config.__dict__["add_value_head"] = bool(getattr(cfg, "add_value_head", False))
        # load model
        checkpoint_dir = download.maybe_download(str(model_path))
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        pytorch_weight_paths = []
        if not weight_paths:
            candidates = [
                "final_b1k_pytorch.pt",
                "final_b1k_pytorch_checkEnd2End.pt",
                "pytorch_model.bin",
                "model.pt",
                "model.pth",
            ]
            for name in candidates:
                p = os.path.join(checkpoint_dir, name)
                if os.path.isfile(p):
                    pytorch_weight_paths.append(p)
                    break
            if not pytorch_weight_paths:
                pytorch_weight_paths = sorted(
                    glob.glob(os.path.join(checkpoint_dir, "*.pt"))
                    + glob.glob(os.path.join(checkpoint_dir, "*.pth"))
                    + glob.glob(os.path.join(checkpoint_dir, "*.bin"))
                )
            if not pytorch_weight_paths:
                weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]

        logger.info(
            "OpenPI loading weights "
            f"strict_load={strict_load} "
            f"safetensors={weight_paths} "
            f"pytorch={pytorch_weight_paths}"
        )

        def _has_value_head_weights_from_keys(keys) -> bool:
            for k in keys:
                ks = str(k)
                if ks.startswith("value_head.") or ".value_head." in ks:
                    return True
            return False

        disabled_value_head_for_load = False
        if is_pi_behavior:
            add_value_head_requested = bool(getattr(actor_model_config, "add_value_head", False))
            if strict_load and add_value_head_requested:
                has_value_head_weights = False
                if weight_paths:
                    for weight_path in weight_paths:
                        try:
                            with safetensors.safe_open(weight_path, framework="pt") as f:
                                has_value_head_weights = _has_value_head_weights_from_keys(f.keys())
                        except Exception:
                            continue
                        if has_value_head_weights:
                            break
                elif pytorch_weight_paths:
                    for weight_path in pytorch_weight_paths:
                        try:
                            _state0 = torch.load(
                                weight_path,
                                map_location="cpu",
                                weights_only=True,
                            )
                        except TypeError:
                            _state0 = torch.load(
                                weight_path,
                                map_location="cpu",
                            )
                        except Exception:
                            _state0 = torch.load(
                                weight_path,
                                map_location="cpu",
                            )
                        _state0 = _unwrap_state_dict(_state0)
                        if isinstance(_state0, dict):
                            has_value_head_weights = _has_value_head_weights_from_keys(
                                _state0.keys()
                            )
                        if has_value_head_weights:
                            break

                if not has_value_head_weights:
                    disabled_value_head_for_load = True
                    setattr(actor_model_config, "add_value_head", False)

            model = PiBehaviorForRLActionPrediction(actor_model_config)
        else:
            model = OpenPi0ForRLActionPrediction(actor_model_config)
        # train expert only
        if getattr(actor_model_config, "train_expert_only", False) or getattr(
            cfg.openpi, "train_expert_only", False
        ):
            if hasattr(model, "freeze_vlm"):
                model.freeze_vlm()

        for weight_path in weight_paths:
            safetensors.torch.load_model(model, weight_path, strict=strict_load)
        for weight_path in pytorch_weight_paths:
            _load_pytorch_weights(model, weight_path, strict_load=strict_load)

        if is_pi_behavior and disabled_value_head_for_load:
            from rlinf.models.embodiment.modules.value_head import ValueHead

            value_after_vlm = bool(getattr(actor_model_config, "value_after_vlm", False))
            proj_width = 2048 if value_after_vlm else 1024
            model.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
            setattr(actor_model_config, "add_value_head", True)
            if hasattr(model, "config"):
                setattr(model.config, "add_value_head", True)
            if hasattr(model, "use_vlm_value"):
                model.use_vlm_value = value_after_vlm
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        # fsdp replace
        # model.paligemma_with_expert.replace_gemma_decoder_layers()
        # load data stats
        data_config = actor_train_config.data.create(
            actor_train_config.assets_dirs, actor_model_config
        )
        norm_stats = None
        if norm_stats is None:
            # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
            # that the policy is using the same normalization stats as the original training process.
            if data_config.asset_id is None:
                raise ValueError("Asset id is required to load norm stats.")
            norm_stats = _load_norm_stats_from_known_locations(
                checkpoint_dir, data_config.asset_id
            )
        # wrappers
        repack_transforms = transforms.Group()
        default_prompt = None
        if is_pi_behavior:
            model_input_transforms = [
                t
                for t in data_config.model_transforms.inputs
                if not isinstance(t, transforms.TokenizePrompt)
            ]
            use_per_timestamp_norm = False
            if isinstance(norm_stats, dict):
                for v in norm_stats.values():
                    if hasattr(v, "per_timestamp_mean") or hasattr(v, "per_timestamp_std"):
                        use_per_timestamp_norm = True
                        break
            if use_per_timestamp_norm:
                from b1k.transforms_normalize import (
                    NormalizeWithPerTimestamp,
                    UnnormalizeWithPerTimestamp,
                )

                model.setup_wrappers(
                    transforms=[
                        *repack_transforms.inputs,
                        *data_config.data_transforms.inputs,
                        NormalizeWithPerTimestamp(
                            norm_stats,
                            use_quantiles=data_config.use_quantile_norm,
                            use_per_timestamp=True,
                        ),
                        *model_input_transforms,
                    ],
                    output_transforms=[
                        *data_config.model_transforms.outputs,
                        UnnormalizeWithPerTimestamp(
                            norm_stats,
                            use_quantiles=data_config.use_quantile_norm,
                            use_per_timestamp=True,
                        ),
                        *data_config.data_transforms.outputs,
                        *repack_transforms.outputs,
                    ],
                )
            else:
                model.setup_wrappers(
                    transforms=[
                        *repack_transforms.inputs,
                        *data_config.data_transforms.inputs,
                        transforms.Normalize(
                            norm_stats, use_quantiles=data_config.use_quantile_norm
                        ),
                        *model_input_transforms,
                    ],
                    output_transforms=[
                        *data_config.model_transforms.outputs,
                        transforms.Unnormalize(
                            norm_stats, use_quantiles=data_config.use_quantile_norm
                        ),
                        *data_config.data_transforms.outputs,
                        *repack_transforms.outputs,
                    ],
                )
        else:
            model.setup_wrappers(
                transforms=[
                    *repack_transforms.inputs,
                    transforms.InjectDefaultPrompt(default_prompt),
                    *data_config.data_transforms.inputs,
                    transforms.Normalize(
                        norm_stats, use_quantiles=data_config.use_quantile_norm
                    ),
                    *data_config.model_transforms.inputs,
                ],
                output_transforms=[
                    *data_config.model_transforms.outputs,
                    transforms.Unnormalize(
                        norm_stats, use_quantiles=data_config.use_quantile_norm
                    ),
                    *data_config.data_transforms.outputs,
                    *repack_transforms.outputs,
                ],
            )

    elif model_type == SupportedModel.MLP_POLICY:
        from .embodiment.mlp_policy import MLPPolicy

        model = MLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
        )
    elif model_type == SupportedModel.GR00T:
        from pathlib import Path

        from rlinf.utils.patcher import Patcher

        Patcher.clear()
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EmbodimentTag",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
        )
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        )
        Patcher.apply()

        from gr00t.experiment.data_config import load_data_config

        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

        from .embodiment.gr00t_action_model import GR00T_N1_5_ForRLActionPrediction

        if cfg.embodiment_tag == "libero_franka":
            data_config = load_data_config(
                "rlinf.models.embodiment.gr00t.modality_config:LiberoFrankaDataConfig"
            )
        elif cfg.embodiment_tag == "maniskill_widowx":
            data_config = load_data_config(
                "rlinf.models.embodiment.gr00t.modality_config:ManiskillWidowXDataConfig"
            )
        else:
            raise ValueError(f"Invalid embodiment tag: {cfg.embodiment_tag}")
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        # The transformer rigisteration is done in gr00t/model/gr00t_n1.py
        model_path = Path(model_path)
        if not model_path.exists():
            # raise error or it triggers auto download from hf(It's cool but we don't have internet connection.)
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=cfg.embodiment_tag,  # This tag determines the state encoder and action head to use
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=cfg.denoising_steps,
            output_action_chunks=cfg.num_action_chunks,
            obs_converter_type=cfg.obs_converter_type,  # TODO(lx): unify the embodiment data format and obs converter
            tune_visual=False,
            tune_llm=False,
            rl_head_config=cfg.rl_head_config,
        )
        model.to(torch_dtype)
        if cfg.rl_head_config.add_value_head:
            # reinitialize the value head after model loading, or there are nan values in the value head after model loading.
            model.action_head.value_head._init_weights()

        if cfg.rl_head_config.disable_dropout:
            replace_dropout_with_identity(model)
    else:
        return None
    if torch.cuda.is_available():
        model = model.cuda()

    if getattr(cfg, "is_lora", False):
        from peft import LoraConfig, PeftModel, get_peft_model

        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            lora_config = LoraConfig(
                r=getattr(cfg, "lora_rank", 32),
                lora_alpha=getattr(cfg, "lora_rank", 32),
                lora_dropout=0.0,
                target_modules=[
                    "proj",
                    "qkv",
                    "fc1",
                    "fc2",  # vision
                    "q",
                    "kv",
                    "fc3",
                    "out_proj",  # project
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",  # llm
                ],
                init_lora_weights="gaussian",
            )
            if model_type == SupportedModel.OPENPI:
                module_to_lora = model.paligemma_with_expert.paligemma
                module_to_lora = get_peft_model(module_to_lora, lora_config)
                tag_vlm_subtree(model, False)
                tag_vlm_subtree(module_to_lora, True)
                model.paligemma_with_expert.paligemma = module_to_lora
            else:
                model = get_peft_model(model, lora_config)
        else:
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

    return model


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)
