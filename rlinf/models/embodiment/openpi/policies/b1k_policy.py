import dataclasses

import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        wrist_images = data.get("observation/wrist_image", None)
        if wrist_images is None:
            wrist_left = np.zeros_like(base_image)
            wrist_right = np.zeros_like(base_image)
        else:
            wrist_images = np.asarray(wrist_images)
            if wrist_images.ndim == 4:
                wrist_left = _parse_image(wrist_images[0])
                wrist_right = _parse_image(wrist_images[1]) if wrist_images.shape[0] > 1 else np.zeros_like(wrist_left)
            else:
                wrist_left = _parse_image(wrist_images)
                wrist_right = np.zeros_like(wrist_left)

        if self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, wrist_left, wrist_right)
        else:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, wrist_left, wrist_right)

        image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "tokenized_prompt" in data:
            inputs["tokenized_prompt"] = data["tokenized_prompt"]
        if "tokenized_prompt_mask" in data:
            inputs["tokenized_prompt_mask"] = data["tokenized_prompt_mask"]

        return inputs


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
