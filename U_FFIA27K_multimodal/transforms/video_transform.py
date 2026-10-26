import math
import random
from typing import Union, List
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ImageToPIL:
    """Convert one RGB image in [C, H, W] or [H, W, C] format to PIL."""
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")

            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            elif image.shape[-1] != 3:
                raise ValueError(f"Expected RGB channel dimension with size 3, got shape {tuple(image.shape)}")

            return TF.to_pil_image(image)

        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")

            if image.shape[0] != 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            elif image.shape[0] != 3:
                raise ValueError(f"Expected RGB channel dimension with size 3, got shape {tuple(image.shape)}")

            return TF.to_pil_image(image)

        return image


class ConsistentVideoTransform:
    """
    Temporally Consistent Video Transform Pipeline (Clip-Level Synchronized Augmentation).
    Guarantees physical kinematic temporal coherence across all T frames in a video clip:
      - Random Horizontal Flip: Decided once per clip, applied identically to all T frames.
      - Random Rotation: Angle sampled once per clip [-15.0, 15.0], applied identically to all T frames.
      - Color Jitter (Brightness/Contrast): Factors sampled once per clip [0.85, 1.15], applied identically to all T frames.
      - Random Erasing / Cutout: Rectangular region sampled once per clip, applied identically to all T frames.
      - Bilinear Resize: Standardized HxW.
      - ImageNet Normalization.

    Accepts:
      - 4D np.ndarray [T, H, W, C] or [T, C, H, W]
      - List/Tuple of 3D frames [frame_0, frame_1, ...]
      - Single 3D frame [H, W, C] or [C, H, W]
    Returns:
      - torch.Tensor [T, C, H, W] or [C, H, W]
    """
    def __init__(
        self,
        image_size: int = 224,
        is_train: bool = True,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        erase_prob: float = 0.3,
        erase_scale: tuple = (0.05, 0.15),
        erase_ratio: tuple = (0.5, 2.0),
        erase_value: float = 0.0,
    ) -> None:
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.mean = mean
        self.std = std
        self.erase_prob = float(erase_prob)
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self.erase_value = float(erase_value)
        self._to_pil = ImageToPIL()

    def __call__(
        self,
        frames: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        is_single_frame = False
        if isinstance(frames, (list, tuple)):
            frame_list = list(frames)
        elif isinstance(frames, (np.ndarray, torch.Tensor)):
            if frames.ndim == 3:
                is_single_frame = True
                frame_list = [frames]
            elif frames.ndim == 4:
                frame_list = [frames[i] for i in range(frames.shape[0])]
            else:
                raise ValueError(f"Expected 3D or 4D video tensor/array, got ndim={frames.ndim}")
        else:
            is_single_frame = True
            frame_list = [frames]

        # Sample augmentation parameters ONCE per video clip (temporal synchronization)
        erase_box = None
        if self.is_train:
            do_flip = (random.random() < 0.5)
            rot_angle = random.uniform(-15.0, 15.0)
            brightness_factor = random.uniform(0.85, 1.15)
            contrast_factor = random.uniform(0.85, 1.15)

            # Sample Random Erasing bounding box ONCE per clip
            if random.random() < self.erase_prob:
                img_h, img_w = self.image_size, self.image_size
                area = img_h * img_w
                for _ in range(10):
                    target_area = random.uniform(self.erase_scale[0], self.erase_scale[1]) * area
                    aspect_ratio = math.exp(random.uniform(math.log(self.erase_ratio[0]), math.log(self.erase_ratio[1])))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if h < img_h and w < img_w:
                        top = random.randint(0, img_h - h)
                        left = random.randint(0, img_w - w)
                        erase_box = (top, left, h, w)
                        break

        transformed_tensors = []
        for frame in frame_list:
            pil_img = self._to_pil(frame)
            pil_img = TF.resize(
                pil_img,
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR
            )
            if self.is_train:
                pil_img = TF.adjust_brightness(pil_img, brightness_factor)
                pil_img = TF.adjust_contrast(pil_img, contrast_factor)
                if do_flip:
                    pil_img = TF.hflip(pil_img)
                pil_img = TF.rotate(
                    pil_img,
                    rot_angle,
                    interpolation=InterpolationMode.BILINEAR
                )

            t_img = TF.to_tensor(pil_img)
            t_img = TF.normalize(t_img, self.mean, self.std)

            # Apply clip-synchronized Random Erasing to the normalized tensor
            if self.is_train and erase_box is not None:
                top, left, h, w = erase_box
                t_img = TF.erase(t_img, top, left, h, w, v=self.erase_value)

            transformed_tensors.append(t_img)

        if is_single_frame:
            return transformed_tensors[0]
        return torch.stack(transformed_tensors, dim=0)

    @classmethod
    def get_transforms(
        cls,
        image_size: int = 224,
        erase_prob: float = 0.3,
        erase_scale: tuple = (0.05, 0.15),
        erase_ratio: tuple = (0.5, 2.0),
        erase_value: float = 0.0,
    ):
        return {
            "train": cls(
                image_size=image_size,
                is_train=True,
                erase_prob=erase_prob,
                erase_scale=erase_scale,
                erase_ratio=erase_ratio,
                erase_value=erase_value,
            ),
            "val": cls(image_size=image_size, is_train=False),
            "test": cls(image_size=image_size, is_train=False),
        }
