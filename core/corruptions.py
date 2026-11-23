"""CIFAR-C-style corruption on raw float RGB copies, before model normalization."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def check_image(image):
    if (image.ndim != 3 or image.shape[0] != 3 or not np.issubdtype(image.dtype, np.floating)
            or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise ValueError("Expected finite float RGB CHW in [0, 1]")


def gaussian_noise(image, severity, rng):
    sigma = (0.04, 0.06, 0.08, 0.09, 0.10)[severity - 1]
    return image + rng.normal(0, sigma, image.shape)


def defocus_blur(image, severity, rng):
    radius, alias = ((0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1))[severity - 1]
    coordinates = np.arange(-8, 9)
    distance = coordinates[:, None] ** 2 + coordinates[None, :] ** 2
    kernel = (distance <= radius**2).astype(np.float32)
    kernel /= kernel.sum()
    kernel = gaussian_filter(kernel, alias, radius=1, mode="mirror")
    return convolve(image, kernel[None, :, :], mode="mirror")


def low_light(image, severity, rng):
    return image * (1 - 0.12 * severity)


def low_contrast(image, severity, rng):
    center = image.mean(axis=(1, 2), keepdims=True)
    return center + (image - center) * (1 - 0.15 * severity)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise, "defocus_blur": defocus_blur,
    "low_light": low_light, "low_contrast": low_contrast,
}
PRIVATE_ROLES = {"private_train", "client_local_val"}


@dataclass(frozen=True)
class Corruption:
    kind: str
    severity: int
    ratio: float

    def __post_init__(self):
        if self.kind not in CORRUPTIONS or self.severity not in range(1, 6) or not 0 <= self.ratio <= 1:
            raise ValueError("Invalid corruption configuration")

    def apply(self, image, *, role, seed):
        if role not in PRIVATE_ROLES:
            raise ValueError(f"Corruption injection forbidden for {role}")
        check_image(image)
        result = CORRUPTIONS[self.kind](image.copy(), self.severity, np.random.default_rng(seed))
        result = np.clip(result, 0, 1).astype(np.float32)
        check_image(result)
        return result
