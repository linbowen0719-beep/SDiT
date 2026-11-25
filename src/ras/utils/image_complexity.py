import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is required for full features
    raise ImportError("torch is required to compute deep complexity features.") from exc


def _to_numpy_gray(image: Image.Image) -> np.ndarray:
    """Convert image to grayscale numpy array in [0, 1]."""
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return arr


def _shannon_entropy(gray: np.ndarray) -> float:
    """Compute normalized Shannon entropy of an image."""
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0.0, 1.0))
    probs = hist.astype(np.float32)
    total = probs.sum()
    if total == 0:
        return 0.0
    probs /= total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / math.log2(256))


def _sobel_edge_density(gray: np.ndarray) -> float:
    """Approximate edge density using Sobel filters."""
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    grad_x = _convolve2d(gray, kernel_x)
    grad_y = _convolve2d(gray, kernel_y)
    magnitude = np.hypot(grad_x, grad_y)
    threshold = magnitude.mean() + magnitude.std()
    if threshold == 0:
        return 0.0
    density = float((magnitude > threshold).mean())
    return density


def _variance_of_laplacian(gray: np.ndarray) -> float:
    """Measure focus/structure via Laplacian variance."""
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = _convolve2d(gray, lap_kernel)
    return float(lap.var())


def _high_frequency_ratio(gray: np.ndarray, high_freq_ratio: float = 0.25) -> float:
    """Compute high frequency energy ratio in the Fourier spectrum."""
    h, w = gray.shape
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    total_energy = np.sum(magnitude ** 2)
    if total_energy <= 0:
        return 0.0

    center_y, center_x = h // 2, w // 2
    radius = int(min(h, w) * (1.0 - high_freq_ratio) * 0.5)
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 >= radius ** 2
    high_freq_energy = np.sum((magnitude ** 2)[mask])
    return float(high_freq_energy / total_energy)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Basic 2D convolution with zero padding (valid)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(image)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            patch = padded[y : y + kh, x : x + kw]
            out[y, x] = np.sum(patch * kernel)
    return out


@dataclass
class ComplexityMetrics:
    entropy: float
    edge_density: float
    laplacian_variance: float
    high_frequency_ratio: float
    clip_attention_entropy: Optional[float] = None
    clip_patch_variance: Optional[float] = None
    reconstruction_error: Optional[float] = None
    combined_score: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ImageComplexityScorer:
    """Estimate image generation complexity using handcrafted and deep metrics.

    The scorer intentionally avoids explicit face or text detectors. Instead,
    it correlates high complexity with rich local details, structural edges,
    frequency content, and deep backbone uncertainty.
    """

    DEFAULT_WEIGHTS = {
        "entropy": 0.2,
        "edge_density": 0.2,
        "laplacian_variance": 0.15,
        "high_frequency_ratio": 0.15,
        "clip_attention_entropy": 0.15,
        "clip_patch_variance": 0.15,
        "reconstruction_error": 0.2,
    }

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        load_clip: bool = True,
        load_vae: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        vae_model_name: str = "stabilityai/sd-vae-ft-ema",
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        self.clip_processor = None
        self.clip_model = None
        if load_clip:
            self._init_clip(clip_model_name)

        self.vae = None
        if load_vae:
            self._init_vae(vae_model_name)

    def _init_clip(self, model_name: str) -> None:
        try:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers is required to compute CLIP-based complexity metrics."
            ) from exc

        self.clip_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(self.device)
        self.clip_model.eval()

    def _init_vae(self, model_name: str) -> None:
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "diffusers is required to compute VAE reconstruction-based metrics."
            ) from exc

        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder=None).to(self.device)
        self.vae.eval()

    def score(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        return_metrics: bool = True,
        normalize: bool = True,
    ) -> Union[float, ComplexityMetrics]:
        """Compute complexity score for a single image.

        Args:
            image: PIL image, numpy array, or path to image file.
            return_metrics: if True, return ComplexityMetrics dataclass.
            normalize: if True, normalize metrics before combining.
        """
        pil_image = self._ensure_pil(image)
        gray = _to_numpy_gray(pil_image)

        entropy = _shannon_entropy(gray)
        edge_density = _sobel_edge_density(gray)
        lap_var = _variance_of_laplacian(gray)
        high_freq = _high_frequency_ratio(gray)

        clip_entropy, clip_patch_var = (None, None)
        if self.clip_model is not None:
            clip_entropy, clip_patch_var = self._clip_complexity(pil_image)

        recon_error = None
        if self.vae is not None:
            recon_error = self._reconstruction_error(pil_image)

        metrics = ComplexityMetrics(
            entropy=entropy,
            edge_density=edge_density,
            laplacian_variance=lap_var,
            high_frequency_ratio=high_freq,
            clip_attention_entropy=clip_entropy,
            clip_patch_variance=clip_patch_var,
            reconstruction_error=recon_error,
        )

        metrics.combined_score = self._combine_metrics(metrics, normalize=normalize)
        return metrics if return_metrics else metrics.combined_score

    def _clip_complexity(self, image: Image.Image) -> Tuple[Optional[float], Optional[float]]:
        assert self.clip_model is not None and self.clip_processor is not None
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs, output_hidden_states=True, output_attentions=True)

        attentions = outputs.attentions  # tuple of (layers, batch, heads, tokens, tokens)
        if attentions is None:
            return None, None

        # Concatenate attentions across layers and heads, compute entropy per token, then average.
        att_stack = torch.stack(attentions, dim=0)  # (layers, batch, heads, seq, seq)
        att_mean = att_stack.mean(dim=(0, 2))  # (batch, seq, seq)
        att_probs = att_mean / att_mean.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        att_entropy = -(att_probs * (att_probs.clamp_min(1e-6).log())).sum(dim=-1).mean()
        attention_entropy = att_entropy.item() / math.log(att_probs.shape[-1])

        # Patch variance: exclude CLS token (index 0)
        hidden_states = outputs.hidden_states[-1]  # (batch, seq, dim)
        patch_tokens = hidden_states[:, 1:, :]  # drop CLS
        patch_var = patch_tokens.var(dim=1).mean().item()

        return attention_entropy, float(patch_var)

    def _reconstruction_error(self, image: Image.Image) -> float:
        assert self.vae is not None
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.vae.encode(tensor).latent_dist.sample() * self.vae.config.scaling_factor
            recon = self.vae.decode(latent).sample
        error = torch.nn.functional.mse_loss(recon, tensor).item()
        return float(error)

    @staticmethod
    def _ensure_pil(image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray((image * 255).astype(np.uint8), mode="L").convert("RGB")
            if image.shape[-1] == 3:
                arr = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
                return Image.fromarray(arr, mode="RGB")
            raise ValueError("Unsupported numpy array shape for image conversion.")
        raise TypeError(f"Unsupported image input type: {type(image)}")

    def _combine_metrics(self, metrics: ComplexityMetrics, normalize: bool = True) -> float:
        values = {}
        for key, weight in self.weights.items():
            value = getattr(metrics, key, None)
            if value is None:
                continue
            values[key] = self._normalize_metric(key, value) if normalize else value

        if not values:
            return 0.0

        weighted = sum(values[k] * self.weights.get(k, 0.0) for k in values)
        total_weight = sum(self.weights.get(k, 0.0) for k in values)
        if total_weight == 0:
            return 0.0
        return float(weighted / total_weight)

    @staticmethod
    def _normalize_metric(name: str, value: float) -> float:
        """Map raw metric values to a comparable [0, 1] range."""
        if name in {"entropy", "clip_attention_entropy"}:
            return float(np.clip(value, 0.0, 1.0))
        if name in {"edge_density", "high_frequency_ratio"}:
            return float(np.clip(value, 0.0, 1.0))
        if name == "laplacian_variance":
            return float(np.tanh(value))
        if name == "reconstruction_error":
            return float(1.0 - math.exp(-value))
        if name == "clip_patch_variance":
            return float(np.tanh(value))
        return float(value)


def score_image_paths(
    paths: Sequence[Union[str, Path]],
    scorer: Optional[ImageComplexityScorer] = None,
    top_k: Optional[int] = None,
) -> Dict[str, ComplexityMetrics]:
    """Convenience helper to score multiple image paths and optionally select Top-K."""
    scorer = scorer or ImageComplexityScorer()
    results: Dict[str, ComplexityMetrics] = {}
    for path in paths:
        metrics = scorer.score(path, return_metrics=True)
        results[str(path)] = metrics

    if top_k is None or top_k <= 0 or top_k >= len(results):
        return results

    sorted_items = sorted(results.items(), key=lambda item: item[1].combined_score or 0.0, reverse=True)
    return dict(sorted_items[:top_k])
