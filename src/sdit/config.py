"""Explicit, validated settings; no model imports or downloads at import time."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

MODEL_IDS = {
    "lumina": "Alpha-VLLM/Lumina-Next-SFT-diffusers",
    "hunyuan": "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
}

@dataclass(frozen=True)
class InferenceConfig:
    model: str = "lumina"
    method: str = "sdit"
    steps: int = 30
    warmup_steps: int | None = None
    rho: float = 0.2
    hops: int = 1
    anchors: int = 1
    backend: str = "sdpa"
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 4.0
    metric: str = "paper"
    starvation_scale: float = 0.1
    quickshift_kernel_size: float = 2.3
    quickshift_max_dist: float = 6.0
    quickshift_ratio: float = 0.5
    teacache_threshold: float = 0.3
    analysis: bool = False

    def __post_init__(self):
        if self.model not in MODEL_IDS:
            raise ValueError(f"model must be one of {list(MODEL_IDS)}")
        if self.method not in ("sdit", "dense", "temporal", "ras", "teacache"):
            raise ValueError("method must be sdit, dense, temporal, ras, or teacache")
        if self.model == "hunyuan" and self.method == "ras":
            raise ValueError("The release RAS baseline supports Lumina only")
        for name in ("steps", "height", "width", "hops", "anchors"):
            v = getattr(self, name)
            if type(v) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.steps < 1 or self.hops < 0 or self.anchors < 0:
            raise ValueError("steps must be positive; hops and anchors must be nonnegative")
        if self.warmup_steps is not None and (type(self.warmup_steps) is not int or not 1 <= self.warmup_steps <= self.steps):
            raise ValueError("warmup_steps must be an integer in [1, steps]")
        if self.backend != "sdpa":
            raise ValueError("This release supports only the sdpa backend")
        if self.height < 256 or self.width < 256 or self.height % 16 or self.width % 16:
            raise ValueError("height and width must be multiples of 16 and at least 256")
        for name in ("rho", "guidance_scale", "starvation_scale", "quickshift_kernel_size", "quickshift_max_dist", "quickshift_ratio", "teacache_threshold"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        if self.guidance_scale <= 1:
            raise ValueError("This release requires classifier-free guidance_scale > 1")
        if min(self.quickshift_kernel_size, self.quickshift_max_dist) <= 0:
            raise ValueError("Quickshift kernel size and max distance must be positive")
        if self.starvation_scale < 0 or self.quickshift_ratio < 0 or self.teacache_threshold < 0:
            raise ValueError("starvation scale, Quickshift ratio and TeaCache threshold must be nonnegative")
        if self.metric not in ("paper", "std"):
            raise ValueError("Unsupported score metric")
        if type(self.analysis) is not bool:
            raise ValueError("analysis must be a boolean")

    @property
    def warmup(self):
        return self.warmup_steps if self.warmup_steps is not None else math.ceil(0.2 * self.steps)

    def to_dict(self):
        return {**asdict(self), "warmup_steps": self.warmup}

    @classmethod
    def load(cls, path=None, **overrides):
        data = json.loads(Path(path).read_text()) if path else {}
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        allowed = {f.name for f in fields(cls)}
        unknown = data.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)
