"""Application level configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict


def _env_bool(name: str, default: bool) -> bool:
    """Read boolean env var with safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read integer env var with safe fallback."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    """Read string env var with safe fallback."""
    raw = os.environ.get(name)
    return raw if raw is not None and raw.strip() else default


@dataclass(frozen=True)
class RenderSettings:
    """Settings that describe how translated text should be rendered."""

    font_name: str = "MTOAstroCity.ttf"
    alignment: str = "center"
    min_font_size: int = 12
    max_font_size: int = 40
    color: str = "#000000"
    upper_case: bool = True
    outline: bool = True

    @property
    def font_path(self) -> str:
        """Absolute path for the font inside project."""
        return f"fonts/{self.font_name}"


@dataclass(frozen=True)
class InpaintingSettings:
    """Settings forwarded to the inpainting model."""

    device: str = "cpu"
    zits_wireframe: bool = True
    hd_strategy: str = "Resize"
    hd_strategy_crop_margin: int = 512
    hd_strategy_resize_limit: int = 960
    hd_strategy_crop_trigger_size: int = 512

    def to_config(self) -> Dict[str, object]:
        """Return a dict config understood by LaMa."""
        return {
            "zits_wireframe": self.zits_wireframe,
            "hd_strategy": self.hd_strategy,
            "hd_strategy_crop_margin": self.hd_strategy_crop_margin,
            "hd_strategy_resize_limit": self.hd_strategy_resize_limit,
            "hd_strategy_crop_trigger_size": self.hd_strategy_crop_trigger_size,
        }


@dataclass(frozen=True)
class DetectionSettings:
    """Paths/devices used by YOLO detectors."""

    bubble_model_path: str = "models/detection/comic-speech-bubble-detector.pt"
    text_seg_model_path: str = "models/detection/comic-text-segmenter.pt"
    text_detect_model_path: str = "models/detection/manga-text-detector.pt"
    device: str = "cpu"


@dataclass(frozen=True)
class OCRSettings:
    """Settings for OCR processing."""

    use_gpu: bool = True


@dataclass(frozen=True)
class TranslatorSettings:
    """Translator configuration."""

    provider: str = "gemini"
    openai_model: str = "gpt-4"
    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-2.5-flash"
    grok_model: str = "grok-3-mini"

    @property
    def provider_key_env(self) -> str:
        """Env var name expected for the active provider."""
        mapping = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "grok": "XAI_API_KEY",
        }
        return mapping.get(self.provider.lower(), "")


@dataclass(frozen=True)
class AppSettings:
    """Top level immutable settings object for the service."""

    language_codes: Dict[str, str] = field(
        default_factory=lambda: {
            "zh": "Chinese",
            "en": "English",
            "vi": "Vietnamese",
        }
    )
    render: RenderSettings = field(default_factory=RenderSettings)
    inpainting: InpaintingSettings = field(default_factory=InpaintingSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    ocr: OCRSettings = field(default_factory=OCRSettings)
    translator: TranslatorSettings = field(default_factory=TranslatorSettings)


def _get_device(env_name: str, default: str) -> str:
    """
    Get device setting with support for global USE_GPU override.
    
    If USE_GPU is set to True, it will override individual device settings
    and return 'cuda' (or 'mps' on Mac). Otherwise, uses the specific env var.
    """
    use_gpu_global = _env_bool("USE_GPU", False)
    if use_gpu_global:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        # Fallback to CPU if GPU not available
        return "cpu"
    
    # Use specific device setting
    return _env_str(env_name, default)


@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    """Load settings once from environment."""
    render = RenderSettings(
        font_name=_env_str("RENDER_FONT_NAME", "MTOAstroCity.ttf"),
        alignment=_env_str("TEXT_ALIGNMENT", "center"),
        min_font_size=_env_int("TEXT_MIN_FONT_SIZE", 12),
        max_font_size=_env_int("TEXT_MAX_FONT_SIZE", 40),
        color=_env_str("TEXT_COLOR", "#000000"),
        upper_case=_env_bool("TEXT_UPPER_CASE", True),
        outline=_env_bool("TEXT_OUTLINE", True),
    )

    # Check global USE_GPU first, then specific settings
    use_gpu_global = _env_bool("USE_GPU", False)
    
    inpainting = InpaintingSettings(
        device=_get_device("INPAINT_DEVICE", "cpu"),
        zits_wireframe=_env_bool("INPAINT_ZITS_WIREFRAME", True),
        hd_strategy=_env_str("INPAINT_HD_STRATEGY", "Resize"),
        hd_strategy_crop_margin=_env_int("INPAINT_CROP_MARGIN", 512),
        hd_strategy_resize_limit=_env_int("INPAINT_RESIZE_LIMIT", 960),
        hd_strategy_crop_trigger_size=_env_int("INPAINT_CROP_TRIGGER_SIZE", 512),
    )

    detection = DetectionSettings(
        bubble_model_path=_env_str(
            "MODEL_BUBBLE_PATH", "models/detection/comic-speech-bubble-detector.pt"
        ),
        text_seg_model_path=_env_str(
            "MODEL_TEXT_SEG_PATH", "models/detection/comic-text-segmenter.pt"
        ),
        text_detect_model_path=_env_str(
            "MODEL_TEXT_DETECT_PATH", "models/detection/manga-text-detector.pt"
        ),
        device=_get_device("DETECTION_DEVICE", "cpu"),
    )

    # OCR GPU setting: use global USE_GPU or specific OCR_USE_GPU
    ocr_use_gpu = use_gpu_global or _env_bool("OCR_USE_GPU", False)
    ocr = OCRSettings(use_gpu=ocr_use_gpu)

    translator = TranslatorSettings(
        provider=_env_str("TRANSLATOR_PROVIDER", "gemini"),
        openai_model=_env_str("OPENAI_TRANSLATION_MODEL", "gpt-4"),
        groq_model=_env_str("GROQ_TRANSLATION_MODEL", "llama-3.1-8b-instant"),
        gemini_model=_env_str("GEMINI_TRANSLATION_MODEL", "gemini-2.5-flash"),
        grok_model=_env_str("GROK_TRANSLATION_MODEL", "grok-3-mini"),
    )

    return AppSettings(
        render=render,
        inpainting=inpainting,
        detection=detection,
        ocr=ocr,
        translator=translator,
    )

