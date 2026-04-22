from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


DEFAULT_DDCOLOR_MODEL_NAME = "pytorch_model.pt"
DEFAULT_DDCOLOR_MODEL_SIZE = "large"
DEFAULT_DDCOLOR_INPUT_SIZE = 512

_BACKEND_CACHE: Dict[Tuple[str, str, int, str], "DDColorBackend"] = {}


class DDColorModelNotFoundError(FileNotFoundError):
    """Raised when DDColor model assets are missing."""


@dataclass
class DDColorBackend:
    model: object
    pipeline: object
    model_path: Path
    device: str
    input_size: int
    model_size: str


def get_ddcolor_model_path(explicit_path: os.PathLike | str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    env_path = os.environ.get("DDCOLOR_MODEL_PATH", "").strip()
    if env_path:
        return Path(env_path)

    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "models" / "colorization" / "ddcolor" / DEFAULT_DDCOLOR_MODEL_NAME


def _build_backend(
    *,
    model_path: Path,
    device: str = "cpu",
    input_size: int = DEFAULT_DDCOLOR_INPUT_SIZE,
    model_size: str = DEFAULT_DDCOLOR_MODEL_SIZE,
) -> DDColorBackend:
    import torch

    from .ddcolor_vendor import DDColor, ColorizationPipeline, build_ddcolor_model

    torch_device = torch.device(device)
    model = build_ddcolor_model(
        DDColor,
        model_path=str(model_path),
        input_size=input_size,
        model_size=model_size,
        device=torch_device,
    )
    pipeline = ColorizationPipeline(
        model,
        input_size=input_size,
        device=torch_device,
    )
    return DDColorBackend(
        model=model,
        pipeline=pipeline,
        model_path=model_path,
        device=str(torch_device),
        input_size=input_size,
        model_size=model_size,
    )


def load_ddcolor_backend(
    *,
    model_path: os.PathLike | str | None = None,
    device: str = "cpu",
    input_size: int = DEFAULT_DDCOLOR_INPUT_SIZE,
    model_size: str = DEFAULT_DDCOLOR_MODEL_SIZE,
    force_reload: bool = False,
) -> DDColorBackend:
    resolved_model_path = get_ddcolor_model_path(model_path)
    if not resolved_model_path.exists():
        raise DDColorModelNotFoundError(
            f"DDColor model not found: {resolved_model_path}"
        )

    cache_key = (
        str(resolved_model_path.resolve()),
        str(device),
        int(input_size),
        str(model_size),
    )
    if force_reload or cache_key not in _BACKEND_CACHE:
        _BACKEND_CACHE[cache_key] = _build_backend(
            model_path=resolved_model_path,
            device=device,
            input_size=input_size,
            model_size=model_size,
        )
    return _BACKEND_CACHE[cache_key]


def run_ddcolor_inference(
    img_bgr: np.ndarray,
    *,
    backend: DDColorBackend | None = None,
    model_path: os.PathLike | str | None = None,
    device: str = "cpu",
    input_size: int = DEFAULT_DDCOLOR_INPUT_SIZE,
    model_size: str = DEFAULT_DDCOLOR_MODEL_SIZE,
) -> np.ndarray:
    backend_obj = backend or load_ddcolor_backend(
        model_path=model_path,
        device=device,
        input_size=input_size,
        model_size=model_size,
    )
    output = backend_obj.pipeline.process(img_bgr)
    return np.asarray(output)
