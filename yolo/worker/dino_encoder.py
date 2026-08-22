"""DINOv2-small encoder for torso-crop team embeddings."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class EmbeddingEncoder(Protocol):
    """Maps BGR crops to L2-normalized embedding rows."""

    def encode_bgr(self, crops: list[np.ndarray]) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize rows of a 1-D or 2-D array."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / max(norm, eps)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


class DinoEncoder:
    """facebook/dinov2-small CLS embeddings, batched at finalize time."""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def encode_bgr(self, crops: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)

        rows: list[np.ndarray] = []
        torch = self._torch
        for start in range(0, len(crops), batch_size):
            chunk = crops[start : start + batch_size]
            rgb = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in chunk]
            inputs = self.processor(images=rgb, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state[:, 0]
            rows.append(hidden.float().cpu().numpy())
        return l2_normalize(np.concatenate(rows, axis=0))
