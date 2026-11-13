"""Public model interface: image in, 98 landmarks per detected face out.

    model = LandmarkModel("weights/modelo.pt", device="cpu")
    landmarks = model.predict(image_rgb)   # (N, 98, 2) in pixels of the full image

Pipeline per image: YuNet face boxes -> square crop (1.25 x longest side, same rule as
training) -> landmark model -> inverse crop transform back to the full image.
Everything is loaded from this folder; nothing is downloaded.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from utils import (FaceDetector, crop_matrix, find_yunet, invert, load_checkpoint,
                   predict_crops, transform_points, warp)

ROOT = Path(__file__).resolve().parent


class LandmarkModel:
    def __init__(self, weights: str, device: str = "cpu"):
        """Carrega o detector de faces e o modelo de landmarks.
        weights: caminho dos pesos (pesos/modelo.pt);
        device: "cpu" ou "cuda"."""
        self.device = torch.device(device)
        self.model, ckpt = load_checkpoint(weights, self.device)
        self.size = ckpt["input_size"]
        self.margin = ckpt["margin"]
        self.detector = FaceDetector(find_yunet(ROOT))
        # média da previsão com a da face espelhada (melhora NME e EAR na validação)
        self.flip_tta = True

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Detecta as faces da imagem e prevê os 98 landmarks de cada uma.
        image: imagem inteira, RGB, uint8, formato (altura, largura, 3)
        retorna: array float (N, 98, 2), uma linha por face detectada,
                 com (x, y) em pixels da imagem inteira, na ordem dos
                 landmarks do WFLW; N = 0 se nenhuma face for encontrada
        """
        image = np.ascontiguousarray(image)
        boxes = self.detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return self.predict_boxes(image, boxes[:, :4])

    def predict_boxes(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Landmarks das faces nos retângulos dados (x_min, y_min, x_max, y_max) de uma imagem RGB."""
        mats = [crop_matrix(b, self.size, self.margin) for b in boxes]
        crops = [warp(image, m, self.size) for m in mats]
        pts = predict_crops(self.model, crops, self.device, self.flip_tta)
        out = [transform_points(p.astype(np.float64), invert(m))
               for p, m in zip(pts, mats)]
        return np.stack(out).astype(np.float32) if out else np.zeros((0, 98, 2), np.float32)
