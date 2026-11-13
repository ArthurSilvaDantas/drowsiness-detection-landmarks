"""Código compartilhado por train.py e predict.py.

O passo a passo de cada parte, com figuras e explicações, está nos notebooks:
    dados e recorte     -> notebooks/01_exploracao_dados.ipynb, 02_preprocessamento.ipynb
    modelo e perda      -> notebooks/03_modelo.ipynb
    treino              -> notebooks/04_treino.ipynb
    avaliação           -> notebooks/05_avaliacao.ipynb

Seções deste arquivo:
    1. WFLW: leitura das anotações, Mirror98 e divisão treino/validação
    2. Geometria do recorte e augmentation
    3. Cache de recortes e Dataset do PyTorch
    4. Modelo: ResNet-18 + cabeça deconv + soft-argmax (Seções 4.3-4.4)
    5. Perda nas coordenadas
    6. Métricas: NME e EAR (Seções 6 e 8)
    7. Detector de faces YuNet e inferência (Seções 7.2 e 11.2)
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

# =========================================================================================
# 1. WFLW
# =========================================================================================
NUM_LANDMARKS = 98
ATTRIBUTES = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]
REGIONS = {
    "contour": list(range(0, 33)),
    "eyebrows": list(range(33, 51)),
    "nose": list(range(51, 60)),
    "eyes": list(range(60, 76)),
    "mouth": list(range(76, 96)),
    "pupils": [96, 97],
}
EYE_POINTS = list(range(60, 76))  # 60-67: olho direito da pessoa (lado esquerdo da imagem)
OUTER_EYE_CORNERS = (60, 72)      # normalização padrão do NME no WFLW


@dataclass
class Face:
    landmarks: np.ndarray   # (98, 2) pixels da imagem original
    box: np.ndarray         # (4,) x_min, y_min, x_max, y_max
    attributes: np.ndarray  # (6,) 0 = normal, 1 = difícil
    image: Path


def read_faces(data_dir, split: str) -> list[Face]:
    """Uma entrada por FACE anotada (linha de 207 campos), não por imagem."""
    data_dir = Path(data_dir)
    path = (data_dir / "WFLW_annotations" / "list_98pt_rect_attr_train_test"
            / f"list_98pt_rect_attr_{split}.txt")
    faces = []
    for line in path.read_text().splitlines():
        f = line.split()
        faces.append(Face(np.array(f[:196], np.float32).reshape(NUM_LANDMARKS, 2),
                          np.array(f[196:200], np.float32),
                          np.array(f[200:206], np.int64),
                          data_dir / "WFLW_images" / f[206]))
    return faces


def read_mirror(data_dir) -> np.ndarray:
    text = (Path(data_dir) / "WFLW_annotations" / "Mirror98.txt").read_text()
    return np.array(text.strip().split(","), dtype=np.int64)


def split_train_val(faces: list[Face], val_fraction: float, seed: int):
    """Validação separada do TREINO oficial, por imagem (faces da mesma foto ficam juntas)."""
    images = sorted({str(f.image) for f in faces})
    rng = np.random.default_rng(seed)
    val_images = set(rng.choice(images, int(round(val_fraction * len(images))), replace=False).tolist())
    train_idx = [i for i, f in enumerate(faces) if str(f.image) not in val_images]
    val_idx = [i for i, f in enumerate(faces) if str(f.image) in val_images]
    return np.array(train_idx), np.array(val_idx)


# =========================================================================================
# 2. Geometria do recorte
# =========================================================================================
def crop_matrix(box, size: int, margin: float = 1.25) -> np.ndarray:
    """Matriz 2x3: quadrado com o centro do retângulo, lado = margin x maior lado, -> size x size.

    p' = s (p - c) + size/2, com s = size / lado.
    """
    x0, y0, x1, y1 = [float(v) for v in box]
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    s = size / (margin * max(x1 - x0, y1 - y0))
    t = size / 2 - s * c
    return np.array([[s, 0, t[0]], [0, s, t[1]]], dtype=np.float64)


def invert(m: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(m.astype(np.float64))


def transform_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    return points @ m[:, :2].T + m[:, 2]


def warp(image: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpAffine(image, m.astype(np.float64), (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def augment_matrix(src_center, src_to_out_scale: float, out_size: int, angle=0.0, scale=1.0,
                   shift=(0.0, 0.0)) -> np.ndarray:
    """Rotação + escala + translação numa única matriz (escala > 1 afasta a face)."""
    cx, cy = src_center
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, src_to_out_scale / scale)
    m[:, 2] += np.array([out_size / 2 - cx, out_size / 2 - cy])
    m[:, 2] += np.asarray(shift, dtype=np.float64) * out_size
    return m


def flip_points(points: np.ndarray, width: int, mirror: np.ndarray) -> np.ndarray:
    """Espelhamento: x' = W - 1 - x E reordenação com Mirror98."""
    flipped = points.copy()
    flipped[..., 0] = width - 1 - points[..., 0]
    return flipped[..., mirror, :]


# =========================================================================================
# 3. Cache de recortes e Dataset
# =========================================================================================
CACHE_SIZE = 384
CACHE_MARGIN = 1.875  # = 1,25 x 1,5: os 256 px centrais são o recorte de entrada; a folga
                      # comporta rotação, afastamento e translação sem bordas pretas
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _crop_one(args):
    image_path, box = args
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    m = crop_matrix(box, CACHE_SIZE, CACHE_MARGIN)
    return warp(image, m, CACHE_SIZE), m


def build_cache(data_dir, split: str, workers: int = 8) -> Path:
    """Salva uma vez cada face como recorte 384 (data/cache_<split>_384/)."""
    out = Path(data_dir) / f"cache_{split}_{CACHE_SIZE}"
    if (out / "meta.npz").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    faces = read_faces(data_dir, split)
    images = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(faces), CACHE_SIZE, CACHE_SIZE, 3))
    matrices = np.zeros((len(faces), 2, 3))
    with ProcessPoolExecutor(workers) as pool:
        for i, (c, m) in enumerate(pool.map(_crop_one, [(f.image, f.box) for f in faces], chunksize=16)):
            images[i], matrices[i] = c, m
    images.flush()
    np.savez(out / "meta.npz",
             landmarks=np.stack([transform_points(f.landmarks.astype(np.float64), m)
                                 for f, m in zip(faces, matrices)]).astype(np.float32),
             landmarks_orig=np.stack([f.landmarks for f in faces]),
             boxes=np.stack([f.box for f in faces]),
             attributes=np.stack([f.attributes for f in faces]), matrices=matrices)
    (out / "images.json").write_text(json.dumps(
        [str(f.image.relative_to(Path(data_dir) / "WFLW_images")) for f in faces]))
    return out


class Cache:
    """Leitura do cache (abre a imagem sob demanda; pode ir para os workers do DataLoader)."""

    def __init__(self, path):
        self.path = Path(path)
        meta = np.load(self.path / "meta.npz")
        self.landmarks = meta["landmarks"]
        self.landmarks_orig = meta["landmarks_orig"]
        self.attributes = meta["attributes"]
        self._images = None

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.path / "images.npy", mmap_mode="r")
        return self._images

    def __getstate__(self):
        return {**self.__dict__, "_images": None}


def normalize(image_rgb_uint8: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (3, H, W) float normalizado como na ImageNet."""
    x = (image_rgb_uint8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class FaceLandmarks(Dataset):
    """Recortes com augmentation consistente com os pontos. Retorna (imagem, pontos, índice)."""

    def __init__(self, cache: Cache, indices, input_size=256, margin=1.25, augment=None,
                 mirror=None, seed=0):
        self.cache, self.indices = cache, np.asarray(indices)
        self.size, self.augment, self.mirror, self.seed = input_size, augment or {}, mirror, seed
        self.base_scale = input_size * CACHE_MARGIN / (margin * CACHE_SIZE)

    def __len__(self):
        return len(self.indices)

    def load(self, i, rng=None):
        a = self.augment
        angle, scale, shift = 0.0, 1.0, (0.0, 0.0)
        if rng is not None and a:
            angle = rng.uniform(-a["rotation"], a["rotation"])
            scale = rng.uniform(*a["scale"])
            shift = rng.uniform(-a["translation"], a["translation"], size=2)
        m = augment_matrix((CACHE_SIZE / 2, CACHE_SIZE / 2), self.base_scale, self.size, angle, scale, shift)
        idx = self.indices[i]
        image = warp(np.asarray(self.cache.images[idx]), m, self.size)
        points = transform_points(self.cache.landmarks[idx].astype(np.float64), m)
        if rng is not None and rng.random() < a.get("flip_prob", 0):
            image = image[:, ::-1]
            points = flip_points(points, self.size, self.mirror)
        if rng is not None and rng.random() < a.get("lowres_prob", 0):
            # face pequena de webcam: reduz 2 a 4 vezes e amplia de volta (perde detalhe, não move pontos)
            f = rng.uniform(*a.get("lowres_factor", (2.0, 4.0)))
            small = cv2.resize(image, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
            image = cv2.resize(small, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if rng is not None and a.get("color", 0):
            c = a["color"]
            alpha, beta = 1 + rng.uniform(-c, c), 255 * rng.uniform(-c, c) / 2   # contraste, brilho
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image), points.astype(np.float32)

    def __getitem__(self, i):
        rng = None
        if self.augment:
            worker = torch.utils.data.get_worker_info()
            rng = np.random.default_rng([self.seed, int(torch.randint(0, 2**31, (1,))), i,
                                         worker.id if worker else 0])
        image, points = self.load(i, rng)
        return normalize(image), torch.from_numpy(points), int(self.indices[i])


# =========================================================================================
# 4. Modelo
# =========================================================================================
class LandmarkNet(nn.Module):
    """ResNet-18 (ImageNet, sem avgpool e fc) + cabeça própria: deconvs 8->16->32->64 + conv 1x1."""

    def __init__(self, num_landmarks=98, pretrained=True, head_channels=256, num_deconv=3, **_):
        super().__init__()
        net = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        layers, c_in = [], 512
        for _ in range(num_deconv):
            layers += [nn.ConvTranspose2d(c_in, head_channels, 4, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(head_channels), nn.ReLU(inplace=True)]
            c_in = head_channels
        layers.append(nn.Conv2d(head_channels, num_landmarks, 1))
        self.head = nn.Sequential(*layers)
        for m in self.head.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.stride = 32 // (2 ** num_deconv)  # pixels da entrada por célula do mapa

    def forward(self, x):
        """Retorna (coordenadas (B, 98, 2) em pixels da entrada, logits dos mapas)."""
        logits = self.head(self.backbone(x))
        return soft_argmax(logits, self.stride), logits


def soft_argmax(logits: torch.Tensor, stride: int) -> torch.Tensor:
    """Posição esperada sob um softmax espacial; centro da célula i = stride*i + (stride-1)/2."""
    b, k, h, w = logits.shape
    prob = F.softmax(logits.reshape(b, k, h * w).float(), dim=-1).reshape(b, k, h, w)
    xs = torch.arange(w, device=logits.device, dtype=prob.dtype)
    ys = torch.arange(h, device=logits.device, dtype=prob.dtype)
    x = (prob.sum(2) * xs).sum(-1)
    y = (prob.sum(3) * ys).sum(-1)
    return torch.stack([x, y], -1) * stride + (stride - 1) / 2


def build_model(cfg: dict, pretrained: bool | None = None) -> LandmarkNet:
    m = cfg["model"]
    return LandmarkNet(pretrained=m["pretrained"] if pretrained is None else pretrained,
                       head_channels=m["head_channels"], num_deconv=m["num_deconv"])


def count_parameters(model: LandmarkNet) -> dict:
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "head": head, "backbone": total - head}


# =========================================================================================
# 5. Perda
# =========================================================================================
# pares verticais (pálpebra de cima, de baixo) e cantos de cada olho, para a abertura do olho
EYELID_PAIRS = [(61, 67), (62, 66), (63, 65), (71, 73), (70, 74), (69, 75)]
EYE_CORNERS = [(60, 64)] * 3 + [(72, 68)] * 3


class CoordinateLoss(nn.Module):
    """Perda nas coordenadas, com três termos opcionais para os olhos.

    coordenadas  L1 (ou Wing) / tamanho da entrada, com peso eye_weight nos pontos 60-75
    abertura     |abertura prevista - anotada| / largura do olho, nos 6 pares das pálpebras
                 (ataca direto o EAR, que a L1 nos pontos quase não pune)
    heatmap      divergência de Jensen-Shannon entre o softmax de cada mapa e uma gaussiana
                 no ponto anotado (como no DSNT): força picos estreitos em vez de mapas espalhados
    """

    def __init__(self, kind="l1", input_size=256, eye_weight=1.0, wing_w=10.0, wing_eps=2.0,
                 opening_weight=0.0, heatmap_weight=0.0, heatmap_sigma=1.5):
        super().__init__()
        w = torch.ones(NUM_LANDMARKS)
        w[EYE_POINTS] = eye_weight
        self.register_buffer("weights", w / w.mean())
        self.kind, self.input_size = kind, input_size
        self.wing_w, self.wing_eps = wing_w, wing_eps
        self.wing_c = wing_w - wing_w * math.log(1 + wing_w / wing_eps)
        self.opening_weight, self.heatmap_weight, self.heatmap_sigma = opening_weight, heatmap_weight, heatmap_sigma
        self.register_buffer("lid_a", torch.tensor([a for a, _ in EYELID_PAIRS]))
        self.register_buffer("lid_b", torch.tensor([b for _, b in EYELID_PAIRS]))
        self.register_buffer("corner_a", torch.tensor([a for a, _ in EYE_CORNERS]))
        self.register_buffer("corner_b", torch.tensor([b for _, b in EYE_CORNERS]))

    def coordinate_term(self, pred, target):
        d = (pred - target).abs()
        if self.kind == "l1":
            err = d.sum(-1)
        elif self.kind == "wing":
            err = torch.where(d < self.wing_w, self.wing_w * torch.log1p(d / self.wing_eps),
                              d - self.wing_c).sum(-1)
        else:
            raise ValueError(self.kind)
        return (err / self.input_size * self.weights).mean()

    def opening_term(self, pred, target):
        width = (target[:, self.corner_a] - target[:, self.corner_b]).norm(dim=-1).clamp(min=1.0)
        open_pred = (pred[:, self.lid_a] - pred[:, self.lid_b]).norm(dim=-1)
        open_gt = (target[:, self.lid_a] - target[:, self.lid_b]).norm(dim=-1)
        return ((open_pred - open_gt).abs() / width).mean()

    def heatmap_term(self, logits, target):
        b, k, h, w = logits.shape
        stride = self.input_size / w
        cells = (target - (stride - 1) / 2) / stride                      # alvo em unidades de célula
        ys = torch.arange(h, device=logits.device, dtype=torch.float32)
        xs = torch.arange(w, device=logits.device, dtype=torch.float32)
        gy = torch.exp(-(ys[None, None] - cells[..., 1:2]) ** 2 / (2 * self.heatmap_sigma ** 2))
        gx = torch.exp(-(xs[None, None] - cells[..., 0:1]) ** 2 / (2 * self.heatmap_sigma ** 2))
        q = (gy[..., :, None] * gx[..., None, :]).reshape(b, k, -1)
        q = q / q.sum(-1, keepdim=True).clamp(min=1e-12)
        logp = F.log_softmax(logits.reshape(b, k, -1).float(), dim=-1)
        p = logp.exp()
        m = 0.5 * (p + q)
        logm = m.clamp(min=1e-12).log()
        js = 0.5 * (p * (logp - logm)).sum(-1) + 0.5 * (q * (q.clamp(min=1e-12).log() - logm)).sum(-1)
        return js.mean()

    def forward(self, pred, target, logits=None):
        loss = self.coordinate_term(pred, target)
        if self.opening_weight:
            loss = loss + self.opening_weight * self.opening_term(pred, target)
        if self.heatmap_weight and logits is not None:
            loss = loss + self.heatmap_weight * self.heatmap_term(logits, target)
        return loss


# =========================================================================================
# 6. Métricas
# =========================================================================================
EAR6_RIGHT = (60, 61, 63, 64, 65, 67)
EAR6_LEFT = (72, 71, 69, 68, 75, 73)


def point_errors(pred, gt):
    """(N, 98) erro de cada ponto dividido pela distância 60-72."""
    a, b = OUTER_EYE_CORNERS
    d = np.linalg.norm(gt[..., a, :] - gt[..., b, :], axis=-1)
    return np.linalg.norm(pred - gt, axis=-1) / d[..., None]


def nme(pred, gt, points=None):
    e = point_errors(pred, gt)
    return e.mean(-1) if points is None else e[..., list(points)].mean(-1)


def ear(pts):
    """Eye Aspect Ratio (6 pontos), média dos dois olhos. pts: (..., 98, 2)."""
    def one(idx):
        p1, p2, p3, p4, p5, p6 = (pts[..., i, :] for i in idx)
        n = lambda u, v: np.linalg.norm(u - v, axis=-1)  # noqa: E731
        return (n(p2, p6) + n(p3, p5)) / (2 * n(p1, p4) + 1e-9)
    return (one(EAR6_RIGHT) + one(EAR6_LEFT)) / 2


def evaluate_arrays(pred, gt, closed_threshold=0.15) -> dict:
    per_face = nme(pred, gt)
    e_gt, e_pred = ear(gt), ear(pred)
    closed = e_gt < closed_threshold
    return {"nme": float(per_face.mean()), "fr10": float((per_face > 0.10).mean()),
            "eye_nme": float(nme(pred, gt, EYE_POINTS).mean()),
            "ear_corr": float(np.corrcoef(e_gt, e_pred)[0, 1]),
            "ear_mae": float(np.abs(e_gt - e_pred).mean()),
            # inclinação da reta EAR previsto x anotado (1 = ideal; < 1 = puxa para a média)
            "ear_slope": float(np.polyfit(e_gt, e_pred, 1)[0]),
            # nos olhos fechados (EAR anotado < 0,15): EAR previsto médio e fração reconhecida
            "closed_ear_pred": float(e_pred[closed].mean()) if closed.any() else float("nan"),
            "closed_recall": float((e_pred[closed] < closed_threshold).mean()) if closed.any() else float("nan")}


# =========================================================================================
# 7. Detector de faces e inferência
# =========================================================================================
YUNET_FILE = "face_detection_yunet_2023mar.onnx"


def find_yunet(root) -> Path:
    """Procura o YuNet em models/ ou modelos/."""
    for folder in ("models", "modelos"):
        p = Path(root) / folder / YUNET_FILE
        if p.is_file():
            return p
    raise FileNotFoundError(f"{YUNET_FILE} não encontrado em {root}/models nem {root}/modelos")


class FaceDetector:
    """YuNet (OpenCV). O retângulo segue a mesma convenção do WFLW (testa incluída)."""

    def __init__(self, model_path, score_threshold=0.6, nms_threshold=0.3, max_side=1600):
        self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                                  score_threshold, nms_threshold, 5000)
        self.max_side = max_side

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        """(N, 5): x_min, y_min, x_max, y_max, score em pixels da imagem original."""
        h, w = image_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        img = image_bgr if scale == 1.0 else cv2.resize(image_bgr, None, fx=scale, fy=scale)
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)
        if faces is None:
            return np.zeros((0, 5), dtype=np.float32)
        x, y, bw, bh = (faces[:, i] / scale for i in range(4))
        return np.stack([x, y, x + bw, y + bh, faces[:, 14]], 1).astype(np.float32)


def load_checkpoint(path, device="cpu"):
    """Reconstrói o modelo a partir do checkpoint, sem baixar nada."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


# ordem dos pontos depois de espelhar a face (Mirror98.txt do WFLW), para o TTA na inferência
MIRROR98 = np.array([32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,46,45,44,43,42,50,49,48,47,37,36,35,34,33,41,40,39,38,51,52,53,54,59,58,57,56,55,72,71,70,69,68,75,74,73,64,63,62,61,60,67,66,65,82,81,80,79,78,77,76,87,86,85,84,83,92,91,90,89,88,95,94,93,97,96])


@torch.no_grad()
def predict_crops(model, crops_rgb, device="cpu", flip_tta=False) -> np.ndarray:
    """Lista de recortes (S, S, 3) RGB -> (N, 98, 2) em pixels do recorte.

    flip_tta: também prevê o recorte espelhado, desfaz o espelhamento (x' = S - 1 - x e
    reordenação pelo Mirror98) e tira a média das duas previsões.
    """
    if not crops_rgb:
        return np.zeros((0, NUM_LANDMARKS, 2), dtype=np.float32)
    x = torch.stack([normalize(c) for c in crops_rgb]).to(device)
    pts = model(x)[0].float().cpu().numpy()
    if flip_tta:
        size = crops_rgb[0].shape[1]
        flipped = model(torch.flip(x, dims=[3]))[0].float().cpu().numpy()
        flipped[..., 0] = size - 1 - flipped[..., 0]
        pts = (pts + flipped[:, MIRROR98]) / 2
    return pts
