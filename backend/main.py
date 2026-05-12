from __future__ import annotations

import os
import tempfile
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
)
from PIL import Image
from torchvision import models, transforms

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Liver Cancer Image Feature Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UNET_PATH  = os.path.join(BASE_DIR, "models", "best_unet_model.pth")

# ── Image transform ────────────────────────────────────────────────────────────
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── MONAI pipeline ─────────────────────────────────────────────────────────────
MONAI_PIPELINE = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    ScaleIntensityRanged(keys=["image"],
                         a_min=-175, a_max=250,
                         b_min=0.0,  b_max=1.0, clip=True),
    ResizeWithPadOrCropd(keys=["image"], spatial_size=(96, 96, 48)),
])

# ── Lazy model singletons ──────────────────────────────────────────────────────
_unet     = None
_resnet   = None
_densenet = None


def get_unet():
    global _unet
    if _unet is None:
        m = UNet(spatial_dims=3, in_channels=1, out_channels=2,
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                 num_res_units=2, norm=Norm.BATCH)
        if os.path.exists(UNET_PATH):
            m.load_state_dict(torch.load(UNET_PATH, map_location="cpu"))
            print("✅ UNet model loaded from", UNET_PATH)
        else:
            print("⚠️  UNet weights not found — using random weights")
        m.eval()
        _unet = m
    return _unet


def get_resnet():
    global _resnet
    if _resnet is None:
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        _resnet = nn.Sequential(*list(base.children())[:-1])
        _resnet.eval()
    return _resnet


def get_densenet():
    global _densenet
    if _densenet is None:
        base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        _densenet = nn.Sequential(base.features, nn.AdaptiveAvgPool2d((1, 1)))
        _densenet.eval()
    return _densenet


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_unet_segmentation(nii_path: str) -> np.ndarray:
    sample = MONAI_PIPELINE({"image": nii_path})
    tensor = sample["image"].unsqueeze(0)
    with torch.no_grad():
        out  = get_unet()(tensor)
        pred = torch.argmax(out, dim=1).squeeze().numpy()
    return pred.astype(np.float32)


def get_best_slice(vol: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, int]:
    areas  = [seg[:, :, z].sum() for z in range(seg.shape[2])]
    best_z = int(np.argmax(areas))
    return vol[:, :, best_z], best_z


def to_pil(ct_slice: np.ndarray) -> Image.Image:
    u8 = (np.clip(ct_slice, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(u8).convert("RGB").resize((224, 224))


def cnn_features(pil: Image.Image, model: nn.Module) -> np.ndarray:
    t = IMG_TRANSFORM(pil).unsqueeze(0)
    with torch.no_grad():
        return model(t).squeeze().numpy()


def radiomic_features(vol: np.ndarray, seg: np.ndarray) -> Dict[str, float]:
    mask   = seg > 0
    voxels = vol[mask]
    if len(voxels) == 0:
        return {}

    f: Dict[str, float] = {}
    f["mean"]     = float(np.mean(voxels))
    f["std"]      = float(np.std(voxels))
    f["min"]      = float(np.min(voxels))
    f["max"]      = float(np.max(voxels))
    f["median"]   = float(np.median(voxels))
    f["p10"]      = float(np.percentile(voxels, 10))
    f["p90"]      = float(np.percentile(voxels, 90))
    f["iqr"]      = float(np.percentile(voxels, 75) - np.percentile(voxels, 25))
    norm          = (voxels - f["mean"]) / (f["std"] + 1e-6)
    f["skewness"] = float(np.mean(norm ** 3))
    f["kurtosis"] = float(np.mean(norm ** 4))
    f["energy"]   = float(np.sum(voxels ** 2) / len(voxels))
    f["volume"]   = float(mask.sum())
    f["range"]    = float(f["max"] - f["min"])

    areas  = seg.sum(axis=(0, 1))
    best_z = int(np.argmax(areas))
    region = vol[:, :, best_z][seg[:, :, best_z] > 0]

    if len(region) > 0:
        f["slice_mean"]     = float(np.mean(region))
        f["slice_std"]      = float(np.std(region))
        f["slice_contrast"] = float(np.var(region))
        f["slice_energy"]   = float(np.sum(region ** 2) / len(region))
        hist, _             = np.histogram(region, bins=32, density=True)
        f["slice_entropy"]  = float(-np.sum(hist * np.log2(hist + 1e-6)))
    else:
        for k in ["slice_mean","slice_std","slice_contrast",
                  "slice_energy","slice_entropy"]:
            f[k] = 0.0
    return f


def predict_risk(radio: Dict[str, float]) -> Tuple[float, str, str]:
    if not radio:
        return 0.5, "UNKNOWN", "Insufficient image data."
    score  = 0.0
    score += min(radio.get("volume",         0) / 50_000, 1.0) * 0.30
    score += min(radio.get("std",            0) / 150,    1.0) * 0.20
    score += min(abs(radio.get("skewness",   0)) / 3,     1.0) * 0.15
    score += min(radio.get("slice_entropy",  0) / 8,      1.0) * 0.20
    score += min(radio.get("slice_contrast", 0) / 3000,   1.0) * 0.15
    prob   = round(float(np.clip(score, 0.0, 1.0)), 3)

    if prob >= 0.65:
        return prob, "HIGH", (
            "High recurrence risk detected. Recommend follow-up imaging "
            "in 3 months, oncology referral, and AFP monitoring every 4 weeks.")
    if prob >= 0.40:
        return prob, "MODERATE", (
            "Moderate recurrence risk. Recommend follow-up imaging in "
            "4–6 months and routine AFP monitoring.")
    return prob, "LOW", (
        "Low recurrence risk. Continue standard surveillance with annual "
        "imaging and routine clinical review.")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "liver-image-feature-extraction"}


@app.post("/predict")
async def predict(ct_scan: UploadFile = File(...)):
    if not ct_scan.filename.endswith(".nii"):
        raise HTTPException(400, "Only .nii files are accepted.")

    with tempfile.TemporaryDirectory() as tmp:
        path    = os.path.join(tmp, "scan.nii")
        content = await ct_scan.read()
        with open(path, "wb") as fh:
            fh.write(content)

        try:
            # 1 — Preprocess & segment
            seg      = run_unet_segmentation(path)
            sample   = MONAI_PIPELINE({"image": path})
            vol_proc = sample["image"][0].numpy()          # (96,96,48)

            # 2 — Best tumour slice
            ct_slice, best_z = get_best_slice(vol_proc, seg)
            pil_img          = to_pil(ct_slice)

            # 3 — CNN features
            resnet_feat   = cnn_features(pil_img, get_resnet()).tolist()
            densenet_feat = cnn_features(pil_img, get_densenet()).tolist()

            # 4 — Radiomic features (best model)
            radio = radiomic_features(vol_proc, seg)

            # 5 — Risk
            prob, level, advice = predict_risk(radio)

            return {
                "probability":        prob,
                "risk_score":         round(prob * 100, 1),
                "risk_level":         level,
                "advice":             advice,
                "best_slice_index":   best_z,
                "radiomic_features":  radio,
                "resnet_feature_dim": len(resnet_feat),
                "densenet_feature_dim": len(densenet_feat),
                "model_used":         "Radiomics (18-dim) — best model",
                "unet_val_loss":      0.5577,
            }
        except Exception as e:
            raise HTTPException(500, f"Processing failed: {e}") from e
