"""Uydu goruntusunden yol maskesi cikarimi."""
import cv2
import numpy as np
import torch

from src.inference.dlinknet_model import DinkNet34


def resolve_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_model(checkpoint_path, device="cpu"):
    model = DinkNet34()
    state = torch.load(str(checkpoint_path), map_location=device)
    # Checkpoint DataParallel ile egitildigi icin anahtarlar "module." onekli gelir.
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def predict_mask(model, image_bgr, device="cpu", threshold=0.5):
    # Egitimdeki normalizasyon ile ayni: [0,255] -> [-1.6, 1.6]
    img = image_bgr.astype(np.float32) / 255.0 * 3.2 - 1.6
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor).squeeze().cpu().numpy()
    return (out > threshold).astype(np.uint8) * 255


def infer_from_image(checkpoint_path, image_path, device="cpu", threshold=0.5):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {image_path}")
    model = load_model(checkpoint_path, device)
    return predict_mask(model, image, device, threshold), image


def load_mask_file(mask_path):
    """Hazir / manuel bir yol maskesini binary (0/255) olarak yukler."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Maske okunamadi: {mask_path}")
    return (mask > 127).astype(np.uint8) * 255
