# image_utils.py
import cv2
import numpy as np


def ajustar_gamma(img, gamma=1):
    """Ajuste de gamma con tabla pre-calculada"""
    inv_gamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, tabla)


def reducir_ruido(img):
    """Bilateral Filter optimizado"""
    return cv2.bilateralFilter(img, d=5, sigmaColor=20, sigmaSpace=20)


def reducir_highlights(img, threshold=250, factor=0.8):
    """Reducir highlights en áreas muy brillantes"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mask = (l > threshold).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (21, 21), 15)
    l_adjusted = l.astype(np.float32) * (1 - mask * (1 - factor))
    l_adjusted = np.clip(l_adjusted, 0, 240).astype(np.uint8)
    lab_adjusted = cv2.merge([l_adjusted, a, b])
    return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)


def tone_mapping_local(img, threshold=200, intensidad=0.5, log_factor=3):
    """Tone mapping local para mejor contraste"""
    img_float = img.astype(np.float32) / 255.0
    mask = np.mean(img_float, axis=2) < threshold/255.0
    img_log = np.log1p(img_float * log_factor) / np.log1p(log_factor)
    img_out = img_float.copy()
    img_out[mask] = img_log[mask] + (img_log[mask] - img_float[mask]) * intensidad
    return (img_out * 255).astype(np.uint8)
