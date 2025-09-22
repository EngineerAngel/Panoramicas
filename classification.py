# classification.py
import cv2
import numpy as np


def clasificar_nivel_oscuridad(img):
    """Clasifica la imagen según su nivel de oscuridad"""

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brillo_promedio = np.mean(gray)
    
    p10 = np.percentile(gray, 10)
    p50 = np.percentile(gray, 50)
    p90 = np.percentile(gray, 90)

    pixeles_muy_oscuros = np.sum(gray < 30) / gray.size * 100
    pixeles_oscuros = np.sum(gray < 80) / gray.size * 100

    if brillo_promedio < 25 and pixeles_muy_oscuros > 70:
        return "EXTREMADAMENTE_OSCURA", brillo_promedio, p10, p50, p90, pixeles_muy_oscuros
    elif brillo_promedio < 40 and pixeles_muy_oscuros > 50:
        return "MUY_OSCURA", brillo_promedio, p10, p50, p90, pixeles_muy_oscuros
    elif brillo_promedio < 60 and pixeles_oscuros > 40:
        return "OSCURA", brillo_promedio, p10, p50, p90, pixeles_muy_oscuros
    elif brillo_promedio < 100:
        return "MEDIO_OSCURA", brillo_promedio, p10, p50, p90, pixeles_muy_oscuros
    elif brillo_promedio < 140:
        return "NORMAL", brillo_promedio, p10, p50, p90, pixeles_muy_oscuros
    elif brillo_promedio < 180:
        return "CLARA", brillo_promedio, p10, p50, p90, pixeles_muy_oscuros
    else:
        return "MUY_CLARA", brillo_promedio, p10, p50, p90, pixeles_muy_oscuros