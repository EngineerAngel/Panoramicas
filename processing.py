# processing.py
import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from image_utils import ajustar_gamma, reducir_ruido, reducir_highlights
from classification import clasificar_nivel_oscuridad
from config import NUM_PROCESOS, NUM_THREADS

# Procesamiento específico por clasificación

def procesar_extremadamente_oscura(img):

    img_procesada = cv2.bilateralFilter(img, d=3, sigmaColor=5, sigmaSpace=5)


    lab = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)


    clahe = cv2.createCLAHE(clipLimit=0.9, tileGridSize=(8, 8))
    l = clahe.apply(l)


    img_procesada = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask_extrema = (v < 50).astype(np.float32)
    mask_suave = cv2.GaussianBlur(mask_extrema, (15, 15), 8)


    v_float = v.astype(np.float32)

    v_float = v_float + mask_suave * (v_float * 0.25)
    v = np.clip(v_float, 0, 255).astype(np.uint8)

    img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    img_procesada = ajustar_gamma(img_procesada, gamma=1.1)

    img_procesada = cv2.bilateralFilter(img_procesada, d=8, sigmaColor=15, sigmaSpace=15)

    hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.1, 0, 255).astype(np.uint8)
    img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    return img_procesada

def ajustar_saturacion_adaptativa(img, factor_base=1.35):
    """
    Valores originales de saturación en diferentes modos:
    - Extremadamente oscura: s * 1.1
    - Muy oscura: s * 1.15
    - Normal: s * 1.2
    - Clara: s * 1.1
    
    Nuevos valores base aumentados:
    - Base: 1.35 (antes 1.15)
    - Desaturada: * 1.4 (antes 1.3)
    - Moderada: * 1.2 (antes 1.1)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Calcular factor adaptativo basado en la saturación promedio actual
    sat_promedio = np.mean(s)
    if sat_promedio < 50:
        factor = factor_base * 1.4  # Más saturación para imágenes desaturadas
    elif sat_promedio < 100:
        factor = factor_base * 1.2  # Saturación moderada
    else:
        factor = factor_base  # Mantener factor base para imágenes ya saturadas
    
    # Aplicar saturación con máscara de luminosidad
    mask_oscura = (v < 30).astype(np.float32)
    mask_clara = (v > 225).astype(np.float32)
    mask_media = 1 - mask_oscura - mask_clara
    
    # Ajustar saturación según la luminosidad
    s_float = s.astype(np.float32)
    s_ajustada = np.clip(s_float * factor * mask_media + 
                        s_float * (factor * 0.7) * mask_oscura +
                        s_float * (factor * 0.8) * mask_clara, 0, 255)
    
    s = s_ajustada.astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def procesar_muy_oscura(img):
    """
    Valores originales:
    - CLAHE: clipLimit=2.5, tileGridSize=(12, 12)
    - Levantar sombras: threshold=60, factor=1.4
    - Gamma: 1.0
    - Saturación: s * 1.15
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))
    l = clahe.apply(l)
    img_procesada = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # Usar nuevas funciones adaptativas
    img_procesada = levantar_sombras_v2(img_procesada, threshold=70, factor=1.2)
    img_procesada = ajustar_gamma(img_procesada, gamma=1.0)
    img_procesada = ajustar_saturacion_adaptativa(img_procesada, factor_base=1.4)  # Valores nuevos aumentados
    
    return img_procesada

def aplicar_clahe_adaptativo_v2(img, modo="normal"):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    brillo = np.mean(l)
    
    if modo == "oscura":
        clip, tiles = 2.0, (12, 12)
    elif modo == "clara":
        clip, tiles = 1.2, (20, 20)
    else:
        if brillo < 80:
            clip, tiles = 1.5, (16, 16)
        elif brillo > 160:
            clip, tiles = 2.0, (16, 16)
        else:
            clip, tiles = 1.8, (16, 16)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l_eq = clahe.apply(l)
    img_lab_eq = cv2.merge([l_eq, a, b])
    img_result = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)
    return img_result, brillo, clip, tiles

def levantar_sombras_v2(img, threshold=60, factor=1.08):
    """
    Valores originales:
    - threshold=60
    - factor=1.08
    - gamma=0.85 (para brillo_promedio < 30)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    brillo_promedio = np.mean(v)
    
    # Ajuste adaptativo del factor según el brillo promedio
    if brillo_promedio < 30:
        factor = max(factor * 1.5, 1.3)  # Factor más agresivo para muy oscuras
    elif brillo_promedio < 50:
        factor = max(factor * 1.3, 1.2)  # Factor moderado para oscuras
    else:
        factor = max(factor, 1.02)  # Mantener el mínimo original
        
    mask = v < threshold
    v_float = v.astype(np.float32)
    v_float[mask] = np.clip(v_float[mask] * factor, 0, 255)
    
    # Suavizado de la transición
    mask_float = mask.astype(np.float32)
    mask_suave = cv2.GaussianBlur(mask_float, (5, 5), 1.5)
    v = np.clip(v_float * mask_suave + v * (1 - mask_suave), 0, 255).astype(np.uint8)
    
    img_sombras = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    if brillo_promedio < 30:
        img_sombras = ajustar_gamma(img_sombras, gamma=0.85)
    return img_sombras

def procesar_imagen_avanzado_optimizado_v2(img, modo="normal"):
    if modo == "oscura":
        img_procesada, brillo, clip, tiles = aplicar_clahe_adaptativo_v2(img, modo="oscura")
        img_procesada = reducir_ruido(img_procesada)
        img_procesada = levantar_sombras_v2(img_procesada, threshold=70, factor=1.25)
        img_procesada = reducir_highlights(img_procesada, threshold=240, factor=0.9)
        img_procesada = ajustar_gamma(img_procesada, gamma=0.8)
        hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.15, 0, 255).astype(np.uint8)
        img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    elif modo == "clara":
        img_procesada, brillo, clip, tiles = aplicar_clahe_adaptativo_v2(img, modo="clara")
        img_procesada = reducir_ruido(img_procesada)
        img_procesada = ajustar_gamma(img_procesada, gamma=0.95)
        hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.1, 0, 255).astype(np.uint8)
        mask_clara = (v > 240).astype(np.float32)
        mask_clara_suave = cv2.GaussianBlur(mask_clara, (15, 15), 8)
        v_float = v.astype(np.float32)
        v_float = v_float - mask_clara_suave * ((v_float - 240) * 0.6)
        v = np.clip(v_float, 0, 255).astype(np.uint8)
        img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    else:
        img_procesada, brillo, clip, tiles = aplicar_clahe_adaptativo_v2(img, modo="normal")
        img_procesada = reducir_ruido(img_procesada)
        img_procesada = levantar_sombras_v2(img_procesada, threshold=60, factor=1.08)
        img_procesada = reducir_highlights(img_procesada, threshold=240, factor=0.85)
        img_procesada = ajustar_gamma(img_procesada, gamma=0.88)
        hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
        img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    return img_procesada, brillo

def procesar_segun_clasificacion(img, clasificacion):
    """
    Procesa una imagen según su clasificación.
    
    Args:
        img: Imagen a procesar
        clasificacion: Tipo de clasificación de la imagen
    """
    if clasificacion == "EXTREMADAMENTE_OSCURA":
        img_procesada = procesar_extremadamente_oscura(img)
        brillo = np.mean(cv2.cvtColor(img_procesada, cv2.COLOR_BGR2GRAY))
        return img_procesada, brillo
    elif clasificacion == "MUY_OSCURA":
        img_procesada = procesar_muy_oscura(img)
        brillo = np.mean(cv2.cvtColor(img_procesada, cv2.COLOR_BGR2GRAY))
        return img_procesada, brillo
    elif clasificacion == "OSCURA":
        return procesar_imagen_avanzado_optimizado_v2(img, modo="oscura")
    elif clasificacion in ["MEDIO_OSCURA", "NORMAL"]:
        return procesar_imagen_avanzado_optimizado_v2(img, modo="normal")
    else:
        return procesar_imagen_avanzado_optimizado_v2(img, modo="clara")

def procesar_imagen_individual_v2(datos_imagen):
    ruta_entrada, ruta_salida, archivo = datos_imagen
    try:
        img = cv2.imread(ruta_entrada)
        if img is None:
            return None, f"No se pudo leer: {archivo}"
        gris_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brillo_original = np.mean(gris_original)
        clasificacion, brillo_prom, p10, p50, p90, pixeles_oscuros = clasificar_nivel_oscuridad(img)
        img_corregida, brillo_procesado = procesar_segun_clasificacion(img, clasificacion)
        gris_corregido = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2GRAY)
        brillo_corregido = np.mean(gris_corregido)
        cv2.imwrite(ruta_salida, img_corregida)
        return [archivo, brillo_original, clasificacion, brillo_corregido, f"{pixeles_oscuros:.1f}%"], f"✅ {archivo} ({clasificacion})"
    except Exception as e:
        return None, f"❌ Error procesando {archivo}: {str(e)}"

def procesar_carpeta_paralelo_v2(ruta_carpeta, ruta_salida_carpeta, usar_procesos=True):
    resultados = []
    os.makedirs(ruta_salida_carpeta, exist_ok=True)
    if not os.path.exists(ruta_carpeta):
        print(f"   La carpeta no existe: {ruta_carpeta}")
        return resultados
    archivos_imagen = [f for f in os.listdir(ruta_carpeta) 
                       if f.lower().endswith((".jpeg", ".jpg", ".png"))]
    if not archivos_imagen:
        print(f"   No se encontraron imágenes en: {ruta_carpeta}")
        return resultados
    print(f"   Procesando {len(archivos_imagen)} imágenes con clasificación automática...")
    datos_procesamiento = [
        (os.path.join(ruta_carpeta, archivo), 
         os.path.join(ruta_salida_carpeta, archivo), 
         archivo)
        for archivo in archivos_imagen
    ]
    inicio = time.time()
    if usar_procesos and len(archivos_imagen) > 4:
        with ProcessPoolExecutor(max_workers=NUM_PROCESOS) as executor:
            resultados_paralelos = list(executor.map(procesar_imagen_individual_v2, datos_procesamiento))
    else:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            resultados_paralelos = list(executor.map(procesar_imagen_individual_v2, datos_procesamiento))
    fin = time.time()
    tiempo_total = fin - inicio
    clasificaciones = {}
    for resultado, mensaje in resultados_paralelos:
        if resultado is not None:
            clasificacion = resultado[2]
            clasificaciones[clasificacion] = clasificaciones.get(clasificacion, 0) + 1
            resultados.append(resultado)
        print(f"    {mensaje}")
    print(f"   📊 Clasificaciones:")
    for clasif, count in clasificaciones.items():
        print(f"      {clasif}: {count} imágenes")
    velocidad = len(archivos_imagen) / tiempo_total if tiempo_total > 0 else 0
    print(f"   Completado en {tiempo_total:.2f}s ({velocidad:.1f} img/s)")
    return resultados
