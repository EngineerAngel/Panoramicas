import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import partial
import time

# --- CONFIGURACIÓN ---
# Cambia esta ruta por cualquiera de tus carpetas base:
carpeta_base = r"C:\Users\ANGEL GOMEZ\OneDrive\Documentos\BC-163-02\S2C2"
carpeta_salida_base = r"C:\Users\ANGEL GOMEZ\Proyectos\Proyecto_semic\Correccion_final"

# CONFIGURACIÓN DE PARALELIZACIÓN
NUM_PROCESOS = min(8, multiprocessing.cpu_count())  # Máximo 8 procesos o el número de CPUs
NUM_THREADS = 4  # Para operaciones I/O (lectura/escritura)

def aplicar_clahe_adaptativo(img):
    # Convertir a LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    # Medir brillo promedio en canal L
    brillo = np.mean(l)

    # --- Ajustar parámetros según brillo ---
    if brillo < 80:
        clip, tiles = 1.5, (16, 16)
    elif brillo > 160:
        clip, tiles = 2.0, (16, 16)
    else:
        clip, tiles = 1.8, (16, 16)

    # Crear CLAHE con los parámetros elegidos
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l_eq = clahe.apply(l)

    # Recomponer imagen LAB → BGR
    img_lab_eq = cv2.merge([l_eq, a, b])
    img_result = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)

    return img_result, brillo, clip, tiles

def reducir_ruido(img):
    # Bilateral Filter optimizado - parámetros más pequeños para mayor velocidad
    return cv2.bilateralFilter(img, d=5, sigmaColor=20, sigmaSpace=20)

def ajustar_gamma(img, gamma=1):
    # Pre-calcular tabla para mejor rendimiento
    inv_gamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, tabla)

def reducir_highlights(img, threshold=240, factor=0.85):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mask = (l > threshold).astype(np.float32)
    # Blur más pequeño para mayor velocidad
    mask = cv2.GaussianBlur(mask, (31, 31), 20)  # Reducido de (51,51),30
    l_adjusted = l.astype(np.float32) * (1 - mask * (1 - factor))
    l_adjusted = np.clip(l_adjusted, 0, 250).astype(np.uint8)
    lab_adjusted = cv2.merge([l_adjusted, a, b])
    return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

def levantar_sombras(img, threshold=60, factor=1.08):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    brillo_promedio = np.mean(v)
    mask = v < threshold
    
    if brillo_promedio >= 25:  # ⬇️ Umbral más bajo
        factor_final = factor
    else:
        factor_final = 1.03  # ⬆️ Factor mínimo en lugar de 0
    
    # ⬆️ Aplicar suavizado gradual
    v_float = v.astype(np.float32)
    v_float[mask] = np.clip(v_float[mask] * factor_final, 0, 255)
    v = v_float.astype(np.uint8)
    
    img_sombras = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    # ⬆️ GAMMA MÁS SUAVE para evitar posterización
    if brillo_promedio < 30:  # ⬇️ Umbral más bajo
        img_sombras = ajustar_gamma(img_sombras, gamma=0.85)  # ⬆️ Menos agresivo

    return img_sombras

def tone_mapping_local(img, threshold=200, intensidad=0.3, log_factor=2):
    img_float = img.astype(np.float32) / 255.0
    mask = np.mean(img_float, axis=2) < threshold/255.0
    img_log = np.log1p(img_float * log_factor) / np.log1p(log_factor)
    img_out = img_float.copy()
    img_out[mask] = img_log[mask] + (img_log[mask] - img_float[mask]) * intensidad
    return (img_out * 255).astype(np.uint8)

def procesar_imagen_avanzado_optimizado(img):
    """Aplica todas las correcciones avanzadas - PARÁMETROS ORIGINALES"""
    
    # 1. CLAHE
    img_procesada, brillo, clip, tiles = aplicar_clahe_adaptativo(img)
    
    # 2. Reducción de ruido
    img_procesada = reducir_ruido(img_procesada)
    
    # 3. Levantar sombras
    img_procesada = levantar_sombras(img_procesada, threshold=60, factor=1.08)
    
    # 4. Reducir highlights
    img_procesada = reducir_highlights(img_procesada, threshold=240, factor=0.85)
    
    # 5. Tone mapping local
    img_procesada = tone_mapping_local(img_procesada, threshold=220, intensidad=0.3)
    
    # 6. Ajuste de gamma
    img_procesada = ajustar_gamma(img_procesada, gamma=0.88)
    
    # 7. Saturación
    hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.2, 0, 255).astype(np.uint8)  # Factor original
    img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    return img_procesada, brillo

def procesar_imagen_individual(datos_imagen):
    """Función para procesar una imagen individual (para paralelización)"""
    ruta_entrada, ruta_salida, archivo = datos_imagen
    
    try:
        # Leer imagen
        img = cv2.imread(ruta_entrada)
        if img is None:
            return None, f"No se pudo leer: {archivo}"
        
        # Calcular brillo original
        gris_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brillo_original = np.mean(gris_original)
        
        # Procesar imagen
        img_corregida, brillo_procesado = procesar_imagen_avanzado_optimizado(img)
        
        # Calcular brillo corregido
        gris_corregido = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2GRAY)
        brillo_corregido = np.mean(gris_corregido)
        
        # Guardar imagen
        cv2.imwrite(ruta_salida, img_corregida)
        
        return [archivo, brillo_original, "Corrección aplicada", brillo_corregido], f"✅ {archivo}"
        
    except Exception as e:
        return None, f" Error procesando {archivo}: {str(e)}"

def procesar_carpeta_paralelo(ruta_carpeta, ruta_salida_carpeta, usar_procesos=True):
    """Versión paralelizada del procesamiento de carpetas"""
    
    resultados = []
    
    # Crear la carpeta de salida si no existe
    os.makedirs(ruta_salida_carpeta, exist_ok=True)
    
    if not os.path.exists(ruta_carpeta):
        print(f"   La carpeta no existe: {ruta_carpeta}")
        return resultados
    
    # Obtener lista de archivos de imagen
    archivos_imagen = [f for f in os.listdir(ruta_carpeta) 
                      if f.lower().endswith((".jpeg", ".jpg", ".png"))]
    
    if not archivos_imagen:
        print(f"   No se encontraron imágenes en: {ruta_carpeta}")
        return resultados
    
    print(f"   Procesando {len(archivos_imagen)} imágenes en paralelo...")
    
    # Preparar datos para procesamiento paralelo
    datos_procesamiento = [
        (os.path.join(ruta_carpeta, archivo), 
         os.path.join(ruta_salida_carpeta, archivo), 
         archivo)
        for archivo in archivos_imagen
    ]
    
    inicio = time.time()
    
    if usar_procesos and len(archivos_imagen) > 4:
        # Usar ProcessPoolExecutor para imágenes múltiples (mejor para CPU intensivo)
        with ProcessPoolExecutor(max_workers=NUM_PROCESOS) as executor:
            resultados_paralelos = list(executor.map(procesar_imagen_individual, datos_procesamiento))
    else:
        # Usar ThreadPoolExecutor para pocas imágenes (mejor para I/O)
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            resultados_paralelos = list(executor.map(procesar_imagen_individual, datos_procesamiento))
    
    fin = time.time()
    tiempo_total = fin - inicio
    
    # Procesar resultados
    for resultado, mensaje in resultados_paralelos:
        if resultado is not None:
            resultados.append(resultado)
        print(f"    {mensaje}")
    
    velocidad = len(archivos_imagen) / tiempo_total if tiempo_total > 0 else 0
    print(f"   Completado en {tiempo_total:.2f}s ({velocidad:.1f} img/s)")
    
    return resultados

def main():
    print(" Iniciando procesamiento de imágenes OPTIMIZADO...")
    print(f" Carpeta base: {carpeta_base}")
    print(f" Carpeta de salida: {carpeta_salida_base}")
    print(f" Configuración: {NUM_PROCESOS} procesos, {NUM_THREADS} threads")
    print("-" * 60)
    
    # Verificar si la carpeta base existe
    if not os.path.exists(carpeta_base):
        print(f" Error: La carpeta base no existe: {carpeta_base}")
        return
    
    # DEFINIR ruta_relativa AL INICIO
    partes_ruta = carpeta_base.split(os.sep)
    ruta_relativa = ""
    
    try:
        indice_documentos = partes_ruta.index("Documentos")
        ruta_relativa = os.sep.join(partes_ruta[indice_documentos + 1:])
        print(f" Estructura detectada: {ruta_relativa}")
    except ValueError:
        ruta_relativa = os.path.basename(carpeta_base)
        print(f" Usando nombre base: {ruta_relativa}")
    
    if not ruta_relativa:
        ruta_relativa = os.path.basename(carpeta_base)
        print(f" Respaldo - usando nombre de carpeta: {ruta_relativa}")
    
    # Lista para almacenar todos los resultados
    todos_los_resultados = []
    
    # Verificar contenido de la carpeta
    archivos_directos = [f for f in os.listdir(carpeta_base) 
                        if f.lower().endswith((".jpeg", ".jpg", ".png")) and 
                        os.path.isfile(os.path.join(carpeta_base, f))]
    
    subcarpetas = [d for d in os.listdir(carpeta_base) 
                  if os.path.isdir(os.path.join(carpeta_base, d))]
    
    inicio_total = time.time()
    
    # Procesar según la estructura encontrada
    if archivos_directos and not subcarpetas:
        print(f" Se encontraron {len(archivos_directos)} imágenes en la carpeta base")
        print("-" * 60)
        
        ruta_salida = os.path.join(carpeta_salida_base, ruta_relativa)
        resultados = procesar_carpeta_paralelo(carpeta_base, ruta_salida)
        todos_los_resultados.extend(resultados)
    
    elif subcarpetas:
        print(f" Se encontraron {len(subcarpetas)} subcarpetas: {subcarpetas}")
        if archivos_directos:
            print(f" También hay {len(archivos_directos)} imágenes en la carpeta base")
        print("-" * 60)
        
        # Procesar imágenes en carpeta base si las hay
        if archivos_directos:
            print(f"\n Procesando imágenes de la carpeta base")
            ruta_salida_base_imgs = os.path.join(carpeta_salida_base, ruta_relativa)
            resultados_base = procesar_carpeta_paralelo(carpeta_base, ruta_salida_base_imgs)
            todos_los_resultados.extend(resultados_base)
        
        # Procesar subcarpetas
        for subcarpeta in subcarpetas:
            print(f"\n Procesando subcarpeta: {subcarpeta}")
            
            ruta_subcarpeta = os.path.join(carpeta_base, subcarpeta)
            ruta_salida_subcarpeta = os.path.join(carpeta_salida_base, ruta_relativa, subcarpeta)
            
            resultados_subcarpeta = procesar_carpeta_paralelo(ruta_subcarpeta, ruta_salida_subcarpeta)
            
            # Agregar información de subcarpeta
            for resultado in resultados_subcarpeta:
                resultado.insert(0, subcarpeta)
            
            todos_los_resultados.extend(resultados_subcarpeta)
    else:
        print(" No se encontraron imágenes ni subcarpetas")
        return
    
    fin_total = time.time()
    tiempo_total_final = fin_total - inicio_total
    
    # Guardar Excel
    if todos_los_resultados:
        if subcarpetas:
            columnas = ["Subcarpeta", "Archivo", "Brillo Original", "Clasificación", "Brillo Corregido"]
        else:
            columnas = ["Archivo", "Brillo Original", "Clasificación", "Brillo Corregido"]
            todos_los_resultados = [[r[1], r[2], r[3], r[4]] if len(r) == 5 else r for r in todos_los_resultados]
        
        df = pd.DataFrame(todos_los_resultados, columns=columnas)
        excel_salida = os.path.join(carpeta_salida_base, ruta_relativa, "resultados_correccion_final.xlsx")
        os.makedirs(os.path.dirname(excel_salida), exist_ok=True)
        df.to_excel(excel_salida, index=False)
        
        total_imagenes = len(todos_los_resultados)
        velocidad_promedio = total_imagenes / tiempo_total_final if tiempo_total_final > 0 else 0
        
        print(f"\n Archivo Excel guardado en: {excel_salida}")
        print(f" Total de imágenes procesadas: {total_imagenes}")
        print(f" Tiempo total: {tiempo_total_final:.2f}s")
        print(f" Velocidad promedio: {velocidad_promedio:.1f} imágenes/segundo")
        print(f" Mejora estimada: {velocidad_promedio:.1f}x más rápido que 1 img/s")
    else:
        print("\n No se procesaron imágenes")
    
    print("\n ¡Procesamiento optimizado completado!")
    print(f" Imágenes guardadas en: {os.path.join(carpeta_salida_base, ruta_relativa)}")

if __name__ == "__main__":
    # Necesario para Windows y multiprocessing
    multiprocessing.freeze_support()
    main()
    