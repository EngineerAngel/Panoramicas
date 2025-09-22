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

# === FUNCIONES BÁSICAS ===
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

# === SISTEMA DE CLASIFICACIÓN ===
def clasificar_nivel_oscuridad(img):
    """Clasifica la imagen según su nivel de oscuridad"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brillo_promedio = np.mean(gray)
    
    # Calcular percentiles para mejor análisis
    p10 = np.percentile(gray, 10)
    p50 = np.percentile(gray, 50)
    p90 = np.percentile(gray, 90)
    
    # Calcular porcentaje de píxeles muy oscuros
    pixeles_muy_oscuros = np.sum(gray < 30) / gray.size * 100
    pixeles_oscuros = np.sum(gray < 80) / gray.size * 100
    
    # Clasificación detallada
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

# === PROCESAMIENTO ESPECÍFICO POR CLASIFICACIÓN ===
def procesar_extremadamente_oscura(img):
    """Procesamiento específico para imágenes extremadamente oscuras (túneles, interiores muy oscuros)"""
    
    # 1. Pre-procesamiento suave
    img_procesada = cv2.bilateralFilter(img, d=3, sigmaColor=5, sigmaSpace=5)
    
    # 2. CLAHE muy suave para evitar ruido
    lab = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=0.9, tileGridSize=(8, 8))  # Tiles más grandes
    l = clahe.apply(l)
    
    img_procesada = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # 3. Levantamiento de sombras muy gradual
    hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Máscara gradual para sombras extremas
    mask_extrema = (v < 50).astype(np.float32)
    mask_suave = cv2.GaussianBlur(mask_extrema, (15, 15), 8)
    
    v_float = v.astype(np.float32)
    # Factor más agresivo pero aplicado gradualmente
    v_float = v_float + mask_suave * (v_float * 0.25)  # Aumentar 80% gradualmente
    v = np.clip(v_float, 0, 255).astype(np.uint8)
    
    img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    # 4. Gamma correction específica para extremos
    img_procesada = ajustar_gamma(img_procesada, gamma=1.1)  # Más agresivo para extremos
    
    # 5. Reducción de ruido final
    img_procesada = cv2.bilateralFilter(img_procesada, d=8, sigmaColor=15, sigmaSpace=15)
    
    # 6. Saturación muy conservadora
    hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.1, 0, 255).astype(np.uint8)  # Muy sutil
    img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    return img_procesada

def procesar_muy_oscura(img):
    """Procesamiento para imágenes muy oscuras"""
    
    # 1. CLAHE moderado
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))
    l = clahe.apply(l)
    img_procesada = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # 2. Levantamiento de sombras graduado
    hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    mask = v < 60
    v_float = v.astype(np.float32)
    v_float[mask] = np.clip(v_float[mask] * 1.4, 0, 255)
    v = v_float.astype(np.uint8)
    
    img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    # 3. Gamma suave
    img_procesada = ajustar_gamma(img_procesada, gamma=1.0)
    
    # 4. Saturación moderada
    hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.15, 0, 255).astype(np.uint8)
    img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    return img_procesada

def aplicar_clahe_adaptativo_v2(img, modo="normal"):
    """CLAHE adaptativo según el modo"""
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    brillo = np.mean(l)

    if modo == "oscura":
        clip, tiles = 2.0, (12, 12)
    elif modo == "clara":
        clip, tiles = 1.2, (20, 20)
    else:  # normal
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
    """Versión mejorada de levantar sombras"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    brillo_promedio = np.mean(v)
    mask = v < threshold
    
    # Evitar el factor 0 que causa pixelación
    factor_final = max(factor, 1.02)  # Mínimo 1.02
    
    v_float = v.astype(np.float32)
    v_float[mask] = np.clip(v_float[mask] * factor_final, 0, 255)
    v = v_float.astype(np.uint8)
    
    img_sombras = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    # Gamma más conservador
    if brillo_promedio < 30:
        img_sombras = ajustar_gamma(img_sombras, gamma=0.85)

    return img_sombras

def procesar_imagen_avanzado_optimizado_v2(img, modo="normal"):
    """Versión mejorada con modos específicos"""
    
    if modo == "oscura":
        # Parámetros para imágenes oscuras pero no extremas
        img_procesada, brillo, clip, tiles = aplicar_clahe_adaptativo_v2(img, modo="oscura")
        img_procesada = reducir_ruido(img_procesada)
        img_procesada = levantar_sombras_v2(img_procesada, threshold=70, factor=1.25)
        img_procesada = reducir_highlights(img_procesada, threshold=240, factor=0.9)
        img_procesada = ajustar_gamma(img_procesada, gamma=0.8)
        
        # Saturación conservadora para oscuras
        hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.15, 0, 255).astype(np.uint8)
        img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        
    elif modo == "clara":
        # CLAHE adaptativo
        img_procesada, brillo, clip, tiles = aplicar_clahe_adaptativo_v2(img, modo="clara")
    
        # Reducción de ruido
        img_procesada = reducir_ruido(img_procesada)

        # Ajuste gamma suave
        img_procesada = ajustar_gamma(img_procesada, gamma=0.95)

        # Ajuste de saturación
        hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.1, 0, 255).astype(np.uint8)

        # Reducción de highlights extremos
        mask_clara = (v > 240).astype(np.float32)
        mask_clara_suave = cv2.GaussianBlur(mask_clara, (15, 15), 8)
        v_float = v.astype(np.float32)
        v_float = v_float - mask_clara_suave * ((v_float - 240) * 0.6)
        v = np.clip(v_float, 0, 255).astype(np.uint8)

        img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        
    else:  # normal
        # Parámetros equilibrados
        img_procesada, brillo, clip, tiles = aplicar_clahe_adaptativo_v2(img, modo="normal")
        img_procesada = reducir_ruido(img_procesada)
        img_procesada = levantar_sombras_v2(img_procesada, threshold=60, factor=1.08)
        img_procesada = reducir_highlights(img_procesada, threshold=240, factor=0.85)
        img_procesada = ajustar_gamma(img_procesada, gamma=0.88)
        
        # Saturación moderada
        hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
        img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    return img_procesada, brillo

def procesar_segun_clasificacion(img, clasificacion):
    """Aplica el procesamiento según la clasificación de oscuridad"""
    
    if clasificacion == "EXTREMADAMENTE_OSCURA":
        img_procesada = procesar_extremadamente_oscura(img)
        brillo = np.mean(cv2.cvtColor(img_procesada, cv2.COLOR_BGR2GRAY))
        return img_procesada, brillo
    
    elif clasificacion == "MUY_OSCURA":
        img_procesada = procesar_muy_oscura(img)
        brillo = np.mean(cv2.cvtColor(img_procesada, cv2.COLOR_BGR2GRAY))
        return img_procesada, brillo
    
    elif clasificacion == "OSCURA":
        # Procesamiento original pero con valores corregidos
        return procesar_imagen_avanzado_optimizado_v2(img, modo="oscura")
    
    elif clasificacion in ["MEDIO_OSCURA", "NORMAL"]:
        # Procesamiento estándar
        return procesar_imagen_avanzado_optimizado_v2(img, modo="normal")
    
    else:  # CLARA, MUY_CLARA
        # Procesamiento para imágenes claras
        return procesar_imagen_avanzado_optimizado_v2(img, modo="clara")

# === FUNCIÓN DE PROCESAMIENTO INDIVIDUAL ===
def procesar_imagen_individual_v2(datos_imagen):
    """Función mejorada con clasificación por oscuridad"""
    ruta_entrada, ruta_salida, archivo = datos_imagen
    
    try:
        # Leer imagen
        img = cv2.imread(ruta_entrada)
        if img is None:
            return None, f"No se pudo leer: {archivo}"
        
        # Calcular brillo original
        gris_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brillo_original = np.mean(gris_original)
        
        # CLASIFICAR NIVEL DE OSCURIDAD
        clasificacion, brillo_prom, p10, p50, p90, pixeles_oscuros = clasificar_nivel_oscuridad(img)
        
        # PROCESAR SEGÚN CLASIFICACIÓN
        img_corregida, brillo_procesado = procesar_segun_clasificacion(img, clasificacion)
        
        # Calcular brillo corregido
        gris_corregido = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2GRAY)
        brillo_corregido = np.mean(gris_corregido)
        
        # Guardar imagen
        cv2.imwrite(ruta_salida, img_corregida)
        
        # Retornar información detallada
        return [archivo, brillo_original, clasificacion, brillo_corregido, f"{pixeles_oscuros:.1f}%"], f"✅ {archivo} ({clasificacion})"
        
    except Exception as e:
        return None, f"❌ Error procesando {archivo}: {str(e)}"

# === PROCESAMIENTO PARALELO ===
def procesar_carpeta_paralelo_v2(ruta_carpeta, ruta_salida_carpeta, usar_procesos=True):
    """Versión mejorada con clasificación automática"""
    
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
    
    print(f"   Procesando {len(archivos_imagen)} imágenes con clasificación automática...")
    
    # Preparar datos para procesamiento paralelo
    datos_procesamiento = [
        (os.path.join(ruta_carpeta, archivo), 
         os.path.join(ruta_salida_carpeta, archivo), 
         archivo)
        for archivo in archivos_imagen
    ]
    
    inicio = time.time()
    
    if usar_procesos and len(archivos_imagen) > 4:
        # Usar ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=NUM_PROCESOS) as executor:
            resultados_paralelos = list(executor.map(procesar_imagen_individual_v2, datos_procesamiento))
    else:
        # Usar ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            resultados_paralelos = list(executor.map(procesar_imagen_individual_v2, datos_procesamiento))
    
    fin = time.time()
    tiempo_total = fin - inicio
    
    # Contar clasificaciones
    clasificaciones = {}
    for resultado, mensaje in resultados_paralelos:
        if resultado is not None:
            clasificacion = resultado[2]
            clasificaciones[clasificacion] = clasificaciones.get(clasificacion, 0) + 1
            resultados.append(resultado)
        print(f"    {mensaje}")
    
    # Mostrar estadísticas de clasificación
    print(f"   📊 Clasificaciones:")
    for clasif, count in clasificaciones.items():
        print(f"      {clasif}: {count} imágenes")
    
    velocidad = len(archivos_imagen) / tiempo_total if tiempo_total > 0 else 0
    print(f"   Completado en {tiempo_total:.2f}s ({velocidad:.1f} img/s)")
    
    return resultados

# === FUNCIÓN PRINCIPAL ===
def main():
    """Función main actualizada con el nuevo sistema"""
    print("🚀 Iniciando procesamiento INTELIGENTE con clasificación automática...")
    print(f"📁 Carpeta base: {carpeta_base}")
    print(f"💾 Carpeta de salida: {carpeta_salida_base}")
    print(f"⚙️  Configuración: {NUM_PROCESOS} procesos, {NUM_THREADS} threads")
    print("🧠 Clasificación automática por nivel de oscuridad activada")
    print("-" * 70)
    
    # Verificar si la carpeta base existe
    if not os.path.exists(carpeta_base):
        print(f"❌ Error: La carpeta base no existe: {carpeta_base}")
        return
    
    # DEFINIR ruta_relativa AL INICIO
    partes_ruta = carpeta_base.split(os.sep)
    ruta_relativa = ""
    
    try:
        indice_documentos = partes_ruta.index("Documentos")
        ruta_relativa = os.sep.join(partes_ruta[indice_documentos + 1:])
        print(f"📋 Estructura detectada: {ruta_relativa}")
    except ValueError:
        ruta_relativa = os.path.basename(carpeta_base)
        print(f"📋 Usando nombre base: {ruta_relativa}")
    
    if not ruta_relativa:
        ruta_relativa = os.path.basename(carpeta_base)
        print(f"📋 Respaldo - usando nombre de carpeta: {ruta_relativa}")
    
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
        print(f"🖼️  Se encontraron {len(archivos_directos)} imágenes en la carpeta base")
        print("-" * 70)
        
        ruta_salida = os.path.join(carpeta_salida_base, ruta_relativa)
        resultados = procesar_carpeta_paralelo_v2(carpeta_base, ruta_salida)
        todos_los_resultados.extend(resultados)
    
    elif subcarpetas:
        print(f"📂 Se encontraron {len(subcarpetas)} subcarpetas: {subcarpetas}")
        if archivos_directos:
            print(f"🖼️  También hay {len(archivos_directos)} imágenes en la carpeta base")
        print("-" * 70)
        
        # Procesar imágenes en carpeta base si las hay
        if archivos_directos:
            print(f"\n🖼️  Procesando imágenes de la carpeta base")
            ruta_salida_base_imgs = os.path.join(carpeta_salida_base, ruta_relativa)
            resultados_base = procesar_carpeta_paralelo_v2(carpeta_base, ruta_salida_base_imgs)
            todos_los_resultados.extend(resultados_base)
        
        # Procesar subcarpetas
        for subcarpeta in subcarpetas:
            print(f"\n📂 Procesando subcarpeta: {subcarpeta}")
            
            ruta_subcarpeta = os.path.join(carpeta_base, subcarpeta)
            ruta_salida_subcarpeta = os.path.join(carpeta_salida_base, ruta_relativa, subcarpeta)
            
            resultados_subcarpeta = procesar_carpeta_paralelo_v2(ruta_subcarpeta, ruta_salida_subcarpeta)
            
            # Agregar información de subcarpeta
            for resultado in resultados_subcarpeta:
                resultado.insert(0, subcarpeta)
            
            todos_los_resultados.extend(resultados_subcarpeta)
    else:
        print("❌ No se encontraron imágenes ni subcarpetas")
        return
    
    fin_total = time.time()
    tiempo_total_final = fin_total - inicio_total
    
    # Guardar Excel con información detallada
    if todos_los_resultados:
        if subcarpetas:
            columnas = ["Subcarpeta", "Archivo", "Brillo Original", "Clasificación", "Brillo Corregido", "% Píxeles Oscuros"]
        else:
            columnas = ["Archivo", "Brillo Original", "Clasificación", "Brillo Corregido", "% Píxeles Oscuros"]
        
        df = pd.DataFrame(todos_los_resultados, columns=columnas)
        # Asegurar que la ruta del excel se crea correctamente
        ruta_excel_dir = os.path.join(carpeta_salida_base, ruta_relativa)
        os.makedirs(ruta_excel_dir, exist_ok=True)
        excel_salida = os.path.join(ruta_excel_dir, "resultados_procesamiento_inteligente.xlsx")
        
        df.to_excel(excel_salida, index=False)
        
        # Estadísticas finales
        total_imagenes = len(todos_los_resultados)
        velocidad_promedio = total_imagenes / tiempo_total_final if tiempo_total_final > 0 else 0
        
        # Contar clasificaciones totales
        clasificaciones_totales = {}
        col_clasificacion = "Clasificación" if "Clasificación" in df.columns else df.columns[2]
        for clasif in df[col_clasificacion]:
            clasificaciones_totales[clasif] = clasificaciones_totales.get(clasif, 0) + 1
        
        print(f"\n" + "="*70)
        print(f"📊 ESTADÍSTICAS FINALES DEL PROCESO COMPLETO:")
        print(f"="*70)
        print(f"✅ Total de imágenes procesadas: {total_imagenes}")
        print(f"⏱️  Tiempo total de ejecución: {tiempo_total_final:.2f} segundos")
        print(f"🚀 Velocidad promedio: {velocidad_promedio:.1f} imágenes por segundo")
        print(f"📈 Resumen de clasificaciones:")
        for clasif, count in clasificaciones_totales.items():
            print(f"    - {clasif}: {count} imágenes")
        print(f"💾 Informe guardado en: {excel_salida}")
        print(f"="*70)

# --- PUNTO DE ENTRADA DEL SCRIPT ---
if __name__ == "__main__":
    main()