import os
import cv2
import numpy as np
import pandas as pd
#nueva mejora
# --- CONFIGURACIÓN ---
carpeta_base = r"C:\Users\ANGEL GOMEZ\OneDrive\Documentos\BC-163-02\S2C2"
carpeta_salida_base = r"C:\Users\ANGEL GOMEZ\Proyectos\Proyecto_semic\Correccion_final"
ruta_relativa = ""
def aplicar_clahe_adaptativo(img):
    # Convertir a LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    # Medir brillo promedio en canal L
    brillo = np.mean(l)

    # --- Ajustar parámetros según brillo ---
    if brillo < 80:
        # Imagen oscura → más contraste
        clip = 1.0
        tiles = (32, 32)
    elif brillo > 160:
        # Imagen muy clara → contraste suave
        clip = 0.8
        tiles = (32, 32)
    else:
        # Imagen balanceada → valores intermedios
        clip = 0.7
        tiles = (32, 32)

    # Crear CLAHE con los parámetros elegidos
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l_eq = clahe.apply(l)

    # Recomponer imagen LAB → BGR
    img_lab_eq = cv2.merge([l_eq, a, b])
    img_result = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)

    return img_result, brillo, clip, tiles

def reducir_ruido(img):
    # 1️⃣ Bilateral Filter (mantiene bordes)
    img_bilateral = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    return img_bilateral

# ---------------- FUNCIONES ----------------

def ajustar_gamma(img, gamma=1):
    inv_gamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, tabla)

def reducir_highlights(img, threshold=250, factor=0.8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mask = (l > threshold).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (51, 51), 30)
    l_adjusted = l.astype(np.float32) * (1 - mask * (1 - factor))
    l_adjusted = np.clip(l_adjusted, 0, 240).astype(np.uint8)
    lab_adjusted = cv2.merge([l_adjusted, a, b])
    return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

def levantar_sombras(img, threshold=40, factor=1.15):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Brillo promedio
    brillo_promedio = np.mean(v)
    
    # Máscara de sombras
    mask = v < threshold
    
    # Ajuste extra si la imagen está muy oscura
    factor_final = factor
    if brillo_promedio < 35:
        factor_final = 0  
    
    # Aplicar aclarado
    v[mask] = np.clip(v[mask] * factor_final, 0, 255)
    
    # Recomponer HSV → BGR
    img_sombras = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    # Aplicar gamma extra solo si muy oscura
    if brillo_promedio < 40:
        img_sombras = ajustar_gamma(img_sombras, gamma=1.1)

    return img_sombras

def tone_mapping_local(img, threshold=200, intensidad=0.5, log_factor=3):
    img_float = img.astype(np.float32) / 255.0
    mask = np.mean(img_float, axis=2) < threshold/255.0
    img_log = np.log1p(img_float * log_factor) / np.log1p(log_factor)
    img_out = img_float.copy()
    img_out[mask] = img_log[mask] + (img_log[mask] - img_float[mask]) * intensidad
    return (img_out * 255).astype(np.uint8)

def ajustar_curva_s(img):
    lookUpTable = np.empty((1, 256), np.uint8)
    
    for i in range(256):
        val = i / 255.0
        
        if val > 0.8:
            new_val = 0.8 + (val - 0.8) * 0.2
        else:
            new_val = val
        
        lookUpTable[0, i] = np.clip(new_val * 255, 0, 255)
    
    return cv2.LUT(img, lookUpTable)

def oscurecer_areas_quemadas(img, umbral=180, factor=0.7):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mascara = v > umbral

    if np.any(mascara):
        v_superiores = v[mascara].astype(np.float32)
        degradado = (v_superiores - umbral) / (255.0 - umbral)
        factor_gradual = 1.0 - (degradado * (1.0 - factor))
        v[mascara] = np.clip(v_superiores * factor_gradual, 0, 255).astype(np.uint8)

    hsv_modificado = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_modificado, cv2.COLOR_HSV2BGR)

def procesar_imagen_avanzado(img):
    """Aplica todas las correcciones avanzadas"""
    
    # 1. CLAHE
    img_procesada, brillo, clip, tiles = aplicar_clahe_adaptativo(img)
    print(f"    CLAHE → Brillo={brillo:.2f}, clipLimit={clip}, tileGridSize={tiles}")
    
    # 2. Reducción de ruido
    img_procesada = reducir_ruido(img_procesada)
    
    # 3. Levantar sombras
    img_procesada = levantar_sombras(img_procesada, threshold=40, factor=1.15)
    
    # 4. Reducir highlights
    img_procesada = reducir_highlights(img_procesada, threshold=250, factor=0.8)
    
    # 5. Tone mapping local
    img_procesada = tone_mapping_local(img_procesada, threshold=245)
    
    # 6. Ajuste de gamma
    img_procesada = ajustar_gamma(img_procesada, gamma=0.79)
    
    # 7. Saturación
    hsv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.9, 0, 255).astype(np.uint8)
    img_procesada = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    return img_procesada

def procesar_carpeta(ruta_carpeta, ruta_salida_carpeta):
    """Procesa todas las imágenes de una carpeta específica"""
    
    resultados = []
    
    # Crear la carpeta de salida si no existe
    os.makedirs(ruta_salida_carpeta, exist_ok=True)
    
    # Verificar si la carpeta existe
    if not os.path.exists(ruta_carpeta):
        print(f"  ⚠️ La carpeta no existe: {ruta_carpeta}")
        return resultados
    
    # Obtener lista de archivos de imagen
    archivos_imagen = [f for f in os.listdir(ruta_carpeta) 
                      if f.lower().endswith((".jpeg", ".jpg", ".png"))]
    
    if not archivos_imagen:
        print(f"  ⚠️ No se encontraron imágenes en: {ruta_carpeta}")
        return resultados
    
    print(f"  📁 Procesando {len(archivos_imagen)} imágenes...")
    
    for archivo in archivos_imagen:
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        img = cv2.imread(ruta_completa)
        
        if img is None:
            print(f"    ❌ No se pudo leer: {archivo}")
            continue
        
        # Calcular brillo original
        gris_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brillo_original = np.mean(gris_original)
        
        # Procesar imagen
        print(f"    🔄 Procesando: {archivo}")
        img_corregida = procesar_imagen_avanzado(img)
        
        # Calcular brillo corregido
        gris_corregido = cv2.cvtColor(img_corregida, cv2.COLOR_BGR2GRAY)
        brillo_corregido = np.mean(gris_corregido)
        
        # Guardar imagen procesada
        ruta_salida_imagen = os.path.join(ruta_salida_carpeta, archivo)
        cv2.imwrite(ruta_salida_imagen, img_corregida)
        
        # Guardar resultados para Excel
        resultados.append([archivo, brillo_original, "Corrección aplicada", brillo_corregido])
        print(f"    ✅ {archivo}: Brillo Original={brillo_original:.2f} | Brillo Corregido={brillo_corregido:.2f}")
    
    return resultados

# ---------------- PROCESAMIENTO PRINCIPAL ----------------

def main():
    print("🚀 Iniciando procesamiento de imágenes...")
    print(f"📂 Carpeta base: {carpeta_base}")
    print(f"💾 Carpeta de salida: {carpeta_salida_base}")
    print("-" * 60)
    
    # Verificar si la carpeta base existe
    if not os.path.exists(carpeta_base):
        print(f"❌ Error: La carpeta base no existe: {carpeta_base}")
        return
    
    # Obtener el nombre de la carpeta base (ej: "S2C2")
    nombre_carpeta_base = os.path.basename(carpeta_base)
    
    # Lista para almacenar todos los resultados
    todos_los_resultados = []
    
    # Obtener todas las subcarpetas
    subcarpetas = [d for d in os.listdir(carpeta_base) 
                  if os.path.isdir(os.path.join(carpeta_base, d))]
    
    if not subcarpetas:
        print(f"⚠️ No se encontraron subcarpetas en: {carpeta_base}")
        return
    
    print(f"📋 Se encontraron {len(subcarpetas)} subcarpetas: {subcarpetas}")
    print("-" * 60)
    
    for subcarpeta in subcarpetas:
        print(f"\n🏗️ Procesando subcarpeta: {subcarpeta}")
        
        # Ruta completa de la subcarpeta de origen
        ruta_subcarpeta = os.path.join(carpeta_base, subcarpeta)
        
        # Ruta completa de la subcarpeta de destino
        ruta_salida_subcarpeta = os.path.join(carpeta_salida_base, nombre_carpeta_base, subcarpeta)
        
        # Procesar la subcarpeta
        resultados_subcarpeta = procesar_carpeta(ruta_subcarpeta, ruta_salida_subcarpeta)
        
        # Agregar información de la subcarpeta a los resultados
        for resultado in resultados_subcarpeta:
            resultado.insert(0, subcarpeta)  # Agregar nombre de subcarpeta al inicio
        
        todos_los_resultados.extend(resultados_subcarpeta)
        
        print(f"  ✅ Subcarpeta {subcarpeta} completada: {len(resultados_subcarpeta)} imágenes procesadas")
    
    # Guardar resultados consolidados en Excel
    if todos_los_resultados:
        # Determinar las columnas según el tipo de estructura
        if subcarpetas:
            # Si hay subcarpetas, incluir columna de subcarpeta
            columnas = ["Subcarpeta", "Archivo", "Brillo Original", "Clasificación", "Brillo Corregido"]
        else:
            # Si solo hay imágenes en carpeta base, sin columna de subcarpeta
            columnas = ["Archivo", "Brillo Original", "Clasificación", "Brillo Corregido"]
            # Remover la primera columna agregada (subcarpeta) si existe
            todos_los_resultados = [[r[1], r[2], r[3], r[4]] if len(r) == 5 else r for r in todos_los_resultados]
        
        df = pd.DataFrame(todos_los_resultados, columns=columnas)
        
        excel_salida = os.path.join(carpeta_salida_base, ruta_relativa, "resultados_correccion_final.xlsx")
        
        # Crear la carpeta padre del Excel si no existe
        os.makedirs(os.path.dirname(excel_salida), exist_ok=True)
        
        df.to_excel(excel_salida, index=False)
        
        print(f"\n📊 Archivo Excel guardado en: {excel_salida}")
        print(f"📈 Total de imágenes procesadas: {len(todos_los_resultados)}")
    else:
        print("\n⚠️ No se procesaron imágenes")
    
    print("\n🎉 ¡Procesamiento completado!")
    print(f"📁 Todas las imágenes procesadas se guardaron en: {os.path.join(carpeta_salida_base, ruta_relativa)}")

if __name__ == "__main__":
    main()