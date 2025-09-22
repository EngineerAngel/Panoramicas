import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import multiprocessing
import time
import gc
import psutil
import logging
from pathlib import Path
import json

# ============================================
# CONFIGURACIÓN (Puedes mantenerla como está o ajustarla)
# ============================================
TOTAL_CPUS = multiprocessing.cpu_count()
NUM_PROCESOS = max(1, TOTAL_CPUS - 1)
BATCH_SIZE = 50
MEMORY_THRESHOLD_PERCENT = 90

# ============================================
# CLASES DE GESTIÓN (Sin cambios)
# ============================================

class ProcesamientoEstado:
    """Gestiona el estado del procesamiento para recuperación ante fallos"""
    def __init__(self, carpeta_salida):
        self.archivo_estado = os.path.join(carpeta_salida, '.procesamiento_estado.json')
        self.estado = self.cargar_estado()
    
    def cargar_estado(self):
        if os.path.exists(self.archivo_estado):
            try:
                with open(self.archivo_estado, 'r') as f:
                    return json.load(f)
            except:
                return {'procesadas': [], 'errores': []}
        return {'procesadas': [], 'errores': []}
    
    def guardar_estado(self):
        try:
            with open(self.archivo_estado, 'w') as f:
                json.dump(self.estado, f)
        except Exception as e:
            logging.error(f"Error guardando estado: {e}")
    
    def agregar_procesada(self, archivo):
        self.estado['procesadas'].append(archivo)
    
    def ya_procesado(self, archivo):
        return archivo in self.estado['procesadas']

class MonitorRecursos:
    """Monitorea el uso de recursos del sistema"""
    @staticmethod
    def esperar_recursos():
        while psutil.virtual_memory().percent > MEMORY_THRESHOLD_PERCENT:
            logging.warning(f"Memoria al {psutil.virtual_memory().percent:.1f}%. Pausando...")
            time.sleep(5)
            gc.collect()

# ============================================
# LÓGICA DE PROCESAMIENTO
# ============================================

def procesar_imagen_ligero(datos_imagen, cancel_signal_path):
    """Procesa una imagen individual y verifica la señal de cancelación."""
    ruta_entrada, ruta_salida, archivo = datos_imagen
    
    # --- NUEVO: Verificación de cancelación ---
    if os.path.exists(cancel_signal_path):
        return None, f"🛑 {archivo} (Cancelado)"
        
    try:
        if os.path.exists(ruta_salida):
            return None, f"⏭️ {archivo} (ya existe)"
        
        img = cv2.imread(ruta_entrada)
        if img is None:
            return None, f"❌ No se pudo leer: {archivo}"
        
        # AQUÍ VA TU LÓGICA DE PROCESAMIENTO
        img_procesada = cv2.bilateralFilter(img, d=5, sigmaColor=20, sigmaSpace=20)
        
        success = cv2.imwrite(ruta_salida, img_procesada, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        del img, img_procesada
        
        if success:
            return archivo, f"✅ {archivo}"
        else:
            return None, f"❌ No se pudo guardar: {archivo}"
            
    except Exception as e:
        return None, f"❌ Error en {archivo}: {str(e)}"
    finally:
        gc.collect()

def procesar_carpeta_alto_volumen(ruta_carpeta, ruta_salida_carpeta, gui_queue, cancel_signal_path, reanudar=True):
    """Función principal de procesamiento que se comunica con la GUI."""
    
    def reportar(mensaje):
        if gui_queue:
            gui_queue.put(mensaje)
        else:
            print(mensaje)

    os.makedirs(ruta_salida_carpeta, exist_ok=True)
    estado = ProcesamientoEstado(ruta_salida_carpeta) if reanudar else None
    
    reportar("📂 Escaneando carpeta de entrada...")
    extensiones = (".jpeg", ".jpg", ".png")
    archivos_imagen = [f for f in Path(ruta_carpeta).rglob('*') if f.is_file() and f.name.lower().endswith(extensiones)]
    
    if estado and reanudar:
        procesadas = set(estado.estado['procesadas'])
        archivos_pendientes = [f for f in archivos_imagen if f.name not in procesadas]
        reportar(f"📊 Reanudando: {len(procesadas)} ya procesadas, {len(archivos_pendientes)} pendientes.")
        archivos_imagen = archivos_pendientes

    total_imagenes = len(archivos_imagen)
    if total_imagenes == 0:
        reportar("✅ No hay imágenes nuevas para procesar.")
        if gui_queue: gui_queue.put("DONE")
        return

    reportar(f"🖼️ {total_imagenes} imágenes a procesar con {NUM_PROCESOS} procesos.")

    datos_procesamiento = []
    carpeta_padre = Path(ruta_carpeta).parent
    for archivo in archivos_imagen:
        ruta_relativa = archivo.relative_to(carpeta_padre)
        ruta_salida = Path(ruta_salida_carpeta) / ruta_relativa
        os.makedirs(ruta_salida.parent, exist_ok=True)
        datos_procesamiento.append((str(archivo), str(ruta_salida), archivo.name))
    
    imagenes_procesadas = 0
    errores = 0
    
    try:
        with ProcessPoolExecutor(max_workers=NUM_PROCESOS) as executor:
            for lote_inicio in range(0, total_imagenes, BATCH_SIZE):
                if os.path.exists(cancel_signal_path):
                    reportar("🛑 Cancelación detectada entre lotes.")
                    break

                MonitorRecursos.esperar_recursos()
                
                lote_fin = min(lote_inicio + BATCH_SIZE, total_imagenes)
                lote = datos_procesamiento[lote_inicio:lote_fin]
                lote_num = (lote_inicio // BATCH_SIZE) + 1
                total_lotes = (total_imagenes + BATCH_SIZE - 1) // BATCH_SIZE
                
                reportar(f"\n📦 Procesando lote {lote_num}/{total_lotes} (Imágenes {lote_inicio+1}-{lote_fin})")
                
                # Pasamos la señal de cancelación a cada worker
                futuros = {executor.submit(procesar_imagen_ligero, datos, cancel_signal_path): datos for datos in lote}
                
                for futuro in as_completed(futuros):
                    resultado, mensaje = futuro.result()
                    reportar(f"  {mensaje}")
                    
                    if resultado:
                        imagenes_procesadas += 1
                        if estado:
                            estado.agregar_procesada(resultado)
                    elif "Error" in mensaje:
                        errores += 1

                if estado:
                    estado.guardar_estado()
                    reportar(f"💾 Checkpoint guardado después del lote {lote_num}.")
    
    except Exception as e:
        reportar(f"💥 Error crítico: {e}")
    
    finally:
        reportar("\n" + "="*50)
        reportar("📊 RESUMEN FINAL")
        reportar("="*50)
        reportar(f"✅ Imágenes procesadas en esta sesión: {imagenes_procesadas}")
        reportar(f"❌ Errores en esta sesión: {errores}")
        if estado:
            reportar(f"📈 Total acumulado en carpeta: {len(estado.estado['procesadas'])} procesadas.")
            estado.guardar_estado()

        if gui_queue:
            gui_queue.put("DONE")
