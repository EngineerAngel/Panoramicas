import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor  # Usamos ThreadPool en lugar de ProcessPool
import time
import gc
import psutil
import logging
from pathlib import Path
import json

# Integración de módulos personalizados
import processing
import classification
import image_utils
# Importamos la configuración desde config.py
from config import NUM_PROCESOS, BATCH_SIZE, MEMORY_THRESHOLD_PERCENT

# ============================================
# CLASES DE GESTIÓN (SIN cambios críticos)
# ============================================

class ProcesamientoEstado:
    """Gestiona el estado del procesamiento para recuperación ante fallos"""
    def __init__(self, carpeta_salida):
        self.archivo_estado = os.path.join(carpeta_salida, '.procesamiento_estado.json')
        self.estado = self.cargar_estado()
    
    def cargar_estado(self):
        if os.path.exists(self.archivo_estado):
            try:
                with open(self.archivo_estado, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"No se pudo cargar estado anterior: {e}")
                return {'procesadas': [], 'errores': [], 'timestamp': time.time()}
        return {'procesadas': [], 'errores': [], 'timestamp': time.time()}
    
    def guardar_estado(self):
        try:
            self.estado['timestamp'] = time.time()
            with open(self.archivo_estado, 'w', encoding='utf-8') as f:
                json.dump(self.estado, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error guardando estado: {e}")
    
    def agregar_procesada(self, archivo_relativo):
        """Guarda la ruta relativa del archivo procesado"""
        if archivo_relativo not in self.estado['procesadas']:
            self.estado['procesadas'].append(archivo_relativo)
    
    def agregar_error(self, archivo, error):
        self.estado['errores'].append({
            'archivo': archivo,
            'error': str(error),
            'timestamp': time.time()
        })
    
    def ya_procesado(self, archivo_relativo):
        """Verifica si un archivo ya fue procesado usando la ruta relativa"""
        return archivo_relativo in self.estado['procesadas']
    
    def obtener_estadisticas(self):
        return {
            'procesadas': len(self.estado['procesadas']),
            'errores': len(self.estado['errores']),
            'ultimo_guardado': self.estado.get('timestamp', 0)
        }

class MonitorRecursos:
    """Monitorea el uso de recursos del sistema"""
    @staticmethod
    def esperar_recursos():
        while psutil.virtual_memory().percent > MEMORY_THRESHOLD_PERCENT:
            logging.warning(f"Memoria al {psutil.virtual_memory().percent:.1f}%. Pausando...")
            time.sleep(5)
            gc.collect()

# ============================================
# FUNCIONES DE CANCELACIÓN (sin cambios)
# ============================================

def verificar_cancelacion(cancel_signal_path):
    """Verifica si se ha solicitado cancelación"""
    if os.path.exists(cancel_signal_path):
        try:
            with open(cancel_signal_path, 'r') as f:
                contenido = f.read().strip()
                if contenido.startswith('cancel'):
                    return True
        except:
            pass
    return False

def marcar_cancelado(cancel_signal_path):
    """Marca el procesamiento como cancelado"""
    try:
        with open(cancel_signal_path + ".cancelled", 'w') as f:
            f.write(f'cancelled_{time.time()}')
    except:
        pass

# ============================================
# PROCESAMIENTO SIMPLIFICADO
# ============================================

def procesar_imagen_thread_safe(datos_imagen, cancel_signal_path):
    """Versión thread-safe del procesamiento de imagen"""
    ruta_entrada, ruta_salida, archivo_relativo = datos_imagen
    
    # Verificación de cancelación al inicio
    if verificar_cancelacion(cancel_signal_path):
        return None, f"Cancelado: {archivo_relativo}"
        
    try:
        # Verificar si ya existe y es válido
        if os.path.exists(ruta_salida):
            try:
                img_test = cv2.imread(ruta_salida)
                if img_test is not None:
                    return archivo_relativo, f"Ya existe: {archivo_relativo}"
            except:
                pass

        # Leer imagen original
        img = cv2.imread(ruta_entrada)
        if img is None:
            return None, f"Error leyendo: {archivo_relativo}"

        # Verificación de cancelación antes del procesamiento
        if verificar_cancelacion(cancel_signal_path):
            return None, f"Cancelado: {archivo_relativo}"

        # CLASIFICACIÓN Y PROCESAMIENTO
        nivel, brillo, p10, p50, p90, pixeles_muy_oscuros = classification.clasificar_nivel_oscuridad(img)
        
        if verificar_cancelacion(cancel_signal_path):
            return None, f"Cancelado: {archivo_relativo}"
        
        img_procesada, brillo_procesado = processing.procesar_segun_clasificacion(img, nivel)

        # Verificación final de cancelación
        if verificar_cancelacion(cancel_signal_path):
            return None, f"Cancelado: {archivo_relativo}"

        # GUARDADO
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        success = cv2.imwrite(ruta_salida, img_procesada, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Limpieza de memoria
        del img, img_procesada
        gc.collect()

        if success:
            return archivo_relativo, f"Procesado: {archivo_relativo} ({nivel})"
        else:
            return None, f"Error guardando: {archivo_relativo}"

    except Exception as e:
        return None, f"Error en {archivo_relativo}: {str(e)}"

def obtener_estructura_carpetas(ruta_entrada):
    """Obtiene información sobre la estructura de carpetas (sin cambios)"""
    try:
        path_entrada = Path(ruta_entrada)
        info_estructura = {
            'carpetas': [],
            'total_imagenes': 0,
            'imagenes_por_carpeta': {}
        }
        
        extensiones = (".jpeg", ".jpg", ".png", ".bmp", ".tiff")
        
        subcarpetas = [d for d in path_entrada.iterdir() if d.is_dir()]
        
        if subcarpetas:
            for subcarpeta in subcarpetas:
                archivos_img = []
                for ext in extensiones:
                    archivos_img.extend(list(subcarpeta.rglob(f'*{ext}')))
                    archivos_img.extend(list(subcarpeta.rglob(f'*{ext.upper()}')))
                
                archivos_img = list(set([f for f in archivos_img if f.is_file()]))
                
                info_estructura['carpetas'].append(subcarpeta.name)
                info_estructura['imagenes_por_carpeta'][subcarpeta.name] = len(archivos_img)
                info_estructura['total_imagenes'] += len(archivos_img)
        else:
            archivos_img = []
            for ext in extensiones:
                archivos_img.extend(list(path_entrada.rglob(f'*{ext}')))
                archivos_img.extend(list(path_entrada.rglob(f'*{ext.upper()}')))
            
            archivos_img = list(set([f for f in archivos_img if f.is_file()]))
            info_estructura['total_imagenes'] = len(archivos_img)
            
        return info_estructura
        
    except Exception as e:
        return {'error': str(e), 'total_imagenes': 0}

def procesar_carpeta_alto_volumen(ruta_carpeta, ruta_salida_carpeta, gui_queue, cancel_signal_path, reanudar=True):
    """
    Función principal SIMPLIFICADA que usa ThreadPoolExecutor en lugar de ProcessPoolExecutor
    para evitar problemas con multiprocessing en ejecutables
    """
    
    def reportar(mensaje):
        if gui_queue:
            try:
                gui_queue.put(mensaje)  # Queue normal, no multiprocessing.Queue
            except:
                print(mensaje)
        else:
            print(mensaje)

    # Configuración inicial
    os.makedirs(ruta_salida_carpeta, exist_ok=True)
    estado = ProcesamientoEstado(ruta_salida_carpeta) if reanudar else None
    
    if estado and reanudar:
        stats = estado.obtener_estadisticas()
        reportar(f"Estado anterior encontrado: {stats['procesadas']} imágenes ya procesadas")
    
    reportar("Analizando estructura de carpetas...")
    
    # Análisis de estructura
    info_estructura = obtener_estructura_carpetas(ruta_carpeta)
    
    if 'error' in info_estructura:
        reportar(f"Error analizando estructura: {info_estructura['error']}")
        if gui_queue: 
            gui_queue.put("DONE")
        return

    if info_estructura['carpetas']:
        reportar(f"Estructura detectada: {len(info_estructura['carpetas'])} subcarpetas")
    
    reportar(f"Total de imágenes encontradas: {info_estructura['total_imagenes']}")
    
    if info_estructura['total_imagenes'] == 0:
        reportar("No se encontraron imágenes para procesar.")
        if gui_queue: 
            gui_queue.put("DONE")
        return

    # Buscar archivos manteniendo estructura
    extensiones = (".jpeg", ".jpg", ".png", ".bmp", ".tiff")
    archivos_imagen = []
    path_entrada = Path(ruta_carpeta)
    
    try:
        for ext in extensiones:
            archivos_imagen.extend(path_entrada.rglob(f'*{ext}'))
            archivos_imagen.extend(path_entrada.rglob(f'*{ext.upper()}'))
        
        archivos_imagen = list(set([f for f in archivos_imagen if f.is_file()]))
        
    except Exception as e:
        reportar(f"Error escaneando imágenes: {e}")
        if gui_queue: 
            gui_queue.put("DONE")
        return

    # Filtrado para reanudación
    if estado and reanudar and len(archivos_imagen) > 0:
        archivos_pendientes = []
        
        for archivo in archivos_imagen:
            try:
                ruta_relativa = archivo.relative_to(path_entrada)
                ruta_salida_archivo = Path(ruta_salida_carpeta) / ruta_relativa
                
                if not estado.ya_procesado(str(ruta_relativa)):
                    if not ruta_salida_archivo.exists():
                        archivos_pendientes.append(archivo)
                    else:
                        try:
                            img_test = cv2.imread(str(ruta_salida_archivo))
                            if img_test is None:
                                archivos_pendientes.append(archivo)
                            else:
                                estado.agregar_procesada(str(ruta_relativa))
                        except:
                            archivos_pendientes.append(archivo)
                            
            except Exception as e:
                archivos_pendientes.append(archivo)
        
        ya_procesadas = len(archivos_imagen) - len(archivos_pendientes)
        reportar(f"Reanudando: {ya_procesadas} ya completadas, {len(archivos_pendientes)} pendientes")
        archivos_imagen = archivos_pendientes

    total_imagenes = len(archivos_imagen)
    if total_imagenes == 0:
        reportar("No hay imágenes nuevas para procesar.")
        if gui_queue: 
            gui_queue.put("DONE")
        return

    reportar(f"Procesando {total_imagenes} imágenes con {NUM_PROCESOS} threads en lotes de {BATCH_SIZE}")

    # Preparar datos de procesamiento
    datos_procesamiento = []
    
    for archivo in archivos_imagen:
        try:
            ruta_relativa = archivo.relative_to(path_entrada)
            ruta_salida_archivo = Path(ruta_salida_carpeta) / ruta_relativa
            os.makedirs(ruta_salida_archivo.parent, exist_ok=True)
            datos_procesamiento.append((
                str(archivo),
                str(ruta_salida_archivo),
                str(ruta_relativa)
            ))
        except Exception as e:
            reportar(f"Error preparando {archivo.name}: {e}")
    
    # Contadores
    imagenes_procesadas = 0
    errores = 0
    cancelado = False
    
    try:
        # CAMBIO PRINCIPAL: ThreadPoolExecutor en lugar de ProcessPoolExecutor
        with ThreadPoolExecutor(max_workers=NUM_PROCESOS) as executor:
            for lote_inicio in range(0, total_imagenes, BATCH_SIZE):
                if verificar_cancelacion(cancel_signal_path):
                    reportar("Cancelación detectada entre lotes.")
                    cancelado = True
                    break

                MonitorRecursos.esperar_recursos()
                
                lote_fin = min(lote_inicio + BATCH_SIZE, total_imagenes)
                lote = datos_procesamiento[lote_inicio:lote_fin]
                lote_num = (lote_inicio // BATCH_SIZE) + 1
                total_lotes = (total_imagenes + BATCH_SIZE - 1) // BATCH_SIZE
                
                reportar(f"Lote {lote_num}/{total_lotes} - Procesando imágenes {lote_inicio+1} a {lote_fin}")
                tiempo_lote_inicio = time.time()
                
                # Submit trabajos al ThreadPool
                futures = []
                for datos in lote:
                    future = executor.submit(procesar_imagen_thread_safe, datos, cancel_signal_path)
                    futures.append(future)
                
                # Recopilar resultados
                lote_exitosas = 0
                lote_errores = 0
                
                for future in futures:
                    try:
                        resultado, mensaje = future.result()
                        
                        if resultado:
                            imagenes_procesadas += 1
                            lote_exitosas += 1
                            if estado:
                                estado.agregar_procesada(resultado)
                        elif "Error" in mensaje and "Cancelado" not in mensaje:
                            errores += 1
                            lote_errores += 1
                            if estado:
                                estado.agregar_error("unknown", mensaje)
                                
                    except Exception as e:
                        errores += 1
                        lote_errores += 1

                # Estadísticas del lote
                tiempo_lote = time.time() - tiempo_lote_inicio
                velocidad = len(lote) / tiempo_lote if tiempo_lote > 0 else 0
                reportar(f"Lote {lote_num} completado en {tiempo_lote:.1f}s ({velocidad:.1f} img/s)")

                # Guardar progreso
                if estado:
                    estado.guardar_estado()

        if cancelado:
            marcar_cancelado(cancel_signal_path)
    
    except Exception as e:
        reportar(f"Error crítico en el procesamiento: {e}")
        marcar_cancelado(cancel_signal_path)
    
    finally:
        # Resumen final
        reportar(f"Imágenes procesadas exitosamente: {imagenes_procesadas}")
        if errores > 0:
            reportar(f"Errores encontrados: {errores}")
        
        if estado:
            estado.guardar_estado()
        
        if cancelado:
            reportar("Procesamiento cancelado por el usuario")
        else:
            reportar("Procesamiento completado exitosamente")

        if gui_queue:
            gui_queue.put("DONE")