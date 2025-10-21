# config.py
# Configuración automática inteligente que se adapta a cada PC
import multiprocessing
import psutil
import platform
import os

# Rutas por defecto
carpeta_base = r"C:\Users\ANGEL GOMEZ\OneDrive\Documentos\BC-163-02\S2C2"
carpeta_salida_base = r"C:\Users\ANGEL GOMEZ\Proyectos\Proyecto_semic\Correccion_final"

def detectar_tipo_sistema():
    """Detecta si es laptop, desktop o servidor"""
    try:
        # Método 1: Verificar batería (indica laptop)
        if psutil.sensors_battery() is not None:
            return "laptop"
        
        # Método 2: Verificar ventiladores (muchos = desktop/servidor)
        try:
            fans = psutil.sensors_fans()
            if fans and len(fans) > 2:
                return "desktop"
        except:
            pass
        
        # Método 3: Por defecto asumir desktop
        return "desktop"
    except:
        return "desktop"

def detectar_configuracion_optima():
    """Detecta automáticamente la configuración óptima para el sistema"""
    
    # Información del sistema
    cpu_count = multiprocessing.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    tipo_sistema = detectar_tipo_sistema()
    
    # Configuración base según recursos
    if ram_gb < 4:  # Sistema muy básico
        base_procesos = max(1, cpu_count // 4)
        batch_size = 10
        memoria_limite = 60
        
    elif ram_gb < 8:  # Sistema estándar
        base_procesos = max(2, cpu_count // 3)
        batch_size = 25
        memoria_limite = 70
        
    elif ram_gb < 16:  # Sistema bueno
        base_procesos = max(2, cpu_count // 2)
        batch_size = 50
        memoria_limite = 80
        
    else:  # Sistema potente (16GB+)
        base_procesos = max(4, min(8, cpu_count - 2))
        batch_size = 75
        memoria_limite = 85
    
    # Ajustes específicos por tipo de sistema
    if tipo_sistema == "laptop":
        # Laptops: más conservador (se calientan, batería limitada)
        procesos_final = max(1, base_procesos // 2)
        memoria_limite -= 10
    else:  # desktop/servidor
        # Desktop: puede ser más agresivo
        procesos_final = base_procesos
    
    # Límites de seguridad
    procesos_final = min(procesos_final, 8)  # Nunca más de 8 procesos
    memoria_limite = max(memoria_limite, 50)  # Mínimo 50% antes de pausar
    
    return procesos_final, batch_size, memoria_limite

# Ejecutar detección automática
NUM_PROCESOS, BATCH_SIZE, MEMORY_THRESHOLD_PERCENT = detectar_configuracion_optima()

# Configuración adicional
NUM_THREADS = max(2, NUM_PROCESOS // 2)  # Para operaciones I/O

# Configuración avanzada (opcional)
MODO_CONSERVADOR = False  # Cambiar a True para ser aún más conservador

if MODO_CONSERVADOR:
    NUM_PROCESOS = max(1, NUM_PROCESOS // 2)
    BATCH_SIZE = max(10, BATCH_SIZE // 2)
    MEMORY_THRESHOLD_PERCENT = max(50, MEMORY_THRESHOLD_PERCENT - 15)