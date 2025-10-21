import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import queue
import os
import time
from pathlib import Path

# Importamos la función de alto rendimiento
from procesamiento_lotes import procesar_carpeta_alto_volumen

# --- CONFIGURACIÓN INICIAL (Opcional) ---
CARPETA_BASE_DEFAULT = ""
CARPETA_SALIDA_DEFAULT = ""

class PanoramicasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes (Semic V1.0)")
        self.root.geometry("700x600")
        
        # --- Variables de Tkinter ---
        self.carpeta_entrada = tk.StringVar(value=CARPETA_BASE_DEFAULT)
        self.carpeta_salida = tk.StringVar(value=CARPETA_SALIDA_DEFAULT)
        
        # --- Control del Hilo y Cancelación ---
        self.thread = None
        # Usaremos un archivo como señal de cancelación para los subprocesos
        self.cancel_signal_path = os.path.join(os.path.expanduser("~"), ".cancel_processing_signal")
        
        # Flag adicional para control local
        self.procesamiento_activo = False

        # --- Cola de comunicación simple (SIN multiprocessing.Manager) ---
        self.gui_queue = queue.Queue()  # Queue normal de threading

        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Sección de Carpetas ---
        tk.Label(main_frame, text="Carpeta de entrada:", anchor="w", font=('Helvetica', 9, 'bold')).pack(fill=tk.X)
        entry_in = tk.Entry(main_frame, textvariable=self.carpeta_entrada, width=70, font=('Helvetica', 9))
        entry_in.pack(fill=tk.X, expand=True)
        tk.Button(main_frame, text="Seleccionar carpeta de entrada...", command=self.seleccionar_entrada).pack(anchor="e", pady=5)

        tk.Label(main_frame, text="Carpeta de salida:", anchor="w", font=('Helvetica', 9, 'bold')).pack(fill=tk.X)
        entry_out = tk.Entry(main_frame, textvariable=self.carpeta_salida, width=70, font=('Helvetica', 9))
        entry_out.pack(fill=tk.X, expand=True)
        
        buttons_salida_frame = tk.Frame(main_frame)
        buttons_salida_frame.pack(anchor="e", pady=5)
        tk.Button(buttons_salida_frame, text="Usar carpeta automática", command=self.generar_ruta_automatica).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_salida_frame, text="Seleccionar carpeta personalizada...", command=self.seleccionar_salida).pack(side=tk.LEFT)

        # --- Sección de Opciones ---
        options_frame = tk.Frame(main_frame)
        options_frame.pack(pady=10, fill=tk.X)
        
        self.reanudar_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Reanudar procesamiento anterior (si existe)", 
                      variable=self.reanudar_var, font=('Helvetica', 9)).pack(anchor="w")

        # --- Sección de Botones ---
        frame_btns = tk.Frame(main_frame)
        frame_btns.pack(pady=10)
        self.btn_iniciar = tk.Button(frame_btns, text="Iniciar procesamiento", command=self.iniciar_procesamiento, bg="#4CAF50", fg="white", font=('Helvetica', 10, 'bold'))
        self.btn_iniciar.pack(side=tk.LEFT, padx=10)
        self.btn_cancelar = tk.Button(frame_btns, text="Cancelar", command=self.cancelar_procesamiento, bg="#f44336", fg="white", font=('Helvetica', 10, 'bold'), state=tk.DISABLED)
        self.btn_cancelar.pack(side=tk.LEFT, padx=10)

        # --- Sección de Estado/Log ---
        self.text_status = scrolledtext.ScrolledText(main_frame, height=12, wrap=tk.WORD, font=("Consolas", 9))
        self.text_status.pack(fill=tk.BOTH, expand=True, pady=5)
        self.text_status.config(state=tk.DISABLED)

    def generar_ruta_automatica(self):
        """Genera la ruta de salida automática manteniendo las dos carpetas finales"""
        entrada = self.carpeta_entrada.get()
        if not entrada:
            messagebox.showwarning("Advertencia", "Primero selecciona una carpeta de entrada.")
            return
        
        try:
            path_entrada = Path(entrada)
            # Obtener las dos carpetas finales: carpeta_abuelo\carpeta_padre
            carpeta_padre = path_entrada.name  # ej: S2C2
            carpeta_abuelo = path_entrada.parent.name  # ej: v2255
            
            # Ir dos carpetas arriba y crear la estructura completa
            carpeta_base = path_entrada.parent.parent
            estructura_salida = f"{carpeta_abuelo}\\{carpeta_padre}_procesadas"
            carpeta_salida = carpeta_base / estructura_salida
            
            self.carpeta_salida.set(str(carpeta_salida))
        except Exception as e:
            messagebox.showerror("Error", f"Error generando ruta automática: {e}")

    def seleccionar_entrada(self):
        carpeta = filedialog.askdirectory(title="Selecciona la carpeta de entrada (que contiene subcarpetas con imágenes)")
        if carpeta:
            self.carpeta_entrada.set(carpeta)
            # Generar automáticamente la ruta de salida
            self.generar_ruta_automatica()

    def seleccionar_salida(self):
        carpeta = filedialog.askdirectory(title="Selecciona dónde crear la estructura de carpetas procesadas")
        if carpeta:
            entrada = self.carpeta_entrada.get()
            if entrada:
                # Mantener la estructura de las dos carpetas finales
                path_entrada = Path(entrada)
                carpeta_padre = path_entrada.name  # ej: S2C2
                carpeta_abuelo = path_entrada.parent.name  # ej: v2255
                
                estructura_salida = f"{carpeta_abuelo}\\{carpeta_padre}_procesadas"
                carpeta_salida_final = Path(carpeta) / estructura_salida
                self.carpeta_salida.set(str(carpeta_salida_final))
            else:
                self.carpeta_salida.set(carpeta)

    def iniciar_procesamiento(self):
        entrada = self.carpeta_entrada.get()
        salida = self.carpeta_salida.get()

        if not os.path.isdir(entrada) or not salida:
            messagebox.showerror("Error", "Por favor, selecciona una carpeta de entrada válida y una de salida.")
            return

        # Verificar que la carpeta de entrada tenga subcarpetas
        try:
            path_entrada = Path(entrada)
            subcarpetas = [d for d in path_entrada.iterdir() if d.is_dir()]
            if not subcarpetas:
                respuesta = messagebox.askyesno("Advertencia", 
                    "La carpeta de entrada no parece tener subcarpetas.\n¿Continuar de todas formas?")
                if not respuesta:
                    return
        except Exception as e:
            messagebox.showerror("Error", f"Error verificando estructura: {e}")
            return

        # Crear la carpeta de salida
        os.makedirs(salida, exist_ok=True)
        
        # Limpiar la señal de cancelación de una ejecución anterior
        if os.path.exists(self.cancel_signal_path):
            os.remove(self.cancel_signal_path)

        # Actualizar la GUI
        reanudar = self.reanudar_var.get()
        self.log_message("Iniciando procesamiento de imágenes...")
        self.log_message("Por favor, espera mientras se procesan las imágenes...")
        
        self.btn_iniciar.config(state=tk.DISABLED)
        self.btn_cancelar.config(state=tk.NORMAL)
        self.procesamiento_activo = True
        
        # Lanzamos el motor de procesamiento en su propio hilo de control
        self.thread = threading.Thread(
            target=procesar_carpeta_alto_volumen,
            args=(entrada, salida, self.gui_queue, self.cancel_signal_path),
            kwargs={'reanudar': reanudar},
            daemon=True
        )
        self.thread.start()
        
        # Empezamos a escuchar los mensajes del motor
        self.revisar_cola()

    def cancelar_procesamiento(self):
        if not self.procesamiento_activo:
            return
            
        self.log_message("\nCancelando procesamiento...")
        self.log_message("El proceso se detendrá después del lote actual.")
        
        # Creamos el archivo de señal
        try:
            with open(self.cancel_signal_path, 'w') as f:
                f.write(f'cancel_{time.time()}')
        except Exception as e:
            self.log_message(f"Error enviando señal de cancelación: {e}")
        
        self.btn_cancelar.config(state=tk.DISABLED)
        self.btn_cancelar.config(text="Cancelando...")

    def revisar_cola(self):
        """Revisa la cola en busca de mensajes del hilo de procesamiento."""
        try:
            # Procesamos múltiples mensajes si están disponibles
            mensajes_procesados = 0
            while mensajes_procesados < 10:  # Límite para evitar bloquear la GUI
                try:
                    mensaje = self.gui_queue.get_nowait()
                    if mensaje == "DONE":
                        self.log_message("\nProcesamiento finalizado.")
                        self.finalizar_procesamiento()
                        return # Detenemos el bucle
                    
                    self.log_message(mensaje)
                    mensajes_procesados += 1
                except queue.Empty:
                    break
        except Exception as e:
            self.log_message(f"Error procesando mensajes: {e}")
        finally:
            # Solo continuar el bucle si el procesamiento está activo
            if self.procesamiento_activo:
                self.root.after(100, self.revisar_cola)
    
    def finalizar_procesamiento(self):
        """Limpia y restaura la GUI al estado inicial."""
        self.procesamiento_activo = False
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_cancelar.config(state=tk.DISABLED)
        self.btn_cancelar.config(text="Cancelar")
        
        # Limpiar archivo de señal
        if os.path.exists(self.cancel_signal_path):
            try:
                os.remove(self.cancel_signal_path)
            except:
                pass
                
        # Determinar tipo de finalización
        if os.path.exists(self.cancel_signal_path + ".cancelled"):
            self.log_message("El procesamiento fue cancelado por el usuario.")
            messagebox.showinfo("Cancelado", "El procesamiento fue cancelado por el usuario.")
            try:
                os.remove(self.cancel_signal_path + ".cancelled")
            except:
                pass
        else:
            self.log_message("El procesamiento ha finalizado exitosamente!")
            messagebox.showinfo("Completado", "El procesamiento ha finalizado exitosamente!")

    def log_message(self, message):
        """Añade un mensaje al cuadro de texto de estado."""
        self.text_status.config(state=tk.NORMAL)
        self.text_status.insert(tk.END, message + "\n")
        self.text_status.see(tk.END)
        self.text_status.config(state=tk.DISABLED)
        # Forzar actualización de la GUI
        self.root.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = PanoramicasApp(root)
    root.mainloop()