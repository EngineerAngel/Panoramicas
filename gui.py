import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import multiprocessing
import queue
import os

# Importamos la función de alto rendimiento
from procesamiento_lotes import procesar_carpeta_alto_volumen

# --- CONFIGURACIÓN INICIAL (Opcional) ---
# Puedes definir aquí carpetas por defecto si lo deseas
CARPETA_BASE_DEFAULT = ""
CARPETA_SALIDA_DEFAULT = ""

class PanoramicasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes de Alto Rendimiento")
        self.root.geometry("700x550")
        
        # --- Variables de Tkinter ---
        self.carpeta_entrada = tk.StringVar(value=CARPETA_BASE_DEFAULT)
        self.carpeta_salida = tk.StringVar(value=CARPETA_SALIDA_DEFAULT)
        
        # --- Control del Hilo y Cancelación ---
        self.thread = None
        # Usaremos un archivo como señal de cancelación para los subprocesos
        self.cancel_signal_path = os.path.join(os.path.expanduser("~"), ".cancel_processing_signal")

        # --- Cola de comunicación para la GUI ---
        # Usamos una cola de `multiprocessing` porque el motor crea procesos.
        self.manager = multiprocessing.Manager()
        self.gui_queue = self.manager.Queue()

        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Sección de Carpetas ---
        tk.Label(main_frame, text="Carpeta de entrada:", anchor="w").pack(fill=tk.X)
        entry_in = tk.Entry(main_frame, textvariable=self.carpeta_entrada, width=70)
        entry_in.pack(fill=tk.X, expand=True)
        tk.Button(main_frame, text="Seleccionar...", command=self.seleccionar_entrada).pack(anchor="e", pady=5)

        tk.Label(main_frame, text="Carpeta de salida:", anchor="w").pack(fill=tk.X)
        entry_out = tk.Entry(main_frame, textvariable=self.carpeta_salida, width=70)
        entry_out.pack(fill=tk.X, expand=True)
        tk.Button(main_frame, text="Seleccionar...", command=self.seleccionar_salida).pack(anchor="e", pady=5)

        # --- Sección de Botones ---
        frame_btns = tk.Frame(main_frame)
        frame_btns.pack(pady=10)
        self.btn_iniciar = tk.Button(frame_btns, text="Iniciar procesamiento", command=self.iniciar_procesamiento, bg="#4CAF50", fg="white", font=('Helvetica', 10, 'bold'))
        self.btn_iniciar.pack(side=tk.LEFT, padx=10)
        self.btn_cancelar = tk.Button(frame_btns, text="Cancelar", command=self.cancelar_procesamiento, bg="#f44336", fg="white", font=('Helvetica', 10, 'bold'), state=tk.DISABLED)
        self.btn_cancelar.pack(side=tk.LEFT, padx=10)

        # --- Sección de Estado/Log ---
        self.text_status = scrolledtext.ScrolledText(main_frame, height=15, wrap=tk.WORD, font=("Consolas", 9))
        self.text_status.pack(fill=tk.BOTH, expand=True, pady=5)
        self.text_status.config(state=tk.DISABLED)

    def seleccionar_entrada(self):
        carpeta = filedialog.askdirectory(title="Selecciona la carpeta de entrada")
        if carpeta:
            self.carpeta_entrada.set(carpeta)
            # Sugerir una carpeta de salida basada en la de entrada
            if not self.carpeta_salida.get():
                self.carpeta_salida.set(os.path.join(carpeta, "procesadas"))

    def seleccionar_salida(self):
        carpeta = filedialog.askdirectory(title="Selecciona la carpeta de salida")
        if carpeta:
            self.carpeta_salida.set(carpeta)

    def iniciar_procesamiento(self):
        entrada = self.carpeta_entrada.get()
        salida = self.carpeta_salida.get()

        if not os.path.isdir(entrada) or not salida:
            messagebox.showerror("Error", "Por favor, selecciona una carpeta de entrada válida y una de salida.")
            return

        os.makedirs(salida, exist_ok=True)
        
        # Limpiar la señal de cancelación de una ejecución anterior
        if os.path.exists(self.cancel_signal_path):
            os.remove(self.cancel_signal_path)

        # Actualizar la GUI
        self.log_message("🚀 Iniciando procesamiento... Por favor, espera.")
        self.btn_iniciar.config(state=tk.DISABLED)
        self.btn_cancelar.config(state=tk.NORMAL)
        
        # Lanzamos el motor de procesamiento en su propio hilo de control
        self.thread = threading.Thread(
            target=procesar_carpeta_alto_volumen,
            args=(entrada, salida, self.gui_queue, self.cancel_signal_path),
            kwargs={'reanudar': True},
            daemon=True
        )
        self.thread.start()
        
        # Empezamos a escuchar los mensajes del motor
        self.revisar_cola()

    def cancelar_procesamiento(self):
        self.log_message("\n🛑 Enviando señal de cancelación... El proceso se detendrá después del lote actual.")
        # Creamos el archivo de señal
        with open(self.cancel_signal_path, 'w') as f:
            f.write('cancel')
        self.btn_cancelar.config(state=tk.DISABLED)

    def revisar_cola(self):
        """Revisa la cola en busca de mensajes del hilo de procesamiento."""
        try:
            mensaje = self.gui_queue.get_nowait()
            if mensaje == "DONE":
                self.log_message("\n✨ Proceso finalizado.")
                self.finalizar_procesamiento()
                return # Detenemos el bucle
            
            self.log_message(mensaje)
        except queue.Empty:
            pass
        finally:
            # Volvemos a llamar a esta función después de 100ms
            self.root.after(100, self.revisar_cola)
    
    def finalizar_procesamiento(self):
        """Limpia y restaura la GUI al estado inicial."""
        self.btn_iniciar.config(state=tk.NORMAL)
        self.btn_cancelar.config(state=tk.DISABLED)
        if os.path.exists(self.cancel_signal_path):
            os.remove(self.cancel_signal_path)
        messagebox.showinfo("Completado", "El procesamiento ha finalizado.")

    def log_message(self, message):
        """Añade un mensaje al cuadro de texto de estado."""
        self.text_status.config(state=tk.NORMAL)
        self.text_status.insert(tk.END, message + "\n")
        self.text_status.see(tk.END)
        self.text_status.config(state=tk.DISABLED)


if __name__ == "__main__":
    # Necesario para que `multiprocessing` funcione correctamente en ejecutables (PyInstaller)
    multiprocessing.freeze_support()
    
    root = tk.Tk()
    app = PanoramicasApp(root)
    root.mainloop()