"""
Interfaz gráfica para upme_autocarga_corregido.py

Reemplaza todos los inputs de consola (archivo Excel, pestaña, radicado,
ítem inicial, carpeta de fichas técnicas y credenciales de Bizagi) por
campos de una ventana, y muestra el progreso de la automatización en vivo.
"""
import contextlib
import os
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

import actualizador
import config_usuario
import upme_autocarga_corregido as core
import procesar_formato3_p as f3proc

BASE_DIR = core.BASE_DIR


class _QueueWriter:
    """File-like que empuja cada línea escrita a un queue.Queue.

    Usado para capturar los print() de procesar_formato3_p vía
    contextlib.redirect_stdout/redirect_stderr, igual que core.run_automation
    recibe su callback `log`.
    """

    def __init__(self, log_queue):
        self._queue = log_queue
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._queue.put(line)

    def flush(self):
        pass


class UpmeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"UPME - Herramientas de Automatización  v{actualizador.leer_version_local().get('version')}")
        self.geometry("760x680")
        self.minsize(700, 560)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.autocarga_tab = ttk.Frame(notebook)
        self.formato3_tab = ttk.Frame(notebook)
        notebook.add(self.autocarga_tab, text="Autocarga a Bizagi")
        notebook.add(self.formato3_tab, text="Procesar Formato 3 (PEPC + BOM)")

        self.excel_path_var = tk.StringVar()
        self.sheet_var = tk.StringVar()
        self.radicado_var = tk.StringVar(value=core.DEFAULT_RADICADO)
        self.start_idx_var = tk.StringVar(value="1")
        self.pdf_dir_var = tk.StringVar(value=core.PDF_DIR)
        self.maestro_var = tk.StringVar(value=core.MAESTRO_DEFAULT)
        self.usuario_var = tk.StringVar(value=core.USUARIO)
        self.password_var = tk.StringVar(value=core.PASSWORD)
        self.usar_costo_obj_var = tk.BooleanVar(value=False)
        self.costo_obj_var = tk.StringVar()

        self.log_queue = queue.Queue()
        self.worker_thread = None

        self._build_form()
        self._build_log_area()
        self.after(150, self._drain_log_queue)

        # ---- Estado de la pestaña "Procesar Formato 3" (independiente) ---- #
        self.f3_papa_var = tk.StringVar()
        self.f3_pepc_var = tk.StringVar()
        self.f3_bom_var = tk.StringVar()
        self.f3_salida_var = tk.StringVar()
        self.f3_completar_pepc_var = tk.BooleanVar(value=True)
        self.f3_pepc_ref_var = tk.StringVar()
        self.f3_completar_bom_var = tk.BooleanVar(value=True)
        self.f3_bom_ref_var = tk.StringVar()
        self.f3_precios_aux_var = tk.StringVar()
        self.f3_umbral_var = tk.StringVar(value="50.0")

        self.f3_log_queue = queue.Queue()
        self.f3_worker_thread = None

        self._build_formato3_tab()
        self._build_formato3_log_area()
        self.after(150, self._drain_f3_log_queue)

    # ---------------- UI ---------------- #
    def _build_form(self):
        pad = {"padx": 8, "pady": 5}
        frame = ttk.Frame(self.autocarga_tab)
        frame.pack(fill="x", padx=10, pady=10)
        frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(frame, text="Archivo Excel (.xlsx):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.excel_path_var).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Examinar...", command=self.browse_excel).grid(row=row, column=2, **pad)

        row += 1
        ttk.Label(frame, text="Pestaña (hoja):").grid(row=row, column=0, sticky="w", **pad)
        self.sheet_combo = ttk.Combobox(frame, textvariable=self.sheet_var, state="readonly")
        self.sheet_combo.grid(row=row, column=1, sticky="ew", **pad)

        row += 1
        ttk.Label(frame, text="Radicado:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.radicado_var).grid(row=row, column=1, sticky="ew", **pad)

        row += 1
        ttk.Label(frame, text="Empezar desde el ítem #:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Spinbox(frame, from_=1, to=99999, textvariable=self.start_idx_var, width=10).grid(
            row=row, column=1, sticky="w", **pad
        )

        row += 1
        ttk.Label(frame, text="Carpeta de fichas técnicas (PDF):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.pdf_dir_var).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Examinar...", command=self.browse_pdf_dir).grid(row=row, column=2, **pad)

        row += 1
        ttk.Label(frame, text="Maestro Odoo ID (El_Papá.xlsx):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.maestro_var).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Examinar...", command=self.browse_maestro).grid(row=row, column=2, **pad)

        row += 1
        ttk.Checkbutton(
            frame, text="Usar costo objetivo",
            variable=self.usar_costo_obj_var, command=self._toggle_costo_obj,
        ).grid(row=row, column=0, sticky="w", **pad)
        self.costo_obj_entry = ttk.Entry(frame, textvariable=self.costo_obj_var)
        self.costo_obj_entry.grid(row=row, column=1, sticky="ew", **pad)
        ttk.Label(frame, text="COP sin IVA").grid(row=row, column=2, sticky="w", **pad)
        self._toggle_costo_obj()

        row += 1
        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)

        row += 1
        ttk.Label(frame, text="Usuario Bizagi:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.usuario_var).grid(row=row, column=1, sticky="ew", **pad)

        row += 1
        ttk.Label(frame, text="Contraseña Bizagi:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.password_var, show="*").grid(row=row, column=1, sticky="ew", **pad)

        row += 1
        cred_frame = ttk.Frame(frame)
        cred_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=8)
        ttk.Button(cred_frame, text="Importar credenciales...", command=self.importar_credenciales).pack(
            side="left", padx=(0, 5))
        ttk.Button(cred_frame, text="Guardar en este PC", command=self.guardar_credenciales).pack(
            side="left", padx=5)
        ttk.Button(cred_frame, text="Olvidar", command=self.olvidar_credenciales).pack(side="left", padx=5)

        row += 1
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=(12, 0))
        self.start_btn = ttk.Button(btn_frame, text="Iniciar carga", command=self.start_automation)
        self.start_btn.pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Limpiar log", command=self.clear_log).pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="Listo.")
        ttk.Label(self.autocarga_tab, textvariable=self.status_var, anchor="w").pack(fill="x", padx=12)

    def _build_log_area(self):
        frame = ttk.LabelFrame(self.autocarga_tab, text="Progreso")
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.log_text = tk.Text(text_frame, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ---------------- Acciones ---------------- #
    def browse_excel(self):
        path = filedialog.askopenfilename(
            title="Selecciona el archivo Excel",
            filetypes=[("Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        self.excel_path_var.set(path)
        self._load_sheets(path)

    def _load_sheets(self, path):
        try:
            xl = pd.ExcelFile(path)
            sheets = xl.sheet_names
        except Exception as e:
            messagebox.showerror("Error leyendo Excel", f"No se pudo leer el archivo:\n{e}")
            return

        self.sheet_combo["values"] = sheets
        matching = [sh for sh in sheets if "formato 3" in sh.lower()]
        self.sheet_var.set(matching[0] if matching else sheets[0])

    def browse_pdf_dir(self):
        path = filedialog.askdirectory(title="Selecciona la carpeta de fichas técnicas (PDF)")
        if path:
            self.pdf_dir_var.set(path)

    def browse_maestro(self):
        path = filedialog.askopenfilename(
            title="Selecciona el Excel maestro de Odoo ID",
            filetypes=[("Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.maestro_var.set(path)

    def importar_credenciales(self):
        path = filedialog.askopenfilename(
            title="Selecciona el archivo de credenciales (.env)",
            filetypes=[("Credenciales", "*.env"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return
        try:
            usuario, password = config_usuario.importar_credenciales(path)
        except Exception as e:
            messagebox.showerror("Credenciales", f"No se pudo importar el archivo:\n{e}")
            return
        self.usuario_var.set(usuario)
        self.password_var.set(password)
        messagebox.showinfo("Credenciales", "Credenciales importadas y guardadas en este PC.")

    def guardar_credenciales(self):
        usuario = self.usuario_var.get().strip()
        password = self.password_var.get()
        if not usuario or not password:
            messagebox.showwarning("Credenciales", "Ingresa usuario y contraseña antes de guardar.")
            return
        config_usuario.guardar_credenciales(usuario, password)
        messagebox.showinfo("Credenciales", f"Credenciales guardadas en:\n{config_usuario.ENV_PATH}")

    def olvidar_credenciales(self):
        if not messagebox.askyesno("Credenciales", "¿Borrar las credenciales guardadas en este PC?"):
            return
        config_usuario.borrar_credenciales()
        self.usuario_var.set("")
        self.password_var.set("")

    def _toggle_costo_obj(self):
        self.costo_obj_entry.configure(state="normal" if self.usar_costo_obj_var.get() else "disabled")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _log(self, message):
        # Llamado desde el hilo de automatización: solo encola, no toca la UI directamente.
        self.log_queue.put(str(message))

    def _drain_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(150, self._drain_log_queue)

    def _validate_inputs(self):
        excel_path = self.excel_path_var.get().strip()
        if not excel_path or not os.path.isfile(excel_path):
            messagebox.showwarning("Falta información", "Selecciona un archivo Excel válido.")
            return None
        sheet = self.sheet_var.get().strip()
        if not sheet:
            messagebox.showwarning("Falta información", "Selecciona la pestaña del Excel.")
            return None
        radicado = self.radicado_var.get().strip() or core.DEFAULT_RADICADO
        try:
            start_idx = int(self.start_idx_var.get())
            if start_idx < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Falta información", "El ítem inicial debe ser un número entero >= 1.")
            return None
        pdf_dir = self.pdf_dir_var.get().strip()
        maestro_path = self.maestro_var.get().strip()
        if not maestro_path or not os.path.isfile(maestro_path):
            messagebox.showwarning("Falta información", "Selecciona el Excel maestro de Odoo ID (El_Papá.xlsx).")
            return None
        usuario = self.usuario_var.get().strip()
        password = self.password_var.get()
        if not usuario or not password:
            messagebox.showwarning("Falta información", "Ingresa el usuario y la contraseña de Bizagi.")
            return None
        costo_objetivo = None
        if self.usar_costo_obj_var.get():
            costo_objetivo = core.parse_costo_objetivo(self.costo_obj_var.get())
            if costo_objetivo is None:
                messagebox.showwarning("Falta información",
                                       "Ingresa un costo objetivo válido (valor total en COP sin IVA, mayor a 0).")
                return None

        return excel_path, sheet, radicado, start_idx, pdf_dir, usuario, password, maestro_path, costo_objetivo

    def start_automation(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("En progreso", "Ya hay una carga en ejecución.")
            return

        values = self._validate_inputs()
        if values is None:
            return
        config_usuario.guardar_config(pdf_dir=values[4], maestro=values[7])
        self.clear_log()
        self.start_btn.configure(state="disabled")
        self.status_var.set("Ejecutando automatización... revisa el navegador que se abrirá.")

        self.worker_thread = threading.Thread(
            target=self._run_worker,
            args=values,
            daemon=True,
        )
        self.worker_thread.start()

    def _run_worker(self, excel_path, sheet, radicado, start_idx, pdf_dir, usuario, password, maestro_path,
                    costo_objetivo):
        try:
            core.run_automation(
                excel_path=excel_path,
                sheet_name=sheet,
                radicado=radicado,
                start_idx=start_idx,
                pdf_dir=pdf_dir,
                usuario=usuario,
                password=password,
                log=self._log,
                keep_browser_open=True,
                maestro_path=maestro_path,
                costo_objetivo=costo_objetivo,
            )
            self._log("=== PROCESO FINALIZADO CORRECTAMENTE ===")
            self._finish(True)
        except Exception as e:
            self._log("=== ERROR DURANTE LA AUTOMATIZACIÓN ===")
            self._log(traceback.format_exc())
            self._finish(False, str(e))

    def _finish(self, ok, error_msg=None):
        def update_ui():
            self.start_btn.configure(state="normal")
            if ok:
                self.status_var.set("Carga finalizada. El navegador queda abierto para revisar.")
            else:
                self.status_var.set("Ocurrió un error. Revisa el log. El navegador queda abierto.")
                messagebox.showerror("Error", f"La automatización falló:\n{error_msg}")

        self.after(0, update_ui)

    # ============= Pestaña "Procesar Formato 3 (PEPC + BOM)" ============= #
    def _build_formato3_tab(self):
        pad = {"padx": 8, "pady": 5}
        frame = ttk.Frame(self.formato3_tab)
        frame.pack(fill="x", padx=10, pady=10)
        frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(frame, text='Excel Formato 3 / "Papá" objetivo:').grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.f3_papa_var).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Examinar...", command=self.browse_f3_papa).grid(row=row, column=2, **pad)

        row += 1
        ttk.Label(frame, text="PEPC del proyecto:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.f3_pepc_var).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Examinar...", command=self.browse_f3_pepc).grid(row=row, column=2, **pad)

        row += 1
        ttk.Label(frame, text="BOM del proyecto:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.f3_bom_var).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Examinar...", command=self.browse_f3_bom).grid(row=row, column=2, **pad)

        row += 1
        ttk.Label(frame, text="Ruta de salida:").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.f3_salida_var).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Guardar como...", command=self.browse_f3_salida).grid(row=row, column=2, **pad)

        row += 1
        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)

        row += 1
        ttk.Checkbutton(
            frame, text="Completar Código Odoo en PEPC",
            variable=self.f3_completar_pepc_var, command=self._toggle_f3_pepc_ref,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)

        row += 1
        ttk.Label(frame, text="PEPC de referencia (opcional):").grid(row=row, column=0, sticky="w", **pad)
        self.f3_pepc_ref_entry = ttk.Entry(frame, textvariable=self.f3_pepc_ref_var)
        self.f3_pepc_ref_entry.grid(row=row, column=1, sticky="ew", **pad)
        self.f3_pepc_ref_btn = ttk.Button(frame, text="Examinar...", command=self.browse_f3_pepc_ref)
        self.f3_pepc_ref_btn.grid(row=row, column=2, **pad)

        row += 1
        ttk.Checkbutton(
            frame, text="Completar columna PROVEEDOR en BOM",
            variable=self.f3_completar_bom_var, command=self._toggle_f3_bom_ref,
        ).grid(row=row, column=0, columnspan=2, sticky="w", **pad)

        row += 1
        ttk.Label(frame, text="BOM de referencia (opcional):").grid(row=row, column=0, sticky="w", **pad)
        self.f3_bom_ref_entry = ttk.Entry(frame, textvariable=self.f3_bom_ref_var)
        self.f3_bom_ref_entry.grid(row=row, column=1, sticky="ew", **pad)
        self.f3_bom_ref_btn = ttk.Button(frame, text="Examinar...", command=self.browse_f3_bom_ref)
        self.f3_bom_ref_btn.grid(row=row, column=2, **pad)

        row += 1
        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)

        row += 1
        ttk.Label(frame, text="Excel de precios auxiliares (opcional):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.f3_precios_aux_var).grid(row=row, column=1, sticky="ew", **pad)
        btns_precios = ttk.Frame(frame)
        btns_precios.grid(row=row, column=2, sticky="w")
        ttk.Button(btns_precios, text="Examinar...", command=self.browse_f3_precios_aux).pack(side="left")
        ttk.Button(btns_precios, text="Limpiar", command=self.clear_f3_precios_aux).pack(side="left", padx=(4, 0))

        row += 1
        ttk.Label(frame, text="Umbral de similitud (0-100):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Spinbox(
            frame, from_=0, to=100, increment=1, textvariable=self.f3_umbral_var, width=10,
        ).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=(12, 0))
        self.f3_start_btn = ttk.Button(btn_frame, text="Procesar Formato 3", command=self.start_f3_processing)
        self.f3_start_btn.pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Limpiar log", command=self.clear_f3_log).pack(side="left", padx=5)

        self.f3_status_var = tk.StringVar(value="Listo.")
        ttk.Label(self.formato3_tab, textvariable=self.f3_status_var, anchor="w").pack(fill="x", padx=12)

        self._toggle_f3_pepc_ref()
        self._toggle_f3_bom_ref()

    def _build_formato3_log_area(self):
        frame = ttk.LabelFrame(self.formato3_tab, text="Progreso")
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True, padx=6, pady=6)

        self.f3_log_text = tk.Text(text_frame, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.f3_log_text.yview)
        self.f3_log_text.configure(yscrollcommand=scrollbar.set)
        self.f3_log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _toggle_f3_pepc_ref(self):
        state = "normal" if self.f3_completar_pepc_var.get() else "disabled"
        self.f3_pepc_ref_entry.configure(state=state)
        self.f3_pepc_ref_btn.configure(state=state)

    def _toggle_f3_bom_ref(self):
        state = "normal" if self.f3_completar_bom_var.get() else "disabled"
        self.f3_bom_ref_entry.configure(state=state)
        self.f3_bom_ref_btn.configure(state=state)

    # ---------------- Acciones (Formato 3) ---------------- #
    def browse_f3_papa(self):
        path = filedialog.askopenfilename(
            title='Selecciona el Excel Formato 3 / "Papá" objetivo',
            filetypes=[("Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.f3_papa_var.set(path)

    def browse_f3_pepc(self):
        path = filedialog.askopenfilename(
            title="Selecciona el PEPC del proyecto",
            filetypes=[("Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.f3_pepc_var.set(path)

    def browse_f3_bom(self):
        path = filedialog.askopenfilename(
            title="Selecciona el BOM del proyecto",
            filetypes=[("Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.f3_bom_var.set(path)

    def browse_f3_salida(self):
        path = filedialog.asksaveasfilename(
            title="Guardar Formato 3 procesado como",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.f3_salida_var.set(path)

    def browse_f3_pepc_ref(self):
        path = filedialog.askopenfilename(
            title="Selecciona el PEPC de referencia",
            filetypes=[("Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.f3_pepc_ref_var.set(path)

    def browse_f3_bom_ref(self):
        path = filedialog.askopenfilename(
            title="Selecciona el BOM de referencia",
            filetypes=[("Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.f3_bom_ref_var.set(path)

    def browse_f3_precios_aux(self):
        path = filedialog.askopenfilename(
            title="Selecciona el Excel de precios auxiliares",
            filetypes=[("Excel", "*.xlsx"), ("Todos los archivos", "*.*")],
        )
        if path:
            self.f3_precios_aux_var.set(path)

    def clear_f3_precios_aux(self):
        self.f3_precios_aux_var.set("")

    def clear_f3_log(self):
        self.f3_log_text.configure(state="normal")
        self.f3_log_text.delete("1.0", "end")
        self.f3_log_text.configure(state="disabled")

    def _f3_log(self, message):
        # Llamado desde el hilo de procesamiento: solo encola, no toca la UI directamente.
        self.f3_log_queue.put(str(message))

    def _drain_f3_log_queue(self):
        try:
            while True:
                msg = self.f3_log_queue.get_nowait()
                self.f3_log_text.configure(state="normal")
                self.f3_log_text.insert("end", msg + "\n")
                self.f3_log_text.see("end")
                self.f3_log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(150, self._drain_f3_log_queue)

    def _validate_f3_inputs(self):
        path_papa = self.f3_papa_var.get().strip()
        if not path_papa or not os.path.isfile(path_papa):
            messagebox.showwarning("Falta información", 'Selecciona un Excel Formato 3 / "Papá" objetivo válido.')
            return None
        path_pepc = self.f3_pepc_var.get().strip()
        if not path_pepc or not os.path.isfile(path_pepc):
            messagebox.showwarning("Falta información", "Selecciona un PEPC del proyecto válido.")
            return None
        path_bom = self.f3_bom_var.get().strip()
        if not path_bom or not os.path.isfile(path_bom):
            messagebox.showwarning("Falta información", "Selecciona un BOM del proyecto válido.")
            return None
        path_salida = self.f3_salida_var.get().strip()
        if not path_salida:
            messagebox.showwarning("Falta información", "Indica la ruta de salida.")
            return None

        path_pepc_referencia = self.f3_pepc_ref_var.get().strip() or None
        path_bom_referencia = self.f3_bom_ref_var.get().strip() or None
        path_precios_aux = self.f3_precios_aux_var.get().strip() or None

        try:
            umbral_similitud = float(self.f3_umbral_var.get())
        except ValueError:
            messagebox.showwarning("Falta información", "El umbral de similitud debe ser un número entre 0 y 100.")
            return None

        return {
            "path_papa": path_papa,
            "path_pepc": path_pepc,
            "path_bom": path_bom,
            "path_salida": path_salida,
            "path_precios_aux": path_precios_aux,
            "completar_pepc": self.f3_completar_pepc_var.get(),
            "path_pepc_referencia": path_pepc_referencia,
            "completar_bom": self.f3_completar_bom_var.get(),
            "path_bom_referencia": path_bom_referencia,
            "umbral_similitud": umbral_similitud,
        }

    def start_f3_processing(self):
        if self.f3_worker_thread and self.f3_worker_thread.is_alive():
            messagebox.showinfo("En progreso", "Ya hay un procesamiento de Formato 3 en ejecución.")
            return

        kwargs = self._validate_f3_inputs()
        if kwargs is None:
            return

        self.clear_f3_log()
        self.f3_start_btn.configure(state="disabled")
        self.f3_status_var.set("Procesando Formato 3...")

        self.f3_worker_thread = threading.Thread(
            target=self._run_f3_worker,
            kwargs=kwargs,
            daemon=True,
        )
        self.f3_worker_thread.start()

    def _run_f3_worker(self, **kwargs):
        writer = _QueueWriter(self.f3_log_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                f3proc.run_procesamiento(**kwargs)
            self._f3_log("=== PROCESO FINALIZADO CORRECTAMENTE ===")
            self._f3_finish(True)
        except Exception as e:
            self._f3_log("=== ERROR DURANTE EL PROCESAMIENTO ===")
            self._f3_log(traceback.format_exc())
            self._f3_finish(False, str(e))

    def _f3_finish(self, ok, error_msg=None):
        def update_ui():
            self.f3_start_btn.configure(state="normal")
            if ok:
                self.f3_status_var.set("Procesamiento finalizado correctamente.")
            else:
                self.f3_status_var.set("Ocurrió un error. Revisa el log.")
                messagebox.showerror("Error", f"El procesamiento de Formato 3 falló:\n{error_msg}")

        self.after(0, update_ui)


def main():
    app = UpmeGUI()
    cfg = config_usuario.leer_config()
    if cfg.get("pdf_dir") and os.path.isdir(cfg["pdf_dir"]):
        app.pdf_dir_var.set(cfg["pdf_dir"])
    elif os.path.isdir(core.PDF_DIR):
        app.pdf_dir_var.set(core.PDF_DIR)
    if cfg.get("maestro") and os.path.isfile(cfg["maestro"]):
        app.maestro_var.set(cfg["maestro"])
    app.mainloop()


if __name__ == "__main__":
    main()
