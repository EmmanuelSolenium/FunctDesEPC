import os
import sys
import re
import math
import json
import difflib
import unicodedata
import pandas as pd
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# ---------------- CONFIGURACIÓN ---------------- #
# Cuando se empaqueta con PyInstaller (--onefile), __file__ apunta a la carpeta
# temporal de extracción (_MEIPASS), no a la carpeta donde está el .exe. En ese
# caso usamos la carpeta del ejecutable para poder leer .env y Fichastecnicas
# ubicados junto al .exe.
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Primero las credenciales del usuario (%APPDATA%\UPME AutoCarga\.env, ver
# config_usuario.py); un .env junto al programa queda como respaldo.
import config_usuario
load_dotenv(config_usuario.ENV_PATH)
load_dotenv(os.path.join(BASE_DIR, ".env"))

URL = "https://automatizacion-upme.bizagi.com/"
USUARIO = os.environ.get("BIZAGI_USER_NEW", "")
PASSWORD = os.environ.get("BIZAGI_PASSWORD_NEW", "")
# Pausa que Playwright agrega a CADA acción (clic, tecla, fill...). Con 120 ms un
# solo campo de texto (~35 acciones) tardaba 4-5 s. Las esperas que Bizagi sí
# necesita ya están explícitas en el código, así que por defecto va en 0.
# Si la carga se vuelve inestable, subirlo en el .env: BIZAGI_SLOW_MO_MS=120
SLOW_MO_MS = int(os.environ.get("BIZAGI_SLOW_MO_MS", "0") or 0)


PDF_DIR = os.path.join(BASE_DIR, "Fichastecnicas")



# Selector del "+" obtenido del Chrome Recorder
ADD_PLUS_XPATH = '//*[@id="mp_IncentivosFNCE_idmInformacionsolicitud_xEquipos"]/div/div[2]/div[3]/table/tbody/tr/th[1]/div/div/ul/li[1]/div'

# Nuevo selector para el clip de adjunto (Soporte) proporcionado por el usuario
ATTACH_CLIP_XPATH = 'span.ui-icon.upload-file'


# ---------------- UTILIDADES ---------------- #
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        c2 = str(c).replace("\n", " ").replace("\r", " ")
        c2 = re.sub(r"\s+", " ", c2).strip()
        cols.append(c2)
    df.columns = cols
    return df


def s(x):
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return ""
    return str(x).strip()


def fmt_number(x):
    if x is None or pd.isna(x):
        return ""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return str(x)
    return str(x).strip()


def fmt_money(x):
    if x is None or pd.isna(x):
        return ""
    try:
        # Si ya es un número, redondear y convertir a entero
        if isinstance(x, (int, float)):
            return str(int(round(x)))
            
        # Si es string, eliminar $, espacios y separadores comunes
        st = str(x).replace("$", "").replace(" ", "").strip()
        
        # Si tiene puntos y comas (formato contable), nos quedamos con la parte entera
        if "," in st and "." in st:
            if st.rfind(",") > st.rfind("."):
                st = st.split(",")[0].replace(".", "")
            else:
                st = st.split(".")[0].replace(",", "")
        elif "," in st and len(st) - st.rfind(",") == 3:
            st = st.split(",")[0].replace(".", "")
        elif "." in st and len(st) - st.rfind(".") == 3:
            st = st.split(".")[0].replace(",", "")
            
        # Limpieza final: dejar solo dígitos
        res = re.sub(r"\D", "", st)
        return res
    except:
        return ""




def set_zoom(page, zoom=0.8):
    page.add_style_tag(content=f"html, body {{ zoom: {zoom}; }}")
    page.wait_for_timeout(250)


def do_login(page, usuario=None, password=None):
    usuario = usuario if usuario else USUARIO
    password = password if password else PASSWORD
    page.wait_for_timeout(1200)

    # Esperar a que la página de login cargue completamente
    page.wait_for_load_state("networkidle", timeout=15000)

    # ─── Usuario ────────────────────────────────────────────────────
    if page.locator("#username").count() > 0:
        us = page.locator("#username")
    elif page.locator("input[name='username']").count() > 0:
        us = page.locator("input[name='username']")
    else:
        us = page.locator("input[type='text']").first

    us.wait_for(state="visible", timeout=20000)
    us.click()
    us.fill("")
    us.type(usuario, delay=35)

    # ─── Contraseña ─────────────────────────────────────────────────
    if page.locator("#password").count() > 0:
        pw = page.locator("#password")
    elif page.locator("input[name='password']").count() > 0:
        pw = page.locator("input[name='password']")
    else:
        pw = page.locator("input[type='password']").first

    page.wait_for_timeout(500)
    pw.evaluate("el => { el.removeAttribute('style'); el.style.display='block'; }")
    pw.fill(password)

    # ─── Botón Ingresar ─────────────────────────────────────────────
    ingresar = page.get_by_role("button", name="Ingresar")
    if ingresar.count() > 0:
        ingresar.first.click()
    elif page.locator("button[type='submit']").count() > 0:
        page.locator("button[type='submit']").first.click()
    else:
        pw.press("Enter")


def open_inbox(page):
    if page.locator("text=Inbox").count() > 0:
        page.locator("text=Inbox").first.click()
    elif page.locator("text=Bandeja de entrada").count() > 0:
        page.locator("text=Bandeja de entrada").first.click()
    page.wait_for_timeout(2000)
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass


def open_case_by_radicado(page, radicado: str):
    if page.locator("input[placeholder*='Buscar']").count() > 0:
        search = page.locator("input[placeholder*='Buscar']").first
    elif page.locator("input[placeholder*='Search']").count() > 0:
        search = page.locator("input[placeholder*='Search']").first
    else:
        # Fallback: solo inputs VISIBLES, para no caer en campos ocultos
        # (p. ej. "categoryProceccess") que nunca serán clicables.
        search = page.locator("input:visible").first

    search.wait_for(state="visible", timeout=15000)
    search.click()
    search.fill(radicado)

    # Clave: el buscador de Bizagi solo dispara la búsqueda real con Enter.
    # Sin esto, el campo se queda con un aviso de "no hay casos" y el
    # radicado nunca llega a renderizarse en la lista de resultados.
    search.press("Enter")

    try:
        page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass

    # Tras el Enter, Bizagi puede llevar a una grilla de resultados (tabla)
    # en vez del dropdown de autocompletar. En la grilla, el clic que abre
    # el caso suele estar en la FILA completa, no en el texto suelto del
    # radicado -- por eso preferimos clicar el <tr> si existe, y solo si
    # no hay tabla caemos al texto plano (caso del dropdown antiguo).
    row = page.locator("tr").filter(has_text=radicado)
    result = row.first if row.count() > 0 else page.locator(f"text={radicado}").first

    try:
        result.wait_for(state="visible", timeout=15000)
    except Exception:
        raise RuntimeError(
            f"No se encontró el caso con radicado '{radicado}' tras buscar en el inbox. "
            "Verifica que el radicado exista y esté bien escrito."
        )

    result.click()
    page.wait_for_timeout(5000)


def open_tab_info_equipos(page, timeout_ms: int = 15000) -> bool:
    """
    Localiza y abre la pestaña 'Información de equipos'. Usa scroll
    incremental acotado (máx. 15 intentos) solo como mecanismo de
    visibilidad, pero la espera de clic/aparición usa wait_for real.
    """
    from bizagi_selectors import SELECTORS

    candidates = [page.locator(f"text={t}") for t in SELECTORS["tab_equipos_texts"]]

    for attempt in range(15):
        for loc in candidates:
            if loc.count() > 0:
                try:
                    loc.first.scroll_into_view_if_needed(timeout=2000)
                    loc.first.click(timeout=2000)
                    # Espera de condición: la pestaña quedó activa/cargada.
                    # Ajustar este selector si Bizagi expone una clase "active" o similar.
                    page.wait_for_load_state("networkidle", timeout=5000)
                    return True
                except Exception:
                    continue
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(200)  # scroll incremental, no espera de negocio -- se mantiene corto

    return False


EQUIPOS_GRID_ID = "mp_IncentivosFNCE_idmInformacionsolicitud_xEquipos"

# Candidatos para el "+": el XPath absoluto del Recorder depende de índices de
# div que cambian cuando la grilla crece (p. ej. aparece el paginador), así que
# se agregan alternativas relativas al contenedor de la grilla.
ADD_PLUS_CANDIDATES = [
    f"xpath={ADD_PLUS_XPATH}",
    f"#{EQUIPOS_GRID_ID} th ul li:first-child > div",
    f"#{EQUIPOS_GRID_ID} th [title*='dicionar' i]",
    f"#{EQUIPOS_GRID_ID} th [title*='gregar' i]",
]

# Campo "Nombre del Elemento": si está visible, el formulario de equipo está abierto.
FORM_EQUIPO_MARKER = "div[data-render-xpath='kpElementoFNCE']"


class FormularioNoCerrado(RuntimeError):
    """Guardar no cerró el formulario (validación de Bizagi o guardado incompleto)."""


def combos_vacios_en_validacion(mensaje: str) -> list:
    """Combos reintentables que el mensaje de validación de Bizagi reporta como vacíos.
    Ese mensaje es validación del lado del cliente: el guardado no llegó a Bizagi, así
    que se puede corregir el campo y volver a Guardar sin riesgo de duplicar el equipo."""
    m = (mensaje or "").lower()
    if "vac" not in m:
        return []
    return [k for k in COMBOS_REINTENTABLES if f"campo {k}" in m]


def formulario_equipo_abierto(page) -> bool:
    try:
        return page.locator(FORM_EQUIPO_MARKER).first.is_visible()
    except Exception:
        return False


def leer_mensajes_validacion(page) -> str:
    """Texto visible de errores/validaciones de Bizagi, para diagnosticar."""
    textos = []
    for sel in [".ui-bizagi-notifications-container", ".ui-bizagi-validation-message",
                ".ui-state-error", ".bz-rn-validation-message", ".ui-dialog .error"]:
        try:
            loc = page.locator(sel)
            for k in range(min(loc.count(), 5)):
                el = loc.nth(k)
                if el.is_visible():
                    t = el.inner_text(timeout=1000).strip()
                    if t and t not in textos:
                        textos.append(t)
        except Exception:
            pass
    return " | ".join(textos)[:500]


def cerrar_formulario_equipo(page):
    """Cierra un formulario de equipo que quedó abierto (Cancelar / X / Escape)."""
    for b in [
        page.get_by_role("button", name="Cancelar"),
        page.locator("button:has-text('Cancelar')"),
        page.locator(".ui-dialog-titlebar-close"),
    ]:
        try:
            if b.count() > 0 and b.first.is_visible():
                b.first.click(timeout=3000)
                page.wait_for_timeout(800)
                if not formulario_equipo_abierto(page):
                    return True
        except Exception:
            pass
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(800)
    except Exception:
        pass
    return not formulario_equipo_abierto(page)


def click_mas_equipos(page):
    print("  -> Buscando botón '+' para adicionar equipo...")

    if formulario_equipo_abierto(page):
        raise RuntimeError("Hay un formulario de equipo abierto del ítem anterior; "
                           "el '+' queda tapado. Revise si el ítem anterior se guardó.")

    # Intentar scroll agresivo hacia abajo si hay muchos items previos
    for i in range(10):
        for sel in ADD_PLUS_CANDIDATES:
            plus = page.locator(sel)
            try:
                if plus.count() > 0:
                    plus.first.scroll_into_view_if_needed(timeout=2000)
                    if plus.first.is_visible():
                        plus.first.click(timeout=5000)
                        return
            except Exception:
                pass

        # Si no lo ve, scrollear fuerte hacia abajo
        page.mouse.wheel(0, 2000)
        page.wait_for_timeout(500)

    # Último intento desesperado
    combined = page.locator(ADD_PLUS_CANDIDATES[0])
    for sel in ADD_PLUS_CANDIDATES[1:]:
        combined = combined.or_(page.locator(sel))
    combined.first.wait_for(state="visible", timeout=10000)
    combined.first.click(force=True)




def wait_form_adicionar_equipo(page, timeout_ms: int = 20000) -> bool:
    """
    Espera a que el formulario 'Adicionar equipo' esté visible, usando
    wait_for nativo de Playwright en vez de polling manual con timeouts fijos.
    """
    from bizagi_selectors import SELECTORS

    texts = SELECTORS["form_adicionar_texts"] + SELECTORS["form_adicionar_label_texts"]
    candidates = [page.locator(f"text={t}") if t in SELECTORS["form_adicionar_texts"]
                  else page.locator(f"label:has-text('{t}')") for t in texts]

    try:
        # or_ encadenado: espera a que CUALQUIERA de los candidatos aparezca
        combined = candidates[0]
        for c in candidates[1:]:
            combined = combined.or_(c)
        combined.first.wait_for(state="visible", timeout=timeout_ms)
        return True
    except Exception:
        return False

def click_guardar(page):
    """
    Busca y hace clic en el botón Guardar o Aceptar.
    """
    print("  -> Guardando equipo...")
    for b in [
        page.get_by_role("button", name="Guardar"),
        page.locator("button:has-text('Guardar')"),
        page.get_by_role("button", name="Aceptar"),
        page.locator("button:has-text('Aceptar')"),
    ]:
        try:
            if b.count() > 0:
                b.first.click(timeout=5000)
                return
        except Exception:
            pass
    page.keyboard.press("Enter")


def click_guardar_esperando_respuesta(page, log=print, timeout_ms: int = 15000):
    """
    Hace clic en Guardar/Aceptar y espera la respuesta de red real de Bizagi
    en vez de un tiempo fijo. Si el hint de red no coincide (por ejemplo,
    porque aún no se confirmó el endpoint exacto con DevTools), hace fallback
    a una espera de red genérica (networkidle) con timeout acotado.
    """
    from bizagi_selectors import SELECTORS

    hint = SELECTORS["network_hints"]["save_case"]
    action = SELECTORS["network_hints"]["save_case_action"]

    def es_guardado(r):
        # La URL de Render es compartida; el guardado se distingue por h_action en el cuerpo.
        if hint not in r.url:
            return False
        try:
            return action in (r.request.post_data or "")
        except Exception:
            return False

    try:
        with page.expect_response(es_guardado, timeout=timeout_ms) as resp_info:
            click_guardar(page)
        resp = resp_info.value
        if resp.status >= 400:
            raise RuntimeError(f"Bizagi respondió con status {resp.status} al guardar")
        log(f"     ✅ Guardado confirmado por red (status {resp.status}).")
    except Exception as e:
        log(f"     ⚠️ No se pudo confirmar el guardado por red ({e}). Usando espera de respaldo.")
        # Fallback: ya se hizo click_guardar dentro del expect_response (o no, si falló antes del click)
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            page.wait_for_timeout(2500)  # último recurso, mucho más corto que los 4000ms originales

    # Un 200 de SAVERELATION no garantiza que Bizagi aceptara el registro, así que
    # la prueba real de que se guardó es que el formulario se cierre. Si sigue abierto, Bizagi rechazó
    # el guardado (campo obligatorio, validación, adjunto pendiente...).
    try:
        page.locator(FORM_EQUIPO_MARKER).first.wait_for(state="hidden", timeout=15000)
    except Exception:
        detalle = leer_mensajes_validacion(page)
        raise FormularioNoCerrado(
            "El formulario no se cerró tras Guardar"
            + (f": {detalle}" if detalle else " (sin mensaje de validación visible)")
        )


OPCIONES_COMBO = "div.ui-select-dropdown.open li[role='option']"


def _normalizar_opcion(texto: str) -> str:
    """Minúsculas, sin tildes y con espacios colapsados, para comparar opciones."""
    t = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t).strip().lower()


def elegir_opcion_combo(value: str, opciones: list, umbral: float = 0.9):
    """
    Índice de la opción del combo que corresponde a value, o None.
    Primero igualdad normalizada; si no hay, la más parecida por encima del umbral.
    Necesario porque las listas de Bizagi tienen erratas propias (p. ej.
    "apantallemiento" en vez de "apantallamiento" en la opción de DPS).
    """
    objetivo = _normalizar_opcion(value)
    normalizadas = [_normalizar_opcion(o) for o in opciones]
    if objetivo in normalizadas:
        return normalizadas.index(objetivo)
    mejor, mejor_ratio = None, umbral
    for k, n in enumerate(normalizadas):
        ratio = difflib.SequenceMatcher(None, objetivo, n).ratio()
        if ratio >= mejor_ratio:
            mejor, mejor_ratio = k, ratio
    return mejor


def fill_field_by_xpath(page, render_xpath: str, value: str, is_dropdown=False):
    """
    Encuentra un campo mediante el atributo 'data-render-xpath' nativo de Bizagi.
    Este es el método MÁS EFECTIVO y robusto.
    """
    if not value:
        return

    print(f"  -> Llenando [{render_xpath}]: {value}")
    try:
        # Buscamos el div contenedor exacto provisto por Bizagi
        container = page.locator(f"div[data-render-xpath='{render_xpath}']").first
        container.wait_for(state="visible", timeout=5000)

        # Espera final por defecto tras soltar el campo (OnBlur). Los campos numéricos y el
        # dropdown la necesitan completa para que la máscara/AJAX de Bizagi termine de procesar;
        # el texto plano pegado con fill() la recorta más abajo.
        final_wait = 300

        if is_dropdown:
            # Para comboboxes (Nombre del elemento)
            # Primero localizar el input visible (el que dice "Seleccione...")
            # Tiempos verificados en vivo contra el DOM/red real de Bizagi (no estimados):
            # - Abrir el combobox: la lista completa queda poblada en el DOM en <30ms, sin
            #   ninguna petición de red de por medio (widget jQuery UI puramente cliente).
            # - Escribir para "filtrar": NO existe filtrado por red. Es un typeahead cliente
            #   que solo resalta/hace scroll a la opción más parecida; la lista completa
            #   sigue en el DOM todo el tiempo. Por eso la búsqueda por texto de más abajo
            #   funciona igual sin depender de que "termine" ningún filtro.
            # - Seleccionar la opción: este SÍ dispara peticiones reales
            #   (POST /Rest/Handlers/Render), medidas en ~600ms combinados. Es el único de
            #   los tres pasos que de verdad necesita el margen de espera.
            dropdown_input = container.locator("input.ui-selectmenu-value, input[role='combobox']").first
            dropdown_input.click(force=True)
            page.wait_for_timeout(100) # margen sobre los <30ms reales de apertura del widget

            # Igual que un usuario (verificado con diagnostico_combo.py): abrir la lista y
            # hacer clic en la opción, sin escribir. La opción se elige aquí comparando
            # textos normalizados, tolerando erratas de la lista de Bizagi.
            opciones = page.locator(OPCIONES_COMBO)
            try:
                opciones.first.wait_for(state="attached", timeout=2000)
                textos = opciones.all_inner_texts()
            except Exception:
                textos = []
            k = elegir_opcion_combo(value, textos)
            if k is not None:
                if _normalizar_opcion(textos[k]) != _normalizar_opcion(value):
                    print(f"     ℹ️ '{value}' no está igual en Bizagi; se elige la opción '{textos[k].strip()}'.")
                option = opciones.nth(k)
                option.scroll_into_view_if_needed(timeout=2000)
                option.click()
                page.wait_for_timeout(700) # margen sobre los ~600ms reales de los Render tras seleccionar
                page.wait_for_timeout(final_wait)
                return
            print(f"     ⚠️ No hay opción parecida a '{value}' entre {len(textos)} opciones; probando con tecleo.")

            # Escribir para que el typeahead resalte la opción (no filtra nada vía red, pero
            # sigue siendo necesario un tecleo real para que Bizagi actualice el value/estado
            # interno del input, a diferencia de fill()).
            dropdown_input.fill("")
            dropdown_input.type(value, delay=15)
            page.wait_for_timeout(150) # margen corto: no hay red que esperar en este paso

            # Hacer clic en la opción que coincida (dentro de la lista flotante que Bizagi
            # agrega al body al abrir el combobox: div.ui-select-dropdown > ul[role='listbox']
            # > li[role='option']). El selector anterior (ul.ui-selectmenu-menu-dropdown
            # li[role='presentation'] a) no hace match en la versión actual de Bizagi -
            # verificado en vivo que option.count() siempre daba 0 con ese selector, por lo
            # que el script dependía siempre del fallback de Enter en la línea de abajo.
            option = page.locator("div.ui-select-dropdown li[role='option']").filter(has_text=re.compile(f"^{re.escape(value)}$", re.IGNORECASE)).first
            if option.count() == 0:
                # Si no hay match exacto, probar con el que contiene el texto
                option = page.locator("div.ui-select-dropdown li[role='option']").filter(has_text=value).first

            if option.count() > 0:
                # Con listas largas la opción puede quedar fuera de la zona visible del
                # listbox; con force=True el clic caía en otra parte y el combo quedaba vacío.
                try:
                    option.scroll_into_view_if_needed(timeout=2000)
                except Exception:
                    pass
                option.click(force=True)
            else:
                # Fallback: presionar Enter si la opción no se detecta claramente
                print(f"     ⚠️ Opción '{value}' no encontrada en la lista; usando Enter.")
                page.keyboard.press("Enter")

            page.wait_for_timeout(700) # margen sobre los ~600ms reales de los Render tras seleccionar

        else:
            # Inputs normales (texto/numero)
            field = container.locator("input:not([type='hidden']), textarea").first
            field.click(force=True)

            # Escribir simulando teclado real, dependiendo del tipo de campo
            # Si es campo numérico/monetario, usamos press carácter por carácter para las máscaras
            is_numeric = render_xpath in ["cValorIVAenCOP", "cValortotalenCOPsinIV", "cValortotalenCOPsinIVA", "iCantidad"]

            # La espera post-limpieza se duplica por rama (en vez de compartirse antes del if)
            # para poder recortarla en texto plano sin tocar el margen que necesita la máscara
            # numérica.
            if is_numeric:
                # Método más riguroso para limpiar inputs numéricos con máscaras en Bizagi.
                # Solo aquí: en texto plano fill() ya reemplaza el contenido, y esta limpieza
                # (~30 acciones) era la mayor parte del tiempo por campo.
                field.press("End")
                for _ in range(25):
                    field.press("Backspace")

                field.fill("")
                field.press("Control+A")
                field.press("Delete")

                page.wait_for_timeout(150)

                # NO usar fill(): estos campos tienen una máscara JS de Bizagi que reformatea
                # el valor mientras se teclea, y fill() rompe esa máscara. Se mantiene el
                # tecleo carácter por carácter, solo se acorta el delay por carácter.
                # Limpieza extra para campos con máscara
                field.click(force=True)
                field.press("Control+A")
                field.press("Backspace")
                page.wait_for_timeout(150)

                for char in str(value):
                    page.keyboard.press(char)
                    page.wait_for_timeout(15) # Delay controlado para que la máscara procese
            else:
                page.wait_for_timeout(50)

                # Campos de texto plano (sin máscara ni filtrado en vivo): se pega el valor
                # completo de una vez con fill(), que es mucho más rápido que tecla por tecla.
                field.fill(str(value))
                # Disparamos 'input'/'change' manualmente por si Bizagi depende de esos
                # eventos (y no solo del value del DOM) para refrescar validaciones OnBlur.
                field.dispatch_event("input")
                field.dispatch_event("change")

                # Sin tecleo real detrás, el margen para el OnBlur también se puede acortar.
                final_wait = 100

            # Clicar fuera y tabular para forzar el guardado temporal (OnBlur)
            page.keyboard.press("Tab")
            page.wait_for_timeout(80 if not is_numeric else 200)
            page.mouse.click(10, 10)

        page.wait_for_timeout(final_wait)
    except Exception as e:
        print(f"     ⚠️ Error en campo con xpath '{render_xpath}': {e}")


def find_pdfs_for_item(idx: int, pdf_folder: str) -> list:
    """
    MÉTODO ANTIGUO, YA NO USADO en el flujo principal (ver find_pdfs_by_link).
    Se conserva por si se necesita revertir o comparar en el futuro.

    Encuentra todos los PDFs en la carpeta que correspondan al índice (item).
    El índice 0 equivale al item 1 (1., 1 , etc).
    Maneja rangos también (e.g. 42-43).
    """
    item_num = idx + 1
    if not os.path.exists(pdf_folder):
        print(f"  -> La carpeta de PDFs {pdf_folder} no existe.")
        return []
        
    matched_files = []
    files = os.listdir(pdf_folder)
    
    for f in files:
        if not f.lower().endswith(".pdf"):
            continue
            
        # Coincidencia de número base con decimales o espacio:
        # e.g., "10 IINTERFLEX.pdf", "10.1 IINTERFLEX.pdf", "2. EN_Certificate..."
        m = re.match(r"^(\d+)(?:\.\d+)?[\s\.]", f)
        if m:
            base_num = int(m.group(1))
            if base_num == item_num:
                matched_files.append(os.path.join(pdf_folder, f))
                continue
                
        # Coincidencia de rango: e.g., "42-43 Certificado..."
        m_range = re.match(r"^(\d+)\-(\d+)[\s\.]", f)
        if m_range:
            start, end = int(m_range.group(1)), int(m_range.group(2))
            if start <= item_num <= end:
                matched_files.append(os.path.join(pdf_folder, f))
                
    return matched_files


# ---------------- MODELO POR ODOO ID ---------------- #
# Los archivos a subir viven en Fichastecnicas con el Odoo ID como prefijo,
# seguido del código de origen (lo genera preparar_por_odoo.py):
#   P01243_CERT001_<nombre>.pdf  -> certificado de conformidad (se sube primero)
#   P01243_L015_<nombre>.pdf     -> soporte técnico
# El maestro El_Papá.xlsx (hoja FORMATO 3) dice qué códigos lleva cada Odoo ID.
MAESTRO_DEFAULT = os.path.join(BASE_DIR, "El_Papá.xlsx")
_SEPARADOR_CODIGOS = re.compile(r"[,;/\s]+")


def normalizar_odoo_id(valor) -> str | None:
    """Odoo ID limpio y en mayúsculas (ej. ' p01243 ' -> 'P01243'), o None si viene vacío/NaN."""
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return None
    if isinstance(valor, float) and valor.is_integer():
        valor = int(valor)  # 1932.0 (celda numérica en Excel) -> "1932"
    v = str(valor).strip().upper()
    return v if v and v not in ("NAN", "NONE") else None


def _codigos_de_celda(valor) -> list:
    """'L049, L050' -> ['L049', 'L050']; vacío/NaN -> []."""
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return []
    return [c.strip().upper() for c in _SEPARADOR_CODIGOS.split(str(valor)) if c.strip()]


def cargar_maestro_odoo(maestro_path: str, log=print) -> dict:
    """
    Lee el Excel maestro (El_Papá.xlsx, hoja "FORMATO 3" o la primera) y
    devuelve {odoo_id: {"soportes": [códigos L...], "certificados": [códigos CERT...]}}.
    Un mismo Odoo ID puede tener varias filas: se unen todos sus códigos.
    Si el archivo no existe o no se puede leer, advierte y devuelve {}.
    """
    if not maestro_path or not os.path.exists(maestro_path):
        log(f"⚠️ No se encontró el maestro de Odoo ID '{maestro_path}'.")
        return {}
    try:
        xl = pd.ExcelFile(maestro_path)
        hojas = [h for h in xl.sheet_names if "formato 3" in h.lower()]
        hoja = hojas[0] if hojas else xl.sheet_names[0]
        raw = pd.read_excel(xl, sheet_name=hoja, header=None)
        header_idx = next((i for i, r in raw.iterrows()
                           if any("odoo" in str(x).lower() for x in r.values)), 0)
        df = normalize_columns(pd.read_excel(xl, sheet_name=hoja, header=header_idx))
    except Exception as e:
        log(f"⚠️ No se pudo leer el maestro de Odoo ID '{maestro_path}' ({e}).")
        return {}

    maestro = {}
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        odoo_id = normalizar_odoo_id(get_col(row_dict, "Odoo ID", "Odoo"))
        if not odoo_id:
            continue
        entrada = maestro.setdefault(odoo_id, {"soportes": [], "certificados": []})
        for clave, columnas in (("soportes", ("Código Soporte", "Codigo Soporte")),
                                ("certificados", ("Código Certificado", "Codigo Certificado"))):
            for codigo in _codigos_de_celda(get_col(row_dict, *columnas)):
                if codigo not in entrada[clave]:
                    entrada[clave].append(codigo)
    return maestro


def find_archivos_por_odoo(odoo_id: str, pdf_dir: str) -> tuple:
    """
    Devuelve (certificados, soportes): rutas de los PDFs de pdf_dir cuyo nombre
    empieza exactamente por "{odoo_id}_". Los que siguen con "CERT" son
    certificados; el resto, soporte técnico. Se suben todos.
    """
    certificados, soportes = [], []
    if not odoo_id or not os.path.exists(pdf_dir):
        return certificados, soportes
    prefijo = f"{odoo_id}_"
    for f in sorted(os.listdir(pdf_dir)):
        if not (f.lower().endswith(".pdf") and f.upper().startswith(prefijo)):
            continue
        ruta = os.path.join(pdf_dir, f)
        (certificados if f[len(prefijo):].upper().startswith("CERT") else soportes).append(ruta)
    return certificados, soportes


# ---------------- FICHAS SEPARADAS POR MARCA ---------------- #
# La tubería IMC de estos Odoo ID se compra a dos marcas con fichas distintas.
# preparar_por_odoo.py etiqueta sus PDFs con la marca después del código
# (P01072_L049_KUBIEC_<nombre>.pdf, P01072_L050_COLMENA_<nombre>.pdf) y cada
# fila sube solo las fichas de su marca, leída de las columnas Marca/Fabricante.
ODOO_IDS_POR_MARCA = {"P01072", "P01076", "P01078"}
CODIGO_MARCA = {"L049": "KUBIEC", "L050": "COLMENA"}
MARCAS_SEPARADAS = sorted(set(CODIGO_MARCA.values()))


def detectar_marca(row: dict) -> str | None:
    """'KUBIEC' o 'COLMENA' si aparece (solo una de ellas) en Marca o Fabricante; si no, None."""
    texto = f"{s(get_col(row, 'Marca'))} {s(get_col(row, 'Fabricante'))}".upper()
    encontradas = [m for m in MARCAS_SEPARADAS if m in texto]
    return encontradas[0] if len(encontradas) == 1 else None


def marca_de_archivo(ruta: str) -> str | None:
    """Marca etiquetada en el nombre (P01072_L049_KUBIEC_x.pdf -> 'KUBIEC'), o None si no tiene."""
    partes = os.path.basename(ruta).upper().split("_")
    return partes[2] if len(partes) > 3 and partes[2] in MARCAS_SEPARADAS else None


def _primer_pdf_fallback(pdf_dir: str) -> str | None:
    """Primer PDF, orden alfabético, del primer nivel de pdf_dir (o None si no hay ninguno).
    Ignora los certificados, que nunca deben usarse como placeholder de soporte."""
    if not os.path.exists(pdf_dir):
        return None
    pdfs = sorted(f for f in os.listdir(pdf_dir)
                  if f.lower().endswith(".pdf") and not re.search(r"_CERT\d*_", f, re.IGNORECASE))
    if not pdfs:
        return None
    return os.path.join(pdf_dir, pdfs[0])


def attach_files_to_equipment(page, file_paths: list, log=print) -> bool:
    """
    Sube múltiples archivos navegando la interfaz de Bizagi:
    1. Clic en el icono inicial de subida (<span class="ui-icon upload-file"></span>).
    2. Esperar modal (por selector, no por tiempo fijo).
    3. Inyectar el archivo en el input file nativo.
    4. Clic en el botón interno de 'Subir' y esperar confirmación por red cuando sea posible.

    Devuelve True si TODOS los archivos se subieron sin error, False si al menos uno falló.
    """
    from bizagi_selectors import SELECTORS

    if not file_paths:
        log("  -> No hay PDFs encontrados para este equipo.")
        return False

    all_ok = True
    for path in file_paths:
        log(f"  -> Adjuntando archivo: {os.path.basename(path)}")
        try:
            # 1. Seleccionar el botón de upload principal
            clip = page.locator(SELECTORS["attach_clip_container_css"]).first
            if clip.count() == 0:
                clip = page.locator(SELECTORS["attach_clip_css"]).first

            clip.scroll_into_view_if_needed()
            clip.wait_for(state="visible", timeout=5000)
            clip.click(force=True)

            # Esperar a que el input de archivo del modal esté disponible en el DOM,
            # en vez de un wait_for_timeout(2500) a ciegas.
            upload_input = page.locator(SELECTORS["attach_file_input_css"])
            upload_input.wait_for(state="attached", timeout=8000)

            upload_input.set_input_files(path)
            log("     ✅ Archivo inyectado en el sistema.")

            # Esperar a que aparezca el botón "Subir" (confirma que Bizagi procesó la inyección)
            btn_subir = page.locator(SELECTORS["attach_upload_button_xpath"]).first
            try:
                btn_subir.wait_for(state="visible", timeout=6000)
            except Exception:
                btn_subir = page.locator(SELECTORS["attach_upload_button_fallback_css"]).first
                btn_subir.wait_for(state="visible", timeout=6000)

            hint = SELECTORS["network_hints"]["upload_file"]
            try:
                with page.expect_response(lambda r: hint in r.url, timeout=10000) as resp_info:
                    btn_subir.click(force=True)
                resp = resp_info.value
                if resp.status >= 400:
                    raise RuntimeError(f"status {resp.status}")
                log("     ✅ Subida confirmada por red.")
            except Exception as e_net:
                log(f"     ⚠️ No se confirmó la subida por red ({e_net}). Verificando con espera de respaldo.")
                page.wait_for_timeout(2000)  # fallback corto, no los 3500ms originales

        except Exception as e:
            log(f"     ⚠️ Error adjuntando archivo ({os.path.basename(path)}): {e}")
            all_ok = False

    return all_ok


def write_value_then_tab(page, value: str, delay_after_type_ms=140):
    """
    ✅ CORRECCIÓN:
    Escribe y SOLO DESPUÉS hace TAB. Si está vacío, solo TAB.
    Esto evita que el cursor se descuadre por latencia de Bizagi.
    """
    if value:
        page.keyboard.type(value, delay=10)
        page.wait_for_timeout(delay_after_type_ms)
    page.keyboard.press("Tab")
    page.wait_for_timeout(70)


def get_col(row: dict, *names):
    """Busca una columna por nombre exacto primero, luego por contenido parcial."""
    # 1. Match exacto
    for n in names:
        if n in row:
            return row.get(n)
    # 2. Match parcial: buscar columna que CONTENGA alguno de los nombres
    for n in names:
        n_lower = n.lower()
        for k in row.keys():
            k_clean = str(k).replace("\n", " ").replace("\r", " ").lower().strip()
            if n_lower in k_clean:
                return row.get(k)
    return None


COLS_NOMBRE = ("Nombre del Elemento/Equipo/Maquinaria", "Nombre del Elemento", "Nombre del elemento", "Nombre Elemento", "Nombre")
COLS_UNIDAD = ("Unidad de Medida", "Unidad", "Unidad medida")

# Combos obligatorios que Bizagi puede rechazar como vacíos aunque se hayan llenado
# (la selección de la opción no siempre "pega"): etiqueta en el mensaje de validación
# -> (data-render-xpath, columnas del Excel).
COMBOS_REINTENTABLES = {
    "nombre del elemento": ("kpElementoFNCE", COLS_NOMBRE),
    "unidad de medida": ("kp_INCUnidadMedidad", COLS_UNIDAD),
}
COLS_IVA = ("Valor IVA en COP", "Valor IVA", "IVA", "Valor del IVA")
COLS_VALOR_SIN_IVA = ("Valor total en COP (Sin incluir IVA)", "Valor total en COP\n(Sin incluir IVA)", "Valor total en COP",
                      "Valor total (Sin IVA)", "Valor sin IVA", "Valor total sin IVA")


def fill_equipo_with_clicks(page, row: dict, idx: int):
    """
    Llenado con CLICS DIRECTOS. 
    Toma los valores del Excel de manera dinámica para cada elemento.
    """
    nombre = s(get_col(row, "Nombre del Elemento/Equipo/Maquinaria", "Nombre del Elemento", "Nombre del elemento", "Nombre Elemento", "Nombre"))
    marca = s(get_col(row, "Marca"))
    subpartida = s(get_col(row, "Subpartida arancelaria", "Subpartida"))
    unidad = s(get_col(row, *COLS_UNIDAD))
    fabricante = s(get_col(row, "Fabricante"))
    # Se busca la columna "Función", incluyendo la versión con el caracter corrupto por la codificación del Excel (Funcin / Funci\ufffdn)
    funcion_keys = ["Función", "Funcion", "Funcion ", "Funci\ufffdn"]
    # Agregar iterativamente cualquier columna que empiece por "Funci"
    funcion = s(get_col(row, *funcion_keys))
    if not funcion:
        # Fallback si las llaves no funcionaron, buscar la columna que contenga "Funci"
        for k in row.keys():
            if "Funci" in str(k):
                funcion = s(row[k])
                break
    modelo = s(get_col(row, "Modelo / Referencia", "Modelo", "Referencia"))
    cantidad = fmt_number(get_col(row, "Cantidad"))
    normas = s(get_col(row, "Normas técnicas", "Normas", "Norma tecnica"))
    proveedor = s(get_col(row, "Proveedor"))
    
    # Búsqueda fortalecida de columnas monetarias usando los nombres literales y aproximaciones
    # Vacío, "$ -" o cualquier texto sin dígitos en el Excel se carga como IVA 0.
    iva_val = fmt_money(get_col(row, *COLS_IVA)) or "0"

    valor_sin_iva = fmt_money(get_col(row, *COLS_VALOR_SIN_IVA))

    # Llenado usando el mapeo exacto de Bizagi (data-render-xpath)
    fill_field_by_xpath(page, "kpElementoFNCE", nombre, is_dropdown=True)
    fill_field_by_xpath(page, "sMarca", marca)
    fill_field_by_xpath(page, "sSubpartidaarancelaria", subpartida)
    fill_field_by_xpath(page, "kp_INCUnidadMedidad", unidad, is_dropdown=True)
    fill_field_by_xpath(page, "sFabricante", fabricante)
    fill_field_by_xpath(page, "sFuncion", funcion)
    fill_field_by_xpath(page, "cValorIVAenCOP", iva_val)
    fill_field_by_xpath(page, "sModeloReferencia", modelo)
    fill_field_by_xpath(page, "iCantidad", cantidad)
    fill_field_by_xpath(page, "sNormastecnicas", normas)
    fill_field_by_xpath(page, "sProveedor", proveedor)
    fill_field_by_xpath(page, "cValortotalenCOPsinIV", valor_sin_iva)
    
    # Eliminamos la llamada aquí para evitar duplicidad, se hará en el main loop


# Mapeo de estado de soporte -> texto legible para el reporte de auditoría
ESTADO_SOPORTE_TEXTOS = {
    "match_odoo": "Soporte encontrado por Odoo ID",
    "solo_certificado": "Solo certificado (el Odoo ID no tiene soporte técnico)",
    "sin_odoo": "⚠️ Placeholder - la fila no tiene Odoo ID",
    "odoo_no_en_maestro": "⚠️ Placeholder - Odoo ID no está en el maestro",
    "odoo_sin_archivos": "⚠️ Placeholder - Odoo ID sin archivos locales",
    "marca_no_identificada": "⚠️ Marca no identificada (Kubiec/Colmena) - se subieron las fichas de todas las marcas",
}


def armar_archivos_a_subir(row_dict, pdf_dir, maestro_odoo, log=print) -> tuple:
    """
    Arma la lista final de archivos a adjuntar a un ítem a partir de su Odoo ID:
    primero el/los certificado(s) de conformidad, después el soporte técnico.
    Si no hay ningún archivo, adjunta un PLACEHOLDER con advertencia (como antes).
    Devuelve (lista_de_rutas, estado_soporte, lleva_certificado).
    """
    odoo_id = normalizar_odoo_id(get_col(row_dict, "Odoo ID", "Odoo", "ID Odoo"))
    certificados, soportes = find_archivos_por_odoo(odoo_id, pdf_dir)

    estado_ok = "match_odoo"
    if odoo_id in ODOO_IDS_POR_MARCA:
        marca = detectar_marca(row_dict)
        if marca:
            def _de_su_marca(rutas):
                return [r for r in rutas if marca_de_archivo(r) in (None, marca)]
            certificados, soportes = _de_su_marca(certificados), _de_su_marca(soportes)
            if not (certificados or soportes):
                log(f"⚠️ FICHAS POR MARCA: no hay archivos de '{marca}' para '{odoo_id}'.")
        else:
            estado_ok = "marca_no_identificada"
            log(f"⚠️ MARCA NO IDENTIFICADA: la fila de '{odoo_id}' no dice {' ni '.join(MARCAS_SEPARADAS)} "
                f"en Marca/Fabricante. Se suben las fichas de todas las marcas — revisar manualmente.")

    esperado = maestro_odoo.get(odoo_id, {}) if odoo_id else {}
    if esperado.get("certificados") and not certificados:
        log(f"⚠️ CERTIFICADO NO ENCONTRADO: el maestro indica {', '.join(esperado['certificados'])} para "
            f"'{odoo_id}' pero no hay archivos locales. No se adjuntará certificado para este ítem.")

    if soportes:
        return certificados + soportes, estado_ok, bool(certificados)
    if certificados:
        return certificados, "solo_certificado" if estado_ok == "match_odoo" else estado_ok, True

    if not odoo_id:
        estado, motivo = "sin_odoo", "la fila no tiene Odoo ID"
    elif odoo_id not in maestro_odoo:
        estado, motivo = "odoo_no_en_maestro", f"el Odoo ID '{odoo_id}' no está en el maestro"
    else:
        estado, motivo = "odoo_sin_archivos", f"el Odoo ID '{odoo_id}' no tiene archivos locales"

    placeholder = _primer_pdf_fallback(pdf_dir)
    if placeholder:
        log(f"⚠️ SOPORTE NO VERIFICADO: {motivo}. Se adjuntó '{os.path.basename(placeholder)}' "
            f"como PLACEHOLDER — revisar y corregir manualmente.")
        return [placeholder], estado, False
    return [], estado, False


def guardar_con_reintento_de_combos(page, row_dict, item_num, log=print, max_reintentos: int = 2):
    """
    Guarda el equipo. Si Bizagi lo rechaza porque un combo obligatorio quedó vacío
    (Nombre del elemento / Unidad de medida), vuelve a seleccionarlo en el mismo
    formulario (los adjuntos ya subidos se conservan) y guarda de nuevo.
    """
    for reintento in range(max_reintentos + 1):
        try:
            click_guardar_esperando_respuesta(page, log)
            return
        except FormularioNoCerrado as e:
            vacios = combos_vacios_en_validacion(str(e))
            if not vacios or reintento == max_reintentos or not formulario_equipo_abierto(page):
                raise
            for k in vacios:
                xpath, cols = COMBOS_REINTENTABLES[k]
                valor = s(get_col(row_dict, *cols))
                log(f"     ⚠️ Ítem {item_num}: Bizagi dejó vacío '{k}'. "
                    f"Reseleccionando '{valor}' (reintento {reintento + 1}/{max_reintentos})...")
                fill_field_by_xpath(page, xpath, valor, is_dropdown=True)


def procesar_item_con_reintentos(page, row_dict, i, item_num, pdf_dir, maestro_odoo,
                                  log=print, max_intentos: int = 2):
    """
    Procesa un único ítem (abrir formulario, llenar, adjuntar, guardar) con
    reintentos localizados. Si falla tras max_intentos, devuelve
    ("Error fatal", mensaje, estado_soporte, lleva_certificado) para que el
    llamador decida abortar todo el proceso; cualquier otro estado permite
    seguir con el siguiente ítem.
    """
    ultimo_error = None

    for intento in range(1, max_intentos + 1):
        try:
            click_mas_equipos(page)

            if not wait_form_adicionar_equipo(page):
                raise RuntimeError("No abrió el formulario de 'Adicionar equipo'.")

            fill_equipo_with_clicks(page, row_dict, i)

            pdfs_to_upload, estado_soporte, lleva_certificado = armar_archivos_a_subir(
                row_dict, pdf_dir, maestro_odoo, log=log
            )
            adjunto_ok = True
            if pdfs_to_upload:
                adjunto_ok = attach_files_to_equipment(page, pdfs_to_upload, log=log)
            else:
                log("     ℹ️ No se encontraron PDFs para este elemento.")

            guardar_con_reintento_de_combos(page, row_dict, item_num, log)

            if not adjunto_ok:
                return "Guardado sin adjunto", "Fallo al adjuntar soporte técnico", estado_soporte, lleva_certificado
            return "Exitoso", None, estado_soporte, lleva_certificado

        except FormularioNoCerrado as e:
            # No se reintenta: si el guardado llegó a Bizagi tarde, reintentar
            # duplicaría el equipo. Se deja el formulario abierto para revisión.
            log(f"  ❌ Ítem {item_num}: {e}")
            return "Error fatal", str(e), None, False

        except Exception as e:
            ultimo_error = str(e)
            log(f"  ⚠️ Intento {intento}/{max_intentos} falló para ítem {item_num}: {e}")
            if intento < max_intentos:
                log("     Reintentando tras recargar el estado del formulario...")
                # Cerrar cualquier formulario a medio llenar antes de reintentar
                if formulario_equipo_abierto(page) and not cerrar_formulario_equipo(page):
                    log("     ⚠️ No se pudo cerrar el formulario abierto.")

    return "Error fatal", ultimo_error, None, False


def guardar_reporte_auditoria(report_data: list, excel_path: str, log=print):
    """
    Genera un Excel de auditoría junto al archivo Excel de origen, con el
    estado de carga de cada ítem procesado (Exitoso / Guardado sin adjunto /
    Error fatal) y el estado de emparejamiento del soporte técnico ("Estado
    Soporte"). Nunca lanza excepción: un fallo al guardar el reporte no debe
    hacer perder el resultado de la carga ya realizada.
    """
    if not report_data:
        return
    try:
        report_df = pd.DataFrame(report_data)
        report_path = os.path.join(os.path.dirname(excel_path), "Reporte_Carga_Soportes.xlsx")
        report_df.to_excel(report_path, index=False)
        log(f"📊 Reporte de auditoría generado en: {report_path}")
    except Exception as e:
        log(f"⚠️ No se pudo generar el reporte de auditoría: {e}")


# ---------------- MAIN ---------------- #
def _es_fila_cabecera(valores) -> bool:
    row_str = " ".join([str(x).lower() for x in valores])
    return ("elemento" in row_str or "equipo" in row_str) and "marca" in row_str


def _es_item_valido(nombre: str) -> bool:
    """Mismo filtro de load_clean_excel: nombre presente y que no sea una fila de totales."""
    return bool(nombre) and nombre.lower() not in ("nan", "none") and "total" not in nombre.lower()


def load_clean_excel(file_path: str, sheet_name: str) -> pd.DataFrame:
    print("Analizando estructura del archivo Excel para encontrar cabeceras...")
    raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    header_idx = -1
    for i, row in raw_df.iterrows():
        if _es_fila_cabecera(row.values):
            header_idx = i
            break
            
    if header_idx != -1:
        print(f"-> Cabeceras encontradas dinámicamente en la fila {header_idx + 1} del Excel.")
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx)
    else:
        print("-> No se detectó preámbulo, leyendo de modo estándar.")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    df = normalize_columns(df)
    
    # Filtrar solo las filas que tengan un Nombre válido de elemento (ignorar footers vacíos)
    valid_rows = []
    for _, row in df.iterrows():
        n = s(get_col(row.to_dict(), *COLS_NOMBRE))
        if _es_item_valido(n):
            valid_rows.append(row)

    if valid_rows:
        df = pd.DataFrame(valid_rows).reset_index(drop=True)
    return df


# ---------------- COSTO OBJETIVO ---------------- #
# Opcional: el usuario da un valor total deseado SIN IVA (Td). Con Ta = suma del
# "Valor total en COP (Sin incluir IVA)" de todos los ítems, f = Td / Ta y se
# multiplican por f el valor sin IVA y el IVA de cada ítem. Se genera una copia
# del Excel con los valores ajustados y la carga a Bizagi se hace desde esa copia.
def parse_costo_objetivo(texto) -> int | None:
    """'1.250.000.000' / '$ 1,250,000,000' / '1250000000,50' -> 1250000000; inválido o <= 0 -> None."""
    valor = fmt_money(s(texto))
    return int(valor) if valor and int(valor) > 0 else None


def _numero_celda(valor):
    """Valor numérico de una celda de dinero (número o texto con formato contable), o None si está vacía."""
    if valor is None or (isinstance(valor, str) and not valor.strip()):
        return None
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        return None if math.isnan(valor) else float(valor)
    txt = fmt_money(valor)
    return float(txt) if txt else None


def generar_excel_costo_objetivo(excel_path: str, sheet_name: str, costo_objetivo: int, log=print) -> str:
    """
    Crea '<nombre>_costo_objetivo.xlsx' junto al Excel original, con el valor sin
    IVA y el IVA de cada ítem multiplicados por f = Td / Ta. Los valores se
    redondean a pesos enteros (así los recibe Bizagi) y el residuo del redondeo
    se suma al ítem de mayor valor, para que el total sin IVA sea exactamente Td.
    Devuelve la ruta del Excel ajustado.
    """
    import openpyxl

    wb = openpyxl.load_workbook(excel_path)
    wb_valores = openpyxl.load_workbook(excel_path, data_only=True)  # resultado de fórmulas, si las hay
    ws, ws_val = wb[sheet_name], wb_valores[sheet_name]

    filas = [[c.value for c in r] for r in ws_val.iter_rows()]
    header_idx = next((i for i, r in enumerate(filas) if _es_fila_cabecera(r)), 0)
    cabeceras = {}  # cabecera normalizada (igual que normalize_columns) -> índice de columna
    for j, h in enumerate(filas[header_idx] if filas else []):
        if h is not None:
            cabeceras.setdefault(re.sub(r"\s+", " ", str(h).replace("\n", " ").replace("\r", " ")).strip(), j)
    mapa = {h: h for h in cabeceras}
    col_nombre, col_valor, col_iva = (get_col(mapa, *cols) for cols in (COLS_NOMBRE, COLS_VALOR_SIN_IVA, COLS_IVA))
    if col_valor is None:
        raise RuntimeError("No se encontró la columna 'Valor total en COP (Sin incluir IVA)' en el Excel.")
    if col_nombre is None:
        raise RuntimeError("No se encontró la columna 'Nombre del Elemento' en el Excel.")
    j_nombre, j_valor = cabeceras[col_nombre], cabeceras[col_valor]
    j_iva = cabeceras[col_iva] if col_iva is not None else None

    # (fila_excel, valor_sin_iva, iva) de cada ítem que se cargaría
    items = []
    for i in range(header_idx + 1, len(filas)):
        r = filas[i]
        if not _es_item_valido(s(r[j_nombre])):
            continue
        items.append((i + 1, _numero_celda(r[j_valor]), _numero_celda(r[j_iva]) if j_iva is not None else None))

    total_actual = sum(v for _, v, _ in items if v)
    if total_actual <= 0:
        raise RuntimeError("El total sin IVA de los ítems del Formato 3 es 0; no se puede aplicar el costo objetivo.")
    factor = costo_objetivo / total_actual
    log(f"Costo objetivo (Td, sin IVA): $ {costo_objetivo:,.0f}")
    log(f"Total actual   (Ta, sin IVA): $ {total_actual:,.0f}")
    log(f"Factor f = Td / Ta = {factor:.10f}")

    nuevos_valores = {fila: int(round(v * factor)) for fila, v, _ in items if v is not None}
    residuo = costo_objetivo - sum(nuevos_valores.values())
    if residuo and nuevos_valores:
        fila_mayor = max(nuevos_valores, key=nuevos_valores.get)
        nuevos_valores[fila_mayor] += residuo
        log(f"Residuo de redondeo ({residuo:+d} COP) ajustado en la fila {fila_mayor} del Excel (ítem de mayor valor).")
    nuevos_iva = {fila: int(round(iv * factor)) for fila, _, iv in items if iv is not None}

    # openpyxl no guarda el resultado de las fórmulas: si se dejan como fórmula,
    # el Excel ajustado se leería con esas celdas vacías (p. ej. una Cantidad
    # "=21900*2/3"). Se reemplazan por su valor calculado en todas las hojas.
    sin_valor = []
    for hoja in wb.worksheets:
        hoja_val = wb_valores[hoja.title]
        for fila_celdas in hoja.iter_rows():
            for celda in fila_celdas:
                if celda.data_type == "f":
                    valor = hoja_val.cell(row=celda.row, column=celda.column).value
                    celda.value = valor
                    if valor is None:
                        sin_valor.append(f"{hoja.title}!{celda.coordinate}")
    if sin_valor:
        log(f"⚠️ {len(sin_valor)} fórmula(s) sin valor calculado quedaron vacías en el Excel ajustado "
            f"(abre y guarda el Formato 3 en Excel para recalcularlas): {', '.join(sin_valor[:10])}")

    for fila, v in nuevos_valores.items():
        ws.cell(row=fila, column=j_valor + 1).value = v
    for fila, iv in nuevos_iva.items():
        ws.cell(row=fila, column=j_iva + 1).value = iv

    base, _ = os.path.splitext(excel_path)
    salida = f"{base}_costo_objetivo.xlsx"
    try:
        wb.save(salida)
    except PermissionError:
        raise RuntimeError(f"No se pudo guardar '{salida}'. Ciérralo si está abierto en Excel y vuelve a intentar.")

    log(f"Nuevo total sin IVA: $ {sum(nuevos_valores.values()):,.0f} | Nuevo IVA total: $ {sum(nuevos_iva.values()):,.0f}")
    log(f"📄 Excel ajustado guardado en: {salida}")
    return salida

def run_automation(excel_path: str, sheet_name: str, radicado: str, start_idx: int,
                    pdf_dir: str, usuario: str = None, password: str = None,
                    log=print, keep_browser_open: bool = True, maestro_path: str = None,
                    costo_objetivo: int = None):
    """
    Ejecuta la autocarga completa. Pensada para ser invocada tanto desde la
    GUI como desde el bloque __main__ (modo consola).

    log: función que recibe un string (por defecto print). La GUI le pasa
    una función que escribe en el cuadro de texto de la interfaz.

    costo_objetivo: valor total deseado SIN IVA (opcional). Si se da, se genera
    el Excel ajustado (ver generar_excel_costo_objetivo) y se carga desde él.

    Devuelve True si terminó sin errores, o lanza una excepción si algo falló
    (la GUI captura la excepción y la muestra al usuario).
    """
    usuario = usuario if usuario else USUARIO
    password = password if password else PASSWORD

    if costo_objetivo:
        log("Aplicando costo objetivo a los precios del Formato 3...")
        excel_path = generar_excel_costo_objetivo(excel_path, sheet_name, costo_objetivo, log=log)

    log("Leyendo archivo Excel...")
    df = load_clean_excel(excel_path, sheet_name)

    if costo_objetivo:
        total_cargar = sum(int(fmt_money(get_col(r.to_dict(), *COLS_VALOR_SIN_IVA)) or 0) for _, r in df.iterrows())
        if total_cargar != costo_objetivo:
            log(f"⚠️ El total sin IVA que se va a cargar (${total_cargar:,.0f}) difiere del costo objetivo "
                f"(${costo_objetivo:,.0f}). Revisa el Excel ajustado.")
        else:
            log(f"✅ Total sin IVA a cargar verificado: ${total_cargar:,.0f}")

    log("Columnas detectadas en Excel:")
    for c in df.columns:
        log(f" - {c}")

    log(f"Equipos detectados emparejados listos para cargar: {len(df)}")

    # Validación previa: Bizagi rechaza el guardado si falta la Unidad de medida
    # o la Cantidad, así que se revisa ANTES de abrir el navegador (y no a mitad
    # de la carga).
    faltantes = []
    for i, row in df.iterrows():
        if i + 1 < start_idx:
            continue
        r = row.to_dict()
        campos = [nombre for nombre, valor in (("Unidad de Medida", s(get_col(r, *COLS_UNIDAD))),
                                               ("Cantidad", fmt_number(get_col(r, "Cantidad"))))
                  if not valor]
        if campos:
            faltantes.append(
                f"  - Ítem de carga #{i + 1} (columna Ítem = {fmt_number(get_col(r, 'Ítem', 'Item')) or '?'}) | "
                f"Odoo ID {normalizar_odoo_id(get_col(r, 'Odoo ID', 'Odoo', 'ID Odoo')) or '(sin ID)'} | "
                f"{s(get_col(r, 'Modelo / Referencia', 'Modelo', 'Referencia'))[:60]} | "
                f"falta: {', '.join(campos)}"
            )
    if faltantes:
        log(f"❌ {len(faltantes)} ítem(s) con campos obligatorios vacíos (Bizagi los exige):")
        for linea in faltantes:
            log(linea)
        log("Completa esos campos en el Formato 3 y vuelve a ejecutar. No se cargó nada.")
        raise RuntimeError(f"{len(faltantes)} ítem(s) con Unidad de Medida o Cantidad vacía. Revisa el log.")

    maestro_path = maestro_path or MAESTRO_DEFAULT
    log(f"Cargando maestro de Odoo ID: {maestro_path}")
    maestro_odoo = cargar_maestro_odoo(maestro_path, log=log)
    log(f"Maestro cargado: {len(maestro_odoo)} Odoo ID(s).")

    playwright_ctx = sync_playwright().start()
    browser = None
    # Chromium de Playwright; si no esta descargado, usar Edge o Chrome del PC.
    for channel in (None, "msedge", "chrome"):
        try:
            browser = playwright_ctx.chromium.launch(headless=False, slow_mo=SLOW_MO_MS, channel=channel)
            break
        except Exception as e:
            log(f"No se pudo abrir navegador ({channel or 'chromium'}): {e}")
    if browser is None:
        playwright_ctx.stop()
        raise RuntimeError("No se encontro ningun navegador (Chromium, Edge o Chrome).")
    page = browser.new_page()

    try:
        log("Abriendo plataforma...")
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        set_zoom(page, 0.8)

        log("Login...")
        do_login(page, usuario, password)
        try:
            page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            log("     ⚠️ networkidle no se alcanzó tras el login, usando espera de respaldo.")
            page.wait_for_timeout(3000)

        log("Entrando a Inbox...")
        open_inbox(page)

        log(f"Buscando radicado: {radicado}")
        open_case_by_radicado(page, radicado)

        log("Abriendo pestaña Información de equipos...")
        if not open_tab_info_equipos(page):
            log("❌ No pude abrir la pestaña 'Información de equipos'.")
            log("El navegador queda abierto para que puedas revisar manualmente.")
            raise RuntimeError("No se pudo abrir la pestaña 'Información de equipos'.")

        log("✅ Pestaña abierta. Iniciando carga...")

        report_data = []  # reporte de auditoría consolidado (ver guardar_reporte_auditoria)

        for i, row in df.iterrows():
            item_num = i + 1
            if item_num < start_idx:
                log(f"Skipping item {item_num} (start_idx es {start_idx})")
                continue

            row_dict = row.to_dict()
            nombre_dbg = s(get_col(row_dict, "Nombre del Elemento", "Nombre del elemento", "Nombre Elemento", "Nombre"))
            log(f"[{item_num}/{len(df)}] Cargando: {nombre_dbg}")

            estado_item, error_item, estado_soporte, lleva_certificado = procesar_item_con_reintentos(
                page, row_dict, i, item_num, pdf_dir, maestro_odoo, log, max_intentos=2
            )

            report_data.append({
                "Ítem": item_num,
                "Nombre del equipo": nombre_dbg,
                "Odoo ID": normalizar_odoo_id(get_col(row_dict, "Odoo ID", "Odoo", "ID Odoo")) or "",
                "Estado": estado_item,
                "Error": error_item or "",
                "Estado Soporte": ESTADO_SOPORTE_TEXTOS.get(estado_soporte, estado_soporte or ""),
                "Certificado Adjunto": "Sí" if lleva_certificado else "No",
            })

            if estado_item == "Error fatal":
                guardar_reporte_auditoria(report_data, excel_path, log)
                log(f"Para continuar, vuelva a ejecutar con ítem inicial = {item_num} "
                    f"(tras verificar en Bizagi qué ítems quedaron guardados).")
                raise RuntimeError(f"Ítem {item_num} falló tras reintentos: {error_item}")

        # Reporte de auditoría
        guardar_reporte_auditoria(report_data, excel_path, log)

        log("✅ Carga finalizada.")
        log("El navegador queda abierto para que puedas revisar el resultado.")
        return True

    except Exception:
        log("El navegador queda abierto para que puedas revisar manualmente.")
        raise
    finally:
        if not keep_browser_open:
            try:
                browser.close()
            finally:
                playwright_ctx.stop()


def get_interactive_config():
    # 1. Buscar archivos excel en BASE_DIR
    base_dir = BASE_DIR
    if not os.path.exists(base_dir):
        print(f"Error: La ruta base {base_dir} no existe.")
        base_dir = os.getcwd()
        print(f"Usando el directorio actual: {base_dir}")

    # El maestro de Odoo ID no es un Excel de proyecto: no se ofrece para cargar.
    files = [f for f in os.listdir(base_dir) if f.lower().endswith(".xlsx") and not f.startswith("~$")
             and os.path.join(base_dir, f) != MAESTRO_DEFAULT]
    if not files:
        print(f"No se encontraron archivos Excel (.xlsx) en {base_dir}")
        excel_path = input("Por favor ingresa la ruta completa al archivo Excel: ").strip()
    else:
        print("\n=== ARCHIVOS EXCEL DISPONIBLES ===")
        for idx, f in enumerate(files, 1):
            print(f" {idx}) {f}")
        
        while True:
            sel = input(f"Selecciona el archivo Excel (1-{len(files)}) [Default: 1]: ").strip()
            if not sel:
                excel_path = os.path.join(base_dir, files[0])
                break
            try:
                sel_idx = int(sel) - 1
                if 0 <= sel_idx < len(files):
                    excel_path = os.path.join(base_dir, files[sel_idx])
                    break
            except ValueError:
                pass
            print("Selección inválida.")

    print(f"-> Archivo seleccionado: {excel_path}")

    # 2. Seleccionar pestaña
    sheet_name = "FORMATO 3"
    try:
        xl = pd.ExcelFile(excel_path)
        sheets = xl.sheet_names
        # Buscar algo similar a FORMATO 3
        matching_sheets = [s for s in sheets if "formato 3" in s.lower()]
        if matching_sheets:
            sheet_name = matching_sheets[0]
        else:
            print("\n=== PESTAÑAS DISPONIBLES ===")
            for idx, s in enumerate(sheets, 1):
                print(f" {idx}) {s}")
            while True:
                sel_sh = input(f"Selecciona la pestaña (1-{len(sheets)}) [Default: 1]: ").strip()
                if not sel_sh:
                    sheet_name = sheets[0]
                    break
                try:
                    sel_sh_idx = int(sel_sh) - 1
                    if 0 <= sel_sh_idx < len(sheets):
                        sheet_name = sheets[sel_sh_idx]
                        break
                except ValueError:
                    pass
                print("Selección inválida.")
    except Exception as e:
        print(f"No se pudieron leer las pestañas del Excel ({e}). Se usará '{sheet_name}'.")

    print(f"-> Pestaña seleccionada: {sheet_name}")

    # 3. Solicitar Radicado
    # Sin valor por defecto: un radicado equivocado carga los equipos en otro caso.
    radicado = ""
    while not radicado:
        radicado = input("\nIngresa el RADICADO: ").strip()
    print(f"-> Radicado: {radicado}")

    # 4. Solicitar Start Item Index
    while True:
        idx_str = input("\nIngresa el número de ítem por el cual empezar (1-based) [Default: 1]: ").strip()
        if not idx_str:
            start_idx = 1
            break
        try:
            start_idx = int(idx_str)
            if start_idx >= 1:
                break
        except ValueError:
            pass
        print("Ingresa un número entero válido mayor o igual a 1.")
    print(f"-> Comenzará desde el ítem: {start_idx}")

    return excel_path, sheet_name, radicado, start_idx


if __name__ == "__main__":
    print("Iniciando configuración interactiva...")
    EXCEL_FILE, SHEET, RADICADO, START_ITEM_INDEX = get_interactive_config()

    try:
        run_automation(
            excel_path=EXCEL_FILE,
            sheet_name=SHEET,
            radicado=RADICADO,
            start_idx=START_ITEM_INDEX,
            pdf_dir=PDF_DIR,
        )
    except Exception as e:
        print("ERROR:", e)

    input("ENTER para cerrar esta ventana (el navegador se queda abierto para revisar)...")