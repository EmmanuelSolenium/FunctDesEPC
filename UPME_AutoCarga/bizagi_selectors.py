"""
Selectores de la interfaz de Bizagi usados por la autocarga UPME.
Si Bizagi cambia su UI (IDs, clases, XPaths), este es el ÚNICO lugar que
debería requerir edición -- la lógica de negocio en upme_autocarga_corregido.py
no debería tocarse.
"""

SELECTORS = {
    # Botón "+" para adicionar un nuevo equipo (capturado con Chrome Recorder)
    "add_plus_xpath": '//*[@id="mp_IncentivosFNCE_idmInformacionsolicitud_xEquipos"]/div/div[2]/div[3]/table/tbody/tr/th[1]/div/div/ul/li[1]/div',

    # Ícono de clip para adjuntar soporte técnico
    "attach_clip_css": 'span.ui-icon.upload-file',
    "attach_clip_container_css": 'div[data-render-xpath="fSoporte"] span.ui-icon.upload-file',

    # Input file oculto que Bizagi usa para inyectar el archivo
    "attach_file_input_css": 'input#file[type="file"]',

    # Botón "Subir" dentro del modal de adjuntos
    "attach_upload_button_xpath": "xpath=//span[contains(@class, 'ui-button-text') and text()='Subir']",
    "attach_upload_button_fallback_css": 'button:has-text("Subir"), span.ui-button-text:has-text("Subir")',

    # Textos de botones de guardado del formulario de equipo
    "save_button_texts": ["Guardar", "Aceptar"],

    # Campos del formulario "Adicionar equipo" por su data-render-xpath nativo de Bizagi
    "field_xpaths": {
        "nombre": "kpElementoFNCE",
        "marca": "sMarca",
        "subpartida": "sSubpartidaarancelaria",
        "unidad": "sUnidaddemedida",
        "fabricante": "sFabricante",
        "funcion": "sFuncion",
        "iva": "cValorIVAenCOP",
        "modelo": "sModeloReferencia",
        "cantidad": "iCantidad",
        "normas": "sNormastecnicas",
        "proveedor": "sProveedor",
        "valor_sin_iva": "cValortotalenCOPsinIV",
    },

    # xpaths de campos que llevan máscara numérica/monetaria (requieren tecleo char-by-char)
    "numeric_field_xpaths": ["cValorIVAenCOP", "cValortotalenCOPsinIV", "cValortotalenCOPsinIVA", "iCantidad"],

    # Textos candidatos para localizar la pestaña "Información de equipos"
    "tab_equipos_texts": ["Información de equipos", "Informacion de equipos", "Equipos"],

    # Textos candidatos para localizar el formulario de adicionar equipo ya abierto
    "form_adicionar_texts": ["Adicionar", "Adicionar equipo", "Agregar equipo"],
    "form_adicionar_label_texts": ["Nombre del Elemento", "Nombre"],

    # Fragmentos de URL usados para detectar respuestas de red relevantes (AJUSTAR tras inspeccionar
    # con DevTools -> pestaña Network las peticiones reales que dispara Bizagi al guardar/adjuntar).
    # Ver sección 6 de este plan para instrucciones de cómo obtenerlos.
    "network_hints": {
        "save_case": "SaveCase",       # placeholder, confirmar con DevTools
        "upload_file": "UploadFile",   # placeholder, confirmar con DevTools
        "add_case_relation": "AddCaseRelation",  # placeholder, confirmar con DevTools
    },
}
