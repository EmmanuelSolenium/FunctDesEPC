import sys
import os

# Ruta absoluta del archivo actual (doc_filler.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta raíz del proyecto (sube un nivel desde /autodocs)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Añadir al PYTHONPATH dinámicamente
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Ahora este import SIEMPRE funciona
from autodocs.funciones_docs import *

# ==============================
# LIBRERÍAS GOOGLE (Drive, Docs, Sheets)
# ==============================
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Opcional (si usarás autenticación interactiva en lugar de service account)
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle

# ==============================
# MANEJO DE DATOS (JSON, CSV, EXCEL)
# ==============================
import json
import csv
import pandas as pd

# ==============================
# MANEJO DE ARCHIVOS Y UTILIDADES
# ==============================
import io
import base64
from datetime import datetime


# ==============================
# CONFIGURACIÓN DE SCOPES (permisos API)
# ==============================
SCOPES = [
    'https://www.googleapis.com/auth/documents',   # Google Docs
    'https://www.googleapis.com/auth/drive',       # Google Drive
    'https://www.googleapis.com/auth/spreadsheets' # Google Sheets
]

# ==============================
# AUTENTICACIÓN (SERVICE ACCOUNT)
# ==============================
def autenticar_servicio(ruta_credenciales: str):
    """
    Autenticación usando Service Account.
    Retorna servicios de Google Docs, Drive y Sheets.
    """
    creds = service_account.Credentials.from_service_account_file(
        ruta_credenciales,
        scopes=SCOPES
    )

    docs_service = build('docs', 'v1', credentials=creds)
    drive_service = build('drive', 'v3', credentials=creds)
    sheets_service = build('sheets', 'v4', credentials=creds)

    return docs_service, drive_service, sheets_service