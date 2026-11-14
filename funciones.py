"""Utilidades comunes: aviso por Telegram (token y chat_id en variables de entorno) y fijación de semillas."""
import os
import random

import numpy as np
import requests
import tensorflow as tf


def enviar_telegram(mensaje):
    token = os.environ.get('TELEGRAM_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    datos = {
        "chat_id": chat_id,
        "text": mensaje
    }
    
    respuesta = requests.post(url, data=datos)
    
    if respuesta.status_code == 200:
        print("¡Notificación enviada con éxito!")
    else:
        print("Error al enviar:", respuesta.text)


def set_seed(seed):
  # Fijar semillas
  os.environ['PYTHONHASHSEED'] = str(seed)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
  np.random.seed(seed)
  random.seed(seed)
  tf.random.set_seed(seed)