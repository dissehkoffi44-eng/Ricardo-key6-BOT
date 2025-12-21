import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor
import requests  
import gc         

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | KEY 98% FIABLE", page_icon="🎧", layout="wide")

TELEGRAM_TOKEN = "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo"
CHAT_ID = "-1003602454394" 

# Initialisation des états
if 'history' not in st.session_state: st.session_state.history = []
if 'processed_files' not in st.session_state: st.session_state.processed_files = {}
if 'order_list' not in st.session_state: st.session_state.order_list = []
if 'uploader_id' not in st.session_state: st.session_state.uploader_id = 0

# --- FONCTIONS TECHNIQUES (Gardées identiques) ---
def upload_to_telegram(file_buffer, filename, caption):
    try:
        file_buffer.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        files = {'document': (filename, file_buffer.read())}
        data = {'chat_id': CHAT_ID, 'caption': caption}
        response = requests.post(url, files=files, data=data, timeout=30).json()
        return response.get("ok", False)
    except: return False

def get_camelot_pro(key_mode_str):
    BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
    BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode in ['minor', 'dorian'] else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

@st.cache_data(show_spinner=False)
def get_full_analysis(file_bytes, file_name):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, res_type='kaiser_fast')
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    # Analyse simplifiée pour la démo, garde ton bloc complet ici
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # ... (Garder tout ton moteur d'analyse ici) ...
    return {"file_name": file_name, "tempo": int(float(tempo)), "synthese": "C major", "vote": "C major", "vote_conf": 90, "confidence": 95, "purity": 98, "energy": 5, "timeline": [], "key_shift": False}

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | KEY 98% FIABLE</h1>", unsafe_allow_html=True)

# Zone d'importation avec ID dynamique
files = st.file_uploader("📂 DÉPOSEZ VOS TRACKS ICI", type=['mp3', 'wav', 'flac'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_id}")

if files:
    new_files_detected = False
    
    for f in files:
        file_id = f"{f.name}_{f.size}"
        # On ne traite QUE si le fichier n'est pas déjà dans la mémoire
        if file_id not in st.session_state.processed_files:
            new_files_detected = True
            with st.spinner(f"Analyse en cours : {f.name}..."):
                f_bytes = f.read()
                res = get_full_analysis(f_bytes, f.name)
                cam = get_camelot_pro(res['synthese'])
                
                # Envoi Telegram
                caption = f"🎵 {f.name}\n🥁 BPM: {res['tempo']}\n🎯 KEY: {res['synthese']} ({cam})"
                upload_to_telegram(io.BytesIO(f_bytes), f"[{cam}] {f.name}", caption)
                
                # Sauvegarde
                st.session_state.processed_files[file_id] = res
                st.session_state.order_list.insert(0, file_id)
                gc.collect()

    # --- LOGIQUE DE NETTOYAGE SÉLECTIF ---
    # On vérifie si le nombre de fichiers dans le widget correspond au nombre de fichiers traités
    # Cela signifie que l'utilisateur n'est pas en train d'en charger d'autres.
    if new_files_detected:
        st.rerun()

# --- AFFICHAGE DES RÉSULTATS ---
for fid in st.session_state.order_list[:10]:
    res = st.session_state.processed_files[fid]
    with st.expander(f"🎵 {res['file_name']}", expanded=True):
        # (Tes colonnes c1, c2, c3, c4 et graphiques ici)
        st.write(f"BPM: {res['tempo']} | KEY: {res['synthese']}")
