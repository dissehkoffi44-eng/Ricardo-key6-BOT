import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import requests

# Configuration
st.set_page_config(page_title="Analyseur de Tonalité Intégral", layout="wide")

# Paramètres Telegram (Récupérés depuis st.secrets)
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

def send_telegram_message(message):
    """Envoie un message texte au canal Telegram via le Bot."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        st.error("Erreur : TELEGRAM_TOKEN ou CHAT_ID non configurés dans les secrets Streamlit.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            st.error(f"Erreur Telegram ({response.status_code}) : {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de l'envoi Telegram : {e}")

def get_camelot_key(key, tone):
    # Inclusion de votre règle personnalisée : F# Minor = 11A
    camelot_map = {
        'C Major': '8B', 'G Major': '9B', 'D Major': '10B', 'A Major': '11B', 'E Major': '12B', 'B Major': '1B',
        'F# Major': '2B', 'C# Major': '3B', 'G# Major': '4B', 'D# Major': '5B', 'A# Major': '6B', 'F Major': '7B',
        'A Minor': '8A', 'E Minor': '9A', 'B Minor': '10A', 'F# Minor': '11A', 'C# Minor': '12A', 'G# Minor': '1A',
        'D# Minor': '2A', 'A# Minor': '3A', 'F Minor': '4A', 'C Minor': '5A', 'G Minor': '6A', 'D Minor': '7A'
    }
    return camelot_map.get(f"{key} {tone}", "Inconnu")

@st.cache_data
def analyze_full_audio(file_path):
    """Analyse l'intégralité du morceau par blocs pour préserver la RAM."""
    target_sr = 22050 
    
    chroma_sum = np.zeros(12)
    count = 0
    
    stream = librosa.stream(file_path, block_length=256, frame_length=2048, hop_length=512)
    
    for y_block in stream:
        chroma_block = librosa.feature.chroma_cqt(y=y_block, sr=target_sr)
        chroma_sum += np.sum(chroma_block, axis=1)
        count += chroma_block.shape[1]
    
    chroma_mean = chroma_sum / count
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = np.argmax(chroma_mean)
    
    maj_prof = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_prof = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    maj_corr = np.corrcoef(chroma_mean, np.roll(maj_prof, key_idx))[0, 1]
    min_corr = np.corrcoef(chroma_mean, np.roll(min_prof, key_idx))[0, 1]
    
    tone = "Major" if maj_corr > min_corr else "Minor"
    return chroma_mean, notes[key_idx], tone

st.title("👂 Perception Auditive : Analyse Intégrale")
st.markdown("Cette version analyse **100% de la durée du morceau** pour une précision maximale.")

uploaded_file = st.file_uploader("Uploadez votre morceau", type=["mp3", "wav", "flac"])

if uploaded_file is not None:
    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    with st.spinner("Analyse de l'intégralité du morceau en cours..."):
        try:
            chroma_vals, key, tone = analyze_full_audio("temp_audio")
            camelot = get_camelot_key(key, tone)
            
            # Affichage des résultats
            c1, c2 = st.columns(2)
            result_key = f"{key} {tone}"
            c1.metric("Tonalité Finale", result_key)
            c2.metric("Code Camelot", camelot)
            
            # Envoi automatique vers Telegram
            msg = f"🎵 *Nouvelle Analyse Audio*\n\n*Fichier :* {uploaded_file.name}\n*Tonalité :* {result_key}\n*Code Camelot :* {camelot}"
            send_telegram_message(msg)
            st.success("✅ Résultats envoyés sur Telegram !")
            
            # Graphique de l'énergie perçue sur tout le morceau
            st.subheader("Empreinte Tonale Globale")
            fig = go.Figure(go.Bar(
                x=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
                y=chroma_vals,
                marker_color='rgb(158,202,225)',
                marker_line_color='rgb(8,48,107)',
                opacity=0.8
            ))
            fig.update_layout(xaxis_title="Notes", yaxis_title="Intensité moyenne sur le morceau")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur durant l'analyse : {e}")
