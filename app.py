import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

# Configuration
st.set_page_config(page_title="Analyseur de Tonalité Intégral", layout="wide")

# Paramètres Telegram (Récupérés depuis st.secrets)
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

def send_telegram_message(message):
    """Envoie un message texte au canal Telegram via le Bot."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        st.warning("Note : Telegram non configuré (TELEGRAM_TOKEN ou CHAT_ID manquants).")
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
    # F# Minor = 11A inclus
    camelot_map = {
        'C Major': '8B', 'G Major': '9B', 'D Major': '10B', 'A Major': '11B', 'E Major': '12B', 'B Major': '1B',
        'F# Major': '2B', 'C# Major': '3B', 'G# Major': '4B', 'D# Major': '5B', 'A# Major': '6B', 'F Major': '7B',
        'A Minor': '8A', 'E Minor': '9A', 'B Minor': '10A', 'F# Minor': '11A', 'C# Minor': '12A', 'G# Minor': '1A',
        'D# Minor': '2A', 'A# Minor': '3A', 'F Minor': '4A', 'C Minor': '5A', 'G Minor': '6A', 'D Minor': '7A'
    }
    return camelot_map.get(f"{key} {tone}", "Inconnu")

@st.cache_data
def analyze_full_audio(file_path):
    """Analyse l'intégralité du morceau avec corrélation croisée sur tous les tons."""
    target_sr = 22050 
    chroma_sum = np.zeros(12)
    count = 0
    
    # Lecture par flux pour économiser la RAM
    stream = librosa.stream(file_path, block_length=256, frame_length=2048, hop_length=512)
    
    for y_block in stream:
        # Utilisation de CQT pour une meilleure précision des notes
        chroma_block = librosa.feature.chroma_cqt(y=y_block, sr=target_sr)
        chroma_sum += np.sum(chroma_block, axis=1)
        count += chroma_block.shape[1]
    
    chroma_mean = chroma_sum / count
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Profils de Krumhansl-Schmuckler
    maj_prof = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_prof = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    best_corr = -1
    best_key = ""
    best_tone = ""

    # Test des 12 notes pour le Majeur et le Mineur
    for i in range(12):
        # On fait défiler les profils pour tester chaque note comme tonique
        corr_maj = np.corrcoef(chroma_mean, np.roll(maj_prof, i))[0, 1]
        corr_min = np.corrcoef(chroma_mean, np.roll(min_prof, i))[0, 1]
        
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = notes[i]
            best_tone = "Major"
            
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = notes[i]
            best_tone = "Minor"
            
    return chroma_mean, best_key, best_tone

# --- Interface Streamlit ---
st.title("👂 Perception Auditive : Analyse Intégrale")
st.markdown("Cette version compare l'empreinte harmonique complète pour une précision maximale.")

uploaded_file = st.file_uploader("Uploadez votre morceau", type=["mp3", "wav", "flac"])

if uploaded_file is not None:
    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    with st.spinner("Analyse harmonique en cours..."):
        try:
            chroma_vals, key, tone = analyze_full_audio("temp_audio")
            camelot = get_camelot_key(key, tone)
            
            # Affichage des résultats
            st.divider()
            c1, c2 = st.columns(2)
            result_key = f"{key} {tone}"
            
            with c1:
                st.markdown("### Tonalité Détectée")
                st.subheader(f"🎵 {result_key}")
            
            with c2:
                st.markdown("### Code Camelot")
                st.subheader(f"🎡 {camelot}")
            
            # Envoi automatique vers Telegram
            msg = f"🎵 *Nouvelle Analyse Audio*\n\n*Fichier :* {uploaded_file.name}\n*Tonalité :* {result_key}\n*Code Camelot :* {camelot}"
            send_telegram_message(msg)
            st.success("✅ Résultats envoyés sur Telegram !")
            
            # Graphique
            st.subheader("Empreinte Tonale Globale")
            fig = go.Figure(go.Bar(
                x=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
                y=chroma_vals,
                marker_color='rgb(158,202,225)',
                marker_line_color='rgb(8,48,107)',
                opacity=0.8
            ))
            fig.update_layout(xaxis_title="Notes", yaxis_title="Intensité")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur durant l'analyse : {e}")
