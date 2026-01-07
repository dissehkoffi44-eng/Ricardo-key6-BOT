import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="Audio Perception AI - Pro v2", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

def get_camelot_key(key, tone):
    """Convertit la tonalité en code Camelot (F# Minor = 11A)."""
    camelot_map = {
        'C Major': '8B', 'G Major': '9B', 'D Major': '10B', 'A Major': '11B', 'E Major': '12B', 'B Major': '1B',
        'F# Major': '2B', 'C# Major': '3B', 'G# Major': '4B', 'D# Major': '5B', 'A# Major': '6B', 'F Major': '7B',
        'A Minor': '8A', 'E Minor': '9A', 'B Minor': '10A', 'F# Minor': '11A', 'C# Minor': '12A', 'G# Minor': '1A',
        'D# Minor': '2A', 'A# Minor': '3A', 'F Minor': '4A', 'C Minor': '5A', 'G Minor': '6A', 'D Minor': '7A'
    }
    return camelot_map.get(f"{key} {tone}", "Inconnu")

def send_telegram_message(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try: requests.post(url, json=payload)
        except Exception as e: st.error(f"Erreur Telegram : {e}")

@st.cache_data(show_spinner=False)
def analyze_human_perception(file_path, original_filename):
    # 1. Chargement avec un échantillonnage suffisant
    y, sr = librosa.load(file_path, sr=22050)
    
    # 2. SÉPARATION HARMONIQUE (Crucial pour ignorer les drums de l'intro/outro)
    # On ne garde que la composante harmonique pour l'analyse de clé
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
    
    # 3. FILTRAGE PASSE-BAS (On se concentre sur les fondamentales < 600Hz)
    y_low = librosa.resample(y_harmonic, orig_sr=sr, target_sr=2000) 
    
    # 4. CHROMA CQT (Plus précis que le Chroma STFT pour la musique)
    # fmin fixé à C1 (environ 32Hz) pour bien capter les basses
    chroma = librosa.feature.chroma_cqt(y=y_low, sr=2000, hop_length=256, fmin=32.7)
    
    # Moyenne énergétique (Square root mean pour lisser)
    chroma_vals = np.mean(chroma**2, axis=1)
    if np.max(chroma_vals) > 0:
        chroma_vals = chroma_vals / np.max(chroma_vals)

    # Profils Krumhansl-Schmuckler (Poids des notes)
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_score = -1
    final_key, final_tone = "", ""

    for i in range(12):
        p_maj, p_min = np.roll(maj_profile, i), np.roll(min_profile, i)
        score_maj = np.corrcoef(chroma_vals, p_maj)[0, 1]
        score_min = np.corrcoef(chroma_vals, p_min)[0, 1]
        
        if score_maj > best_score:
            best_score, final_key, final_tone = score_maj, notes[i], "Major"
        if score_min > best_score:
            best_score, final_key, final_tone = score_min, notes[i], "Minor"

    return chroma_vals, final_key, final_tone

# --- INTERFACE ---
st.title("🧠 Audio Perception AI (Optimisé HSS)")
st.info("Cette version utilise la séparation harmonique pour isoler la mélodie des percussions.")

uploaded_file = st.file_uploader("Fichier audio", type=["mp3", "wav", "flac"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.audio(uploaded_file)
    
    with st.spinner("Analyse en cours..."):
        try:
            chroma_vals, key, tone = analyze_human_perception(tmp_path, uploaded_file.name)
            camelot = get_camelot_key(key, tone)
            result_text = f"{key} {tone}"

            col1, col2 = st.columns(2)
            col1.metric("Tonalité", result_text)
            col2.metric("Code Camelot", camelot)

            # Radar Plot
            categories = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            fig = go.Figure(data=go.Scatterpolar(r=chroma_vals, theta=categories, fill='toself', line_color='#00FFAA'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", title="Empreinte Harmonique")
            st.plotly_chart(fig, use_container_width=True)

            send_telegram_message(f"🎵 *Analyse*\n*Fichier :* {uploaded_file.name}\n*Résultat :* {result_text}\n*Camelot :* {camelot}")

        except Exception as e:
            st.error(f"Erreur : {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
