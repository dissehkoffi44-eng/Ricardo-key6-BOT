import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="Audio Perception AI - Pro", layout="wide")

# Récupération sécurisée des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

def get_camelot_key(key, tone):
    """Convertit la tonalité en code Camelot (Inclut F# Minor = 11A)."""
    camelot_map = {
        'C Major': '8B', 'G Major': '9B', 'D Major': '10B', 'A Major': '11B', 'E Major': '12B', 'B Major': '1B',
        'F# Major': '2B', 'C# Major': '3B', 'G# Major': '4B', 'D# Major': '5B', 'A# Major': '6B', 'F Major': '7B',
        'A Minor': '8A', 'E Minor': '9A', 'B Minor': '10A', 'F# Minor': '11A', 'C# Minor': '12A', 'G# Minor': '1A',
        'D# Minor': '2A', 'A# Minor': '3A', 'F Minor': '4A', 'C Minor': '5A', 'G Minor': '6A', 'D Minor': '7A'
    }
    return camelot_map.get(f"{key} {tone}", "Inconnu")

def generate_verification_chord(key, tone, duration=2.5, sr=22050):
    """Génère un accord de piano synthétique (sinusoïdal) pour vérification."""
    notes_freq = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 
        'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }
    
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_idx = chromatic_scale.index(key)
    
    # Construction de l'accord (Tierce majeure ou mineure + Quinte)
    if tone == "Major":
        intervals = [0, 4, 7] 
    else:
        intervals = [0, 3, 7]
        
    t = np.linspace(0, duration, int(sr * duration))
    chord_wave = np.zeros_like(t)
    
    for i in intervals:
        note_name = chromatic_scale[(root_idx + i) % 12]
        freq = notes_freq[note_name]
        # On ajoute une octave pour la clarté si besoin, ou on reste sur la base
        chord_wave += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Appliquer une enveloppe simple pour éviter les clics (fade out)
    fade_out = np.linspace(1, 0, len(t))
    chord_wave *= fade_out
    
    return chord_wave

def send_telegram_message(message):
    """Envoie les résultats sur Telegram."""
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload)
        except Exception as e:
            st.error(f"Erreur Telegram : {e}")

@st.cache_data(show_spinner=False)
def analyze_human_perception(file_path, original_filename):
    """Analyse avancée imitant la perception humaine."""
    y, sr = librosa.load(file_path, sr=22050)
    y = librosa.effects.preemphasis(y)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, bins_per_octave=24)
    chroma_vals = np.mean(chroma**2, axis=1)
    
    if np.max(chroma_vals) > 0:
        chroma_vals = chroma_vals / np.max(chroma_vals)

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
st.title("🧠 Perception Auditive AI (Multi-Profils)")
st.markdown("---")

uploaded_file = st.file_uploader("Glissez votre fichier audio ici", type=["mp3", "wav", "flac"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.audio(uploaded_file)
    
    with st.spinner(f"Analyse de : {uploaded_file.name}..."):
        try:
            chroma_vals, key, tone = analyze_human_perception(tmp_path, uploaded_file.name)
            camelot = get_camelot_key(key, tone)
            result_text = f"{key} {tone}"

            col1, col2, col3 = st.columns([1, 1, 1])
            col1.metric("Tonalité Détectée", result_text)
            col2.metric("Code Camelot", camelot)
            
            # --- SECTION VÉRIFICATION SONORE ---
            with col3:
                st.write("🎹 **Vérification (Accord)**")
                chord_audio = generate_verification_chord(key, tone)
                st.audio(chord_audio, format="audio/wav", sample_rate=22050)

            # Graphique Radar
            categories = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            fig = go.Figure(data=go.Scatterpolar(r=chroma_vals, theta=categories, fill='toself', line_color='#00FFAA'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            send_telegram_message(f"🎵 *Analyse*\n*Fichier :* {uploaded_file.name}\n*Résultat :* {result_text}\n*Camelot :* {camelot}")
            st.success("Analyse terminée.")

        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
