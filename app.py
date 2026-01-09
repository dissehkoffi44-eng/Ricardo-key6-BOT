import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="DJ Ricardo's Pro Ear", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

def get_camelot_key(key, tone):
    camelot_map = {
        'C Major': '8B', 'G Major': '9B', 'D Major': '10B', 'A Major': '11B', 'E Major': '12B', 'B Major': '1B',
        'F# Major': '2B', 'C# Major': '3B', 'G# Major': '4B', 'D# Major': '5B', 'A# Major': '6B', 'F Major': '7B',
        'A Minor': '8A', 'E Minor': '9A', 'B Minor': '10A', 'F# Minor': '11A', 'C# Minor': '12A', 'G# Minor': '1A',
        'D# Minor': '2A', 'A# Minor': '3A', 'F Minor': '4A', 'C Minor': '5A', 'G Minor': '6A', 'D Minor': '7A'
    }
    return camelot_map.get(f"{key} {tone}", "Inconnu")

@st.cache_data(show_spinner=False)
def analyze_pro_ear(file_path):
    """Analyse haute précision simulant l'oreille humaine."""
    # 1. Chargement avec un taux d'échantillonnage optimal
    y, sr = librosa.load(file_path, sr=22050)

    # 2. Séparation Harmonique/Percussive (HPSS)
    # On ne garde que la partie harmonique pour éviter que les transitoires (drums) polluent la détection
    y_harmonic = librosa.effects.hpss(y)[0]

    # 3. Application d'une pondération A (A-weighting)
    # Simule la sensibilité de l'oreille selon les fréquences
    fft_size = 2048
    weights = librosa.A_weighting(librosa.fft_frequencies(sr=sr, n_fft=fft_size))
    # Note: On utilise principalement CQT car il est aligné sur les demi-tons musicaux
    
    # 4. Chroma CQT avec résolution accrue
    # On utilise 36 bins par octave pour une meilleure précision d'accordage
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)
    
    # Médiane temporelle pour ignorer les notes accidentelles (outliers)
    chroma_vals = np.median(chroma_cqt, axis=1)
    
    # Normalisation
    if np.max(chroma_vals) > 0:
        chroma_vals = chroma_vals / np.max(chroma_vals)

    # 5. Profils de Temperley (plus précis pour la Pop/EDM/Jazz)
    maj_profile = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
    min_profile = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_score = -1
    final_key, final_tone = "", ""

    for i in range(12):
        p_maj, p_min = np.roll(maj_profile, i), np.roll(min_profile, i)
        # Utilisation de la corrélation de Pearson
        score_maj = np.corrcoef(chroma_vals, p_maj)[0, 1]
        score_min = np.corrcoef(chroma_vals, p_min)[0, 1]
        
        if score_maj > best_score:
            best_score, final_key, final_tone = score_maj, notes[i], "Major"
        if score_min > best_score:
            best_score, final_key, final_tone = score_min, notes[i], "Minor"

    return chroma_vals, final_key, final_tone

# --- UI STREAMLIT ---
st.title("DJ Ricardo's Pro Ear 🎧")
st.markdown("Algorithme avec séparation harmonique (HPSS) et pondération psychoacoustique.")

uploaded_file = st.file_uploader("Fichier audio", type=["mp3", "wav", "flac"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with st.spinner("Analyse harmonique profonde en cours..."):
        try:
            chroma_vals, key, tone = analyze_pro_ear(tmp_path)
            camelot = get_camelot_key(key, tone)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Tonalité", f"{key} {tone}")
            col2.metric("Code Camelot", camelot)
            col3.metric("Confiance", f"{int(np.max(chroma_vals)*100)}%")

            # Radar Chart
            fig = go.Figure(data=go.Scatterpolar(
                r=chroma_vals, theta=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
                fill='toself', line_color='#1DB954'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur : {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
