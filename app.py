import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests

# --- CONFIGURATION & SECRETS ---
st.set_page_config(page_title="Audio Perception AI", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

def get_camelot_key(key, tone):
    # Rappel : F# Minor = 11A (Votre règle personnalisée)
    camelot_map = {
        'C Major': '8B', 'G Major': '9B', 'D Major': '10B', 'A Major': '11B', 'E Major': '12B', 'B Major': '1B',
        'F# Major': '2B', 'C# Major': '3B', 'G# Major': '4B', 'D# Major': '5B', 'A# Major': '6B', 'F Major': '7B',
        'A Minor': '8A', 'E Minor': '9A', 'B Minor': '10A', 'F# Minor': '11A', 'C# Minor': '12A', 'G# Minor': '1A',
        'D# Minor': '2A', 'A# Minor': '3A', 'F Minor': '4A', 'C Minor': '5A', 'G Minor': '6A', 'D Minor': '7A'
    }
    return camelot_map.get(f"{key} {tone}", "Inconnu")

@st.cache_data
def analyze_human_perception(file_path):
    # 1. Chargement avec un taux d'échantillonnage standard
    y, sr = librosa.load(file_path, sr=22050)

    # 2. PRÉ-EMPHASE : Simule la sensibilité de l'oreille humaine aux hautes fréquences
    y_filt = librosa.effects.preemphasis(y)

    # 3. CHROMA CQT avec haute résolution (3 bins par demi-ton pour la précision du pitch)
    chroma = librosa.feature.chroma_cqt(y=y_filt, sr=sr, bins_per_octave=36)
    
    # 4. AGGREGATION TEMPORELLE (Médiane au lieu de Moyenne)
    # La médiane réduit l'impact des bruits impulsionnels (percussions)
    chroma_vals = np.median(chroma, axis=1)

    # 5. PROFILS PSYCHOACOUSTIQUES (Temperley - mieux pour l'oreille moderne)
    maj_profile = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
    min_profile = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_score = -1
    final_key = ""
    final_tone = ""

    for i in range(12):
        # Rotation des profils pour tester chaque tonique
        p_maj = np.roll(maj_profile, i)
        p_min = np.roll(min_profile, i)
        
        # Corrélation de Pearson
        score_maj = np.corrcoef(chroma_vals, p_maj)[0, 1]
        score_min = np.corrcoef(chroma_vals, p_min)[0, 1]
        
        if score_maj > best_score:
            best_score = score_maj
            final_key, final_tone = notes[i], "Major"
        if score_min > best_score:
            best_score = score_min
            final_key, final_tone = notes[i], "Minor"

    return chroma_vals, final_key, final_tone

# --- UI STREAMLIT ---
st.title("🧠 Perception Auditive Avancée")
st.info("Cette version utilise le filtrage de pré-emphase et la corrélation de Temperley pour simuler l'oreille humaine.")

uploaded_file = st.file_uploader("Fichier Audio", type=["mp3", "wav"])

if uploaded_file:
    with open("temp.audio", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyse psychoacoustique..."):
        chroma_vals, key, tone = analyze_human_perception("temp.audio")
        camelot = get_camelot_key(key, tone)
        
        # Affichage
        st.subheader(f"Résultat : {key} {tone} ({camelot})")
        
        # Radar Chart pour visualiser la "force" de chaque note (très parlant pour l'oreille)
        fig = go.Figure(go.Scatterpolar(
            r=chroma_vals,
            theta=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
            fill='toself',
            line_color='teal'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=False)
        st.plotly_chart(fig)

        # Envoi Telegram (optionnel)
        if TELEGRAM_TOKEN and CHAT_ID:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                          json={"chat_id": CHAT_ID, "text": f"Analyse : {key} {tone} / {camelot}"})
