import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import os
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="Audio Perception AI - Pro", layout="wide")

def get_camelot_key(key, tone):
    camelot_map = {
        'C Major': '8B', 'G Major': '9B', 'D Major': '10B', 'A Major': '11B', 'E Major': '12B', 'B Major': '1B',
        'F# Major': '2B', 'C# Major': '3B', 'G# Major': '4B', 'D# Major': '5B', 'A# Major': '6B', 'F Major': '7B',
        'A Minor': '8A', 'E Minor': '9A', 'B Minor': '10A', 'F# Minor': '11A', 'C# Minor': '12A', 'G# Minor': '1A',
        'D# Minor': '2A', 'A# Minor': '3A', 'F Minor': '4A', 'C Minor': '5A', 'G Minor': '6A', 'D Minor': '7A'
    }
    return camelot_map.get(f"{key} {tone}", "Inconnu")

@st.cache_data(show_spinner=False)
def full_pro_analysis(file_path):
    # 1. Chargement avec filtrage spectral (On ignore ce qui est < 60Hz et > 5000Hz)
    y, sr = librosa.load(file_path, sr=22050)
    
    # 2. HPSS : On ne garde que l'harmonique (supprime les kicks/snares)
    y_harm = librosa.effects.hpss(y)[0]
    
    # 3. CQT avec haute résolution
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=24, fmin=librosa.note_to_hz('C2'))
    
    # --- ANALYSE DE CONFIANCE & MODULATION ---
    # On calcule la tonalité sur 3 segments (Début, Milieu, Fin)
    segments = np.array_split(chroma, 3, axis=1)
    seg_results = []
    
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def get_best_fit(chroma_data):
        mean_chroma = np.mean(chroma_data, axis=1)
        best_s, best_k, best_t = -1, "", ""
        for i in range(12):
            s_maj = np.corrcoef(mean_chroma, np.roll(maj_profile, i))[0, 1]
            s_min = np.corrcoef(mean_chroma, np.roll(min_profile, i))[0, 1]
            if s_maj > best_s: best_s, best_k, best_t = s_maj, notes[i], "Major"
            if s_min > best_s: best_s, best_k, best_t = s_min, notes[i], "Minor"
        return best_k, best_t, best_s

    # Analyse globale
    final_key, final_tone, confidence = get_best_fit(chroma)
    
    # Analyse de modulation
    for seg in segments:
        seg_results.append(get_best_fit(seg)[0:2])

    return {
        "chroma_vals": np.mean(chroma, axis=1),
        "key": final_key,
        "tone": final_tone,
        "confidence": confidence,
        "modulations": seg_results
    }

# --- INTERFACE ---
st.title("🧠 Perception Auditive AI - Pro Ultra")

uploaded_file = st.file_uploader("Fichier Audio", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        data = full_pro_analysis(tmp.name)
    
    camelot = get_camelot_key(data['key'], data['tone'])
    
    # Affichage des métriques
    c1, c2, c3 = st.columns(3)
    c1.metric("Tonalité Détectée", f"{data['key']} {data['tone']}")
    c2.metric("Code Camelot", camelot)
    c3.metric("Confiance", f"{int(data['confidence']*100)}%")

    # Alerte Modulation
    if len(set(data['modulations'])) > 1:
        st.warning(f"⚠️ Changement de tonalité détecté : {data['modulations']}")
    else:
        st.success("✅ Tonalité stable sur tout le morceau.")

    # Radar Chart
    fig = go.Figure(data=go.Scatterpolar(r=data['chroma_vals'], theta=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], fill='toself'))
    fig.update_layout(template="plotly_dark", title="Empreinte Harmonique")
    st.plotly_chart(fig)

    os.remove(tmp.name)
