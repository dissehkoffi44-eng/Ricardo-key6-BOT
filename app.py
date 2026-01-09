import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import tempfile
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="DJ Ricardo's Ultimate Ear V4", layout="wide")

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

def send_telegram_message(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception:
            pass

@st.cache_data(show_spinner=False)
def analyze_pro_ear_v4(file_path):
    # 1. Chargement du signal
    y, sr = librosa.load(file_path, sr=22050)
    
    # 2. Analyse du Tempo et des Beats (L'oreille rythmique)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    # 3. Correction du Tuning
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    
    # 4. Isolation Harmonique (On retire les drums pour ne pas polluer les notes)
    y_harmonic = librosa.effects.hpss(y, margin=3.0)[0]

    # 5. Chroma CQT Synchronisé sur les Beats
    # Au lieu d'analyser chaque milliseconde, on analyse la couleur sonore à chaque "temps"
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, tuning=tuning)
    
    # Synchronisation : On fait la moyenne du chroma entre chaque beat détecté
    chroma_sync = librosa.util.sync(chroma_cqt, beat_frames, aggregate=np.median)

    # 6. Analyse par segments de 32 temps (environ 8 mesures)
    window_size = 32
    keys_detected = []
    
    maj_profile = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
    min_profile = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for i in range(0, chroma_sync.shape[1], window_size):
        segment = chroma_sync[:, i:i+window_size]
        if segment.shape[1] < 4: continue
        
        chroma_vals = np.mean(segment, axis=1)
        
        best_segment_score = -1
        current_key, current_tone = "", ""
        
        for n in range(12):
            p_maj, p_min = np.roll(maj_profile, n), np.roll(min_profile, n)
            score_maj = np.corrcoef(chroma_vals, p_maj)[0, 1]
            score_min = np.corrcoef(chroma_vals, p_min)[0, 1]
            
            if score_maj > best_segment_score:
                best_segment_score, current_key, current_tone = score_maj, notes[n], "Major"
            if score_min > best_segment_score:
                best_segment_score, current_key, current_tone = score_min, notes[n], "Minor"
        
        keys_detected.append((current_key, current_tone))

    # Vote final
    most_common = Counter(keys_detected).most_common(1)[0][0]
    final_key, final_tone = most_common
    
    # Chroma global pour affichage
    global_chroma = np.median(chroma_sync, axis=1)
    if np.max(global_chroma) > 0: global_chroma /= np.max(global_chroma)

    return global_chroma, final_key, final_tone, tuning, tempo, keys_detected

# --- INTERFACE ---
st.title("DJ Ricardo's Ultimate Ear 💎 V4")
st.subheader("Analyse Synchronisée sur le Rythme (Beat-Matching Analysis)")

uploaded_file = st.file_uploader("Glissez un morceau", type=["mp3", "wav", "flac"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with st.spinner("Analyse du BPM et de la structure harmonique..."):
        try:
            chroma_vals, key, tone, tuning_val, tempo, history = analyze_pro_ear_v4(tmp_path)
            camelot = get_camelot_key(key, tone)

            # Métriques Principales
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tonalité", f"{key} {tone}")
            c2.metric("Camelot", camelot)
            c3.metric("Tempo (BPM)", f"{int(tempo)}")
            c4.metric("Tuning", f"{tuning_val:+.2f}c")

            # Timeline
            st.write("### ⏱ Évolution de la structure")
            timeline_str = " ➔ ".join([f"[{k}{'m' if t=='Minor' else ''}]" for k, t in history])
            st.info(timeline_str)

            # Radar Chart
            notes_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            fig = go.Figure(data=go.Scatterpolar(
                r=chroma_vals, theta=notes_labels, fill='toself', line_color='#00E676'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # Envoi Telegram
            msg = (f"🎼 *V4 Advanced Analysis*\n"
                   f"*Track:* {uploaded_file.name}\n"
                   f"*Key:* {key} {tone} ({camelot})\n"
                   f"*BPM:* {int(tempo)}\n"
                   f"*Tuning:* {tuning_val:.2f}c")
            send_telegram_message(msg)

        except Exception as e:
            st.error(f"Erreur : {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
