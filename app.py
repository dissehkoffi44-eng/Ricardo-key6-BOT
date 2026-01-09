import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import io

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

def send_telegram_data(message, image_bytes=None):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    try:
        if image_bytes:
            files = {'photo': ('radar.png', image_bytes)}
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': CHAT_ID, 'caption': message, 'parse_mode': 'Markdown'}, files=files, timeout=15)
        else:
            requests.post(f"{base_url}/sendMessage", json={'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        st.error(f"Erreur Telegram : {e}")

def analyze_audio_optimized(file_buffer):
    # 1. Chargement
    y, sr = librosa.load(file_buffer, sr=22050)
    
    # 2. CORRECTION DU DÉSACCORDAGE (Auto-Tuning)
    # On estime le décalage par rapport au 440Hz
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    
    # 3. Pré-emphase
    y = librosa.effects.preemphasis(y)
    
    # 4. Extraction CQT avec prise en compte du décalage (tuning)
    # Le paramètre 'tuning' recale les fréquences sur les bonnes notes
    chroma = librosa.feature.chroma_cqt(
        y=y, 
        sr=sr, 
        hop_length=512, 
        bins_per_octave=24, 
        tuning=tuning
    )
    
    chroma_vals = np.mean(chroma**2, axis=1)
    if np.max(chroma_vals) > 0:
        chroma_vals = chroma_vals / np.max(chroma_vals)

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    best_score = -1
    key, tone = "", ""
    for i in range(12):
        p_maj, p_min = np.roll(maj_profile, i), np.roll(min_profile, i)
        s_maj = np.corrcoef(chroma_vals, p_maj)[0, 1]
        s_min = np.corrcoef(chroma_vals, p_min)[0, 1]
        if s_maj > best_score: best_score, key, tone = s_maj, notes[i], "Major"
        if s_min > best_score: best_score, key, tone = s_min, notes[i], "Minor"

    sorted_indices = np.argsort(chroma_vals)[::-1]
    top_notes = [(notes[i], round(chroma_vals[i]*100, 1)) for i in sorted_indices[:3]]
    
    return chroma_vals, key, tone, top_notes, tuning

# --- INTERFACE ---
st.title("DJ Ricardo's Pro Ear 🎧")
st.markdown("Analyse multi-fichiers avec **Auto-Tuning Correction**.")

uploaded_files = st.file_uploader("Glissez vos morceaux ici", type=["mp3", "wav", "flac"], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        with st.expander(f"📊 Résultats pour : {f.name}", expanded=True):
            try:
                chroma_vals, key, tone, top_notes, tuning = analyze_audio_optimized(f)
                camelot = get_camelot_key(key, tone)
                
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.metric("Tonalité", f"{key} {tone}")
                c2.metric("Code Camelot", camelot)
                
                # Affichage du décalage détecté
                tuning_info = f"Décalage détecté : {round(tuning, 2)} cents"
                note_details = "\n".join([f"• {n}: {p}%" for n, p in top_notes])
                c3.markdown(f"**Analyse :**\n{tuning_info}\n\n**Dominances :**\n{note_details}")

                # Graphique Radar
                categories = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                fig = go.Figure(data=go.Scatterpolar(r=chroma_vals, theta=categories, fill='toself', line_color='#00FFAA'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template="plotly_dark", title=f"Empreinte : {f.name}")
                st.plotly_chart(fig, use_container_width=True)

                try:
                    img_bytes = fig.to_image(format="png", width=800, height=600)
                except:
                    img_bytes = None

                tg_msg = (
                    f"🎵 *RAPPORT DJ RICARDO*\n\n"
                    f"📄 *Fichier :* `{f.name}`\n"
                    f"🎼 *Clé :* {key} {tone} ({camelot})\n"
                    f"📉 *Pitch :* {round(tuning, 2)} cents\n\n"
                    f"🎹 *Notes dominantes :*\n{note_details}"
                )
                
                send_telegram_data(tg_msg, img_bytes)
                st.success(f"Analyse envoyée pour {f.name}")

            except Exception as e:
                st.error(f"Erreur sur {f.name} : {e}")
