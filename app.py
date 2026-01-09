import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import io
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="DJ's Ear Pro", layout="wide")

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

def get_scale_notes(key, tone):
    """Retourne les notes appartenant à la gamme détectée."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    intervals = [0, 2, 4, 5, 7, 9, 11] if tone == "Major" else [0, 2, 3, 5, 7, 8, 10]
    start_idx = notes.index(key)
    return [notes[(start_idx + i) % 12] for i in intervals]

def generate_real_piano_chord(key, tone, duration=4.0):
    """Génère un accord de piano réaliste par synthèse additive harmonique."""
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), False)
    
    notes_freq = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23,
        'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }
    
    notes_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_idx = notes_list.index(key)
    
    # Structure de l'accord : Fondamentale, Tierce (M/m), Quinte, Octave
    intervals = [0, 4, 7, 12] if tone == "Major" else [0, 3, 7, 12]
    
    chord_wave = np.zeros_like(t)
    
    for interval in intervals:
        freq = notes_freq[notes_list[(root_idx + interval) % 12]]
        if interval == 12: freq *= 2
        
        # Synthèse type Piano : Fondamentale forte + harmoniques décroissantes
        # Un vrai piano a des harmoniques qui ne sont pas parfaitement multiples (inharmonicité)
        for h in range(1, 8):
            amplitude = 1.0 / (h ** 1.5)  # Décroissance du volume des harmoniques
            # Enveloppe de résonance : Attaque rapide, déclin exponentiel plus long pour les graves
            decay = np.exp(-t * (1.5 + h * 0.5)) 
            
            harmonic_wave = amplitude * np.sin(2 * np.pi * freq * h * t) * decay
            chord_wave += harmonic_wave

    # Normalisation et ajout d'une enveloppe globale ADSR pour le réalisme
    attack = int(0.02 * sr)
    chord_wave[:attack] *= np.linspace(0, 1, attack)
    
    # Compression légère pour stabiliser le volume
    chord_wave = np.tanh(chord_wave) 
    
    chord_wave = (chord_wave * 0.5).astype(np.float32)
    return chord_wave, sr

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

def analyze_audio_optimized(file_buffer, progress_bar):
    progress_bar.progress(10, text="Chargement du fichier...")
    y, sr = librosa.load(file_buffer, sr=22050)
    
    progress_bar.progress(30, text="Correction du pitch (Auto-Tuning)...")
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    
    progress_bar.progress(50, text="Analyse harmonique (CQT)...")
    y = librosa.effects.preemphasis(y)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, bins_per_octave=24, tuning=tuning)
    
    progress_bar.progress(70, text="Calcul de la tonalité...")
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

    progress_bar.progress(90, text="Finalisation du rapport...")
    sorted_indices = np.argsort(chroma_vals)[::-1]
    top_notes = [(notes[i], round(chroma_vals[i]*100, 1)) for i in sorted_indices[:3]]
    
    progress_bar.progress(100, text="Analyse terminée !")
    time.sleep(0.5)
    progress_bar.empty()
    
    return chroma_vals, key, tone, top_notes, tuning

# --- INTERFACE ---
st.title("DJ's Ear Pro🎧")
st.markdown("Analyse de tonalité avec **Vérificateur d'Accord Piano Réel**.")

uploaded_files = st.file_uploader("Glissez vos morceaux ici", type=["mp3", "wav", "flac"], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        with st.expander(f"📊 Résultats pour : {f.name}", expanded=True):
            try:
                my_bar = st.progress(0, text=f"Analyse de {f.name}...")
                
                chroma_vals, key, tone, top_notes, tuning = analyze_audio_optimized(f, my_bar)
                camelot = get_camelot_key(key, tone)
                scale_notes = get_scale_notes(key, tone)
                
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.metric("Tonalité Détectée", f"{key} {tone}")
                c2.metric("Code Camelot", camelot)
                
                with c3:
                    st.markdown("### 🎹 Vérificateur d'accord (Piano)")
                    # Génération de l'accord piano réel
                    chord_audio, sr = generate_real_piano_chord(key, tone)
                    st.audio(chord_audio, sample_rate=sr)
                    st.caption(f"Accord acoustique de {key} {tone}")
                    
                    st.markdown(f"**Notes Dominantes :**\n" + "\n".join([f"• {n}: {p}%" for n, p in top_notes]))

                # --- Graphique Radar ---
                categories = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                radar_labels = [f"**{n}** ({tone})" if n in scale_notes else n for n in categories]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=chroma_vals, theta=radar_labels, fill='toself',
                    name=f"{key} {tone}", line_color='#00FFAA'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    template="plotly_dark",
                    title=f"Empreinte Harmonique : {f.name} | Clé : {camelot}",
                    margin=dict(l=80, r=80, t=100, b=80)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Export Image pour Telegram
                try:
                    img_bytes = fig.to_image(format="png", width=800, height=600)
                except:
                    img_bytes = None

                # --- Rapport Telegram ---
                tg_msg = (
                    f"🎵 *DJ's Ear Pro*\n\n"
                    f"📄 *Fichier :* `{f.name}`\n"
                    f"🎼 *Clé Camelot :* `{camelot}`\n"
                    f"🎹 *Mode :* {key} {tone}\n"
                    f"🌟 *Notes de la gamme :* {', '.join(scale_notes)}"
                )
                
                send_telegram_data(tg_msg, img_bytes)
                st.success(f"Analyse terminée pour {f.name}")

            except Exception as e:
                st.error(f"Erreur sur {f.name} : {e}")
