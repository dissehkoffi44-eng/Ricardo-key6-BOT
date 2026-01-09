import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import tempfile
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="DJ Ricardo's Pro Ear", layout="wide")

# Secrets Telegram
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
    """Envoie le rapport texte et l'image du radar à Telegram."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
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
    """Analyse optimisée utilisant le buffer direct (UploadedFile)."""
    # Librosa peut lire directement depuis un objet de type fichier (buffer)
    y, sr = librosa.load(file_buffer, sr=22050)
    y = librosa.effects.preemphasis(y)
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
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

    # Top 3 des notes dominantes
    sorted_indices = np.argsort(chroma_vals)[::-1]
    top_notes = [(notes[i], round(chroma_vals[i]*100, 1)) for i in sorted_indices[:3]]
    
    return chroma_vals, key, tone, top_notes

# --- INTERFACE ---
st.title("DJ Ricardo's Pro Ear 🎧")
st.markdown("Analyse multi-fichiers optimisée (Gestion mémoire améliorée).")

# accept_multiple_files=True permet de glisser plusieurs fichiers d'un coup
uploaded_files = st.file_uploader("Glissez vos morceaux ici", type=["mp3", "wav", "flac"], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        with st.expander(f"Résultats pour : {f.name}", expanded=True):
            try:
                # OPTIMISATION MÉMOIRE : on passe 'f' directement au lieu de 'f.read()'
                # 'f' est un objet UploadedFile qui se comporte comme un buffer ouvert.
                chroma_vals, key, tone, top_notes = analyze_audio_optimized(f)
                
                camelot = get_camelot_key(key, tone)
                
                # Affichage UI
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.metric("Tonalité", f"{key} {tone}")
                c2.metric("Code Camelot", camelot)
                
                note_details = "\n".join([f"• {n}: {p}%" for n, p in top_notes])
                c3.markdown(f"**Dominances harmoniques :**\n{note_details}")

                # Graphique Radar
                categories = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                fig = go.Figure(data=go.Scatterpolar(r=chroma_vals, theta=categories, fill='toself', line_color='#00FFAA'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", title=f"Profil : {f.name}")
                st.plotly_chart(fig, use_container_width=True)

                # Export Image pour Telegram (nécessite la lib 'kaleido')
                try:
                    img_bytes = fig.to_image(format="png", width=800, height=600)
                except:
                    img_bytes = None
                    st.warning("Installez 'kaleido' pour envoyer le graphique sur Telegram.")

                # Rapport enrichi
                tg_msg = (
                    f"🎵 *RAPPORT DJ RICARDO*\n\n"
                    f"📄 *Fichier :* `{f.name}`\n"
                    f"🎼 *Clé :* {key} {tone} ({camelot})\n\n"
                    f"🎹 *Notes dominantes :*\n{note_details}"
                )
                
                send_telegram_data(tg_msg, img_bytes)
                st.success(f"Analyse envoyée pour {f.name}")

            except Exception as e:
                st.error(f"Erreur sur {f.name} : {e}")
else:
    st.info("Prêt pour l'analyse. Déposez vos fichiers audio.")
