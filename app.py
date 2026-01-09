import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="DJ Ricardo's Ultimate Ear", layout="wide")

# Récupération sécurisée des secrets
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
def analyze_pro_ear_v2(file_path):
    """Analyse de niveau mastering avec correction d'accordage et filtrage harmonique."""
    # 1. Chargement du signal
    y, sr = librosa.load(file_path, sr=22050)

    # 2. Correction du Tuning (Gestion du 432Hz vs 440Hz)
    # On calcule le décalage moyen par rapport au diapason standard
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    
    # 3. Séparation Harmonique (HPSS) - On ignore le bruit et les percussions
    y_harmonic = librosa.effects.hpss(y)[0]

    # 4. Extraction du Chroma CQT avec compensation du tuning
    # On utilise une haute résolution (36 bins) pour plus de finesse
    chroma_cqt = librosa.feature.chroma_cqt(
        y=y_harmonic, 
        sr=sr, 
        tuning=tuning, 
        bins_per_octave=36,
        n_chroma=12
    )
    
    # 5. Lissage par médiane temporelle
    # Élimine les notes "accidentelles" rapides qui ne définissent pas la clé
    chroma_vals = np.median(chroma_cqt, axis=1)
    
    if np.max(chroma_vals) > 0:
        chroma_vals = chroma_vals / np.max(chroma_vals)

    # 6. Profils de Temperley (Le standard pour la musique moderne)
    maj_profile = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
    min_profile = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
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

    return chroma_vals, final_key, final_tone, tuning

# --- INTERFACE ---
st.title("DJ Ricardo's Ultimate Ear 💎")
st.markdown("Version 2.0 : Correction de pitch (Tuning) + Isolation Harmonique (HPSS).")

uploaded_file = st.file_uploader("Glissez un morceau (MP3, WAV, FLAC)", type=["mp3", "wav", "flac"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with st.spinner("Analyse chirurgicale de la tonalité..."):
        try:
            chroma_vals, key, tone, tuning_val = analyze_pro_ear_v2(tmp_path)
            camelot = get_camelot_key(key, tone)
            result_text = f"{key} {tone}"

            # Métriques
            col1, col2, col3 = st.columns(3)
            col1.metric("Tonalité Détectée", result_text)
            col2.metric("Code Camelot", camelot)
            # Affichage de l'écart d'accordage (plus proche de 0 = parfait 440Hz)
            col3.metric("Décalage Tuning", f"{tuning_val:+.2f} cents")

            # Radar Chart
            categories = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            fig = go.Figure(data=go.Scatterpolar(
                r=chroma_vals, theta=categories, fill='toself', 
                line_color='#00E676', marker=dict(size=8)
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_dark",
                title="Signature Harmonique"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Envoi Telegram
            send_telegram_message(f"🎼 *Expert Analysis*\n*Track:* {uploaded_file.name}\n*Key:* {result_text}\n*Camelot:* {camelot}\n*Tuning:* {tuning_val:.2f}c")
            st.success("Analyse terminée. Résultat envoyé sur Telegram.")

        except Exception as e:
            st.error(f"Erreur d'analyse : {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
else:
    st.info("Prêt pour l'analyse. Importez un fichier pour tester la précision.")
