import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="Audio Perception AI - Pro v2.2", layout="wide")

# Récupération sécurisée des secrets
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
    """Analyse avancée sans erreur n_bins et avec protection Nyquist."""
    # 1. Chargement (sr=4000Hz pour isoler les fréquences fondamentales)
    sr_target = 4000 
    y, sr = librosa.load(file_path, sr=sr_target)
    
    # 2. SÉPARATION HARMONIQUE (HPSS)
    # On nettoie l'intro/outro des kicks pour ne garder que la musique
    y_harmonic = librosa.effects.harmonic(y, margin=3.0)
    
    # 3. CHROMA CQT 
    # On utilise bins_per_octave=12 (standard) sans forcer n_bins pour éviter les crashs
    fmin_hz = librosa.note_to_hz('C1') 
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic, 
        sr=sr_target, 
        hop_length=512, 
        fmin=fmin_hz, 
        bins_per_octave=12
    )
    
    # Moyenne temporelle
    chroma_raw = np.mean(chroma**2, axis=1)
    
    # 4. REPLIEMENT DES OCTAVES (Gestion manuelle de n_bins)
    # On additionne toutes les octaves détectées pour obtenir 12 notes finales
    num_notes = 12
    chroma_vals = np.zeros(num_notes)
    for i in range(len(chroma_raw)):
        chroma_vals[i % num_notes] += chroma_raw[i]

    # Normalisation
    if np.max(chroma_vals) > 0:
        chroma_vals = chroma_vals / np.max(chroma_vals)

    # Profils de corrélation Krumhansl-Schmuckler
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    scores = []
    for i in range(12):
        p_maj, p_min = np.roll(maj_profile, i), np.roll(min_profile, i)
        scores.append((np.corrcoef(chroma_vals, p_maj)[0, 1], notes[i], "Major"))
        scores.append((np.corrcoef(chroma_vals, p_min)[0, 1], notes[i], "Minor"))

    # Classement pour trouver la meilleure clé et calculer la confiance
    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, final_key, final_tone = scores[0]
    second_best_score = scores[1][0]
    
    # Calcul de confiance (différence entre le 1er et le 2ème choix)
    diff = best_score - second_best_score
    confidence_pct = min(int(diff * 500), 100) if best_score > 0 else 0

    return chroma_vals, final_key, final_tone, confidence_pct

# --- INTERFACE ---
st.title("🧠 Audio Perception AI (Elite v2.2)")
st.markdown("---")

uploaded_file = st.file_uploader("Glissez votre fichier audio ici", type=["mp3", "wav", "flac"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.audio(uploaded_file)
    
    with st.spinner(f"Analyse chirurgicale : {uploaded_file.name}..."):
        try:
            chroma_vals, key, tone, confidence = analyze_human_perception(tmp_path, uploaded_file.name)
            camelot = get_camelot_key(key, tone)
            result_text = f"{key} {tone}"

            # Affichage des résultats
            col1, col2, col3 = st.columns(3)
            col1.metric("Tonalité Détectée", result_text)
            col2.metric("Code Camelot", camelot)
            col3.metric("Confiance", f"{confidence}%")

            # Radar Chart
            categories = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            fig = go.Figure(data=go.Scatterpolar(r=chroma_vals, theta=categories, fill='toself', line_color='#00FFAA'))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_dark",
                title="Signature Harmonique"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Envoi Telegram
            msg = (f"🎵 *Analyse*\n"
                   f"*Fichier :* `{uploaded_file.name}`\n"
                   f"*Résultat :* `{result_text}`\n"
                   f"*Camelot :* `{camelot}`\n"
                   f"*Confiance :* `{confidence}%` Ratio")
            send_telegram_message(msg)
            
            st.success("Analyse terminée.")

        except Exception as e:
            st.error(f"Erreur d'analyse : {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

st.markdown("---")
st.caption("Mise à jour : Gestion automatique des octaves et filtrage harmonique activés.")
