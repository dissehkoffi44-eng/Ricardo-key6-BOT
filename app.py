import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import requests
import os
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="Audio Perception AI - Pro v2.1", layout="wide")

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
    """Analyse avancée avec HSS et gestion de la limite de Nyquist."""
    # 1. Chargement avec un taux d'échantillonnage de sécurité
    sr_target = 4000 
    y, sr = librosa.load(file_path, sr=sr_target)
    
    # 2. SÉPARATION HARMONIQUE (HPSS)
    # On élimine les drums/percussions qui polluent l'intro et l'outro
    y_harmonic = librosa.effects.harmonic(y, margin=3.0)
    
    # 3. CHROMA CQT 
    # n_bins=60 (5 octaves) pour ne pas dépasser la fréquence de Nyquist (sr_target/2)
    fmin_hz = librosa.note_to_hz('C1') # 32.7 Hz pour capter les basses fréquences
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic, 
        sr=sr_target, 
        hop_length=512, 
        fmin=fmin_hz, 
        n_bins=60
    )
    
    # Moyenne et normalisation
    chroma_vals = np.mean(chroma**2, axis=1)
    if np.max(chroma_vals) > 0:
        chroma_vals = chroma_vals / np.max(chroma_vals)

    # Profils Krumhansl-Schmuckler
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    scores = []
    for i in range(12):
        p_maj, p_min = np.roll(maj_profile, i), np.roll(min_profile, i)
        scores.append((np.corrcoef(chroma_vals, p_maj)[0, 1], notes[i], "Major"))
        scores.append((np.corrcoef(chroma_vals, p_min)[0, 1], notes[i], "Minor"))

    # Tri des scores pour trouver le meilleur et calculer la confiance
    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, final_key, final_tone = scores[0]
    second_best_score = scores[1][0]
    
    # Calcul d'un indice de confiance (écart entre le 1er et le 2ème meilleur choix)
    confidence = (best_score - second_best_score) / best_score if best_score > 0 else 0
    confidence_pct = min(int(confidence * 500), 100) # Normalisation simple

    return chroma_vals, final_key, final_tone, confidence_pct

# --- INTERFACE STREAMLIT ---
st.title("🧠 Audio Perception AI (Elite Edition)")
st.markdown("---")

uploaded_file = st.file_uploader("Glissez votre fichier audio ici", type=["mp3", "wav", "flac"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.audio(uploaded_file)
    
    with st.spinner(f"Analyse chirurgicale de : {uploaded_file.name}..."):
        try:
            chroma_vals, key, tone, confidence = analyze_human_perception(tmp_path, uploaded_file.name)
            camelot = get_camelot_key(key, tone)
            result_text = f"{key} {tone}"

            # Affichage des métriques
            m1, m2, m3 = st.columns(3)
            m1.metric("Tonalité Détectée", result_text)
            m2.metric("Code Camelot", camelot)
            m3.metric("Indice de Confiance", f"{confidence}%")

            # Graphique Radar
            categories = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            fig = go.Figure(data=go.Scatterpolar(r=chroma_vals, theta=categories, fill='toself', line_color='#00FFAA'))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_dark",
                title="Signature Harmonique (Profil de Notes)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Envoi Telegram
            msg = (f"🎵 *Analyse Audio*\n"
                   f"*Fichier :* `{uploaded_file.name}`\n"
                   f"*Résultat :* `{result_text}`\n"
                   f"*Camelot :* `{camelot}`\n"
                   f"*Confiance :* `{confidence}%`")
            send_telegram_message(msg)
            
            st.success("Analyse terminée avec succès.")

        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

st.markdown("---")
st.caption("Note : La séparation harmonique (HPSS) est active. L'algorithme ignore les kicks et percussions pour se concentrer sur la mélodie.")
