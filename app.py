import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Psycho-Acoustique : Perception des Tons", layout="wide")

def get_camelot_key(key, tone):
    """
    Convertit une clé musicale en notation Camelot (ex: F# Minor -> 11A).
    Inclut votre référence spécifique : F# Minor = 11A.
    """
    camelot_map = {
        'C Major': '8B', 'G Major': '9B', 'D Major': '10B', 'A Major': '11B', 'E Major': '12B', 'B Major': '1B',
        'F# Major': '2B', 'C# Major': '3B', 'G# Major': '4B', 'D# Major': '5B', 'A# Major': '6B', 'F Major': '7B',
        'A Minor': '8A', 'E Minor': '9A', 'B Minor': '10A', 'F# Minor': '11A', 'C# Minor': '12A', 'G# Minor': '1A',
        'D# Minor': '2A', 'A# Minor': '3A', 'F Minor': '4A', 'C Minor': '5A', 'G Minor': '6A', 'D Minor': '7A'
    }
    return camelot_map.get(f"{key} {tone}", "Inconnu")

def analyze_audio(audio_file):
    # Chargement de l'audio
    y, sr = librosa.load(audio_file, sr=None)
    
    # 1. Analyse de la Tonalité (Chromagramme)
    # On utilise CQT (Constant-Q Transform) qui imite la résolution logarithmique de l'oreille
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = np.argmax(chroma_mean)
    detected_key = notes[key_idx]
    
    # Estimation Simple Majeur/Mineur basé sur l'énergie du chromagramme
    # (Approche simplifiée pour une démo professionnelle)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Corrélation pour déterminer le mode
    major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, key_idx))[0, 1]
    minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, key_idx))[0, 1]
    tone = "Major" if major_corr > minor_corr else "Minor"
    
    return y, sr, chroma_mean, detected_key, tone

# UI Streamlit
st.title("👂 Analyseur de Perception Auditive")
st.markdown("""
Cette application simule la manière dont l'oreille humaine perçoit les fréquences et les tonalités d'une chanson, 
en utilisant la pondération psycho-acoustique et la transformation de chromagramme.
""")

uploaded_file = st.file_uploader("Choisissez un fichier audio (mp3, wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    with st.spinner("Analyse de l'empreinte tonale..."):
        y, sr, chroma_mean, key, tone = analyze_audio(uploaded_file)
        camelot = get_camelot_key(key, tone)
        
        # Affichage des résultats clés
        col1, col2, col3 = st.columns(3)
        col1.metric("Tonalité Détectée", f"{key} {tone}")
        col2.metric("Code Camelot", camelot)
        col3.metric("Sample Rate", f"{sr} Hz")

        st.divider()

        # Visualisation de la perception des notes
        st.subheader("📊 Répartition de l'énergie tonale (Perception des notes)")
        
        df_chroma = pd.DataFrame({
            'Note': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
            'Intensité Perçue': chroma_mean
        })
        
        fig = go.Figure(data=[
            go.Bar(x=df_chroma['Note'], y=df_chroma['Intensité Perçue'],
                   marker_color=['#636EFA' if n != key else '#EF553B' for n in df_chroma['Note']])
        ])
        fig.update_layout(xaxis_title="Classes de hauteurs (Pitch Classes)", yaxis_title="Énergie")
        st.plotly_chart(fig, use_container_width=True)

        # Spectrogramme avec pondération A (Pondération de l'oreille humaine)
        st.subheader("🔊 Spectrogramme de perception (Pondération A)")
        st.info("Ce graphique montre les fréquences dominantes ajustées selon la sensibilité de l'oreille humaine.")
        
        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        a_weights = librosa.A_weighting(freqs)
        # Application de la pondération
        weighted_spectrogram = librosa.amplitude_to_db(D, ref=np.max) + a_weights[:, np.newaxis]
        
        fig_spec, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(weighted_spectrogram, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        plt.colorbar(img, ax=ax, format="%+2.f dB")
        st.pyplot(fig_spec)

else:
    st.info("Veuillez uploader un fichier pour commencer l'analyse.")

st.sidebar.title("À propos")
st.sidebar.info(f"""
**Note technique :**
- Utilise la Transformée de Constant-Q pour la détection de notes.
- Intègre la table de correspondance Camelot (ex: **F# Minor = 11A**).
- Applique une pondération de type 'A' pour simuler la perte de sensibilité aux basses et très hautes fréquences.
""")
