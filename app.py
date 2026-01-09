import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import requests
import gc
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter

# --- CONFIGURATION ---
st.set_page_config(page_title="Ultra Key Detector PRO", page_icon="🎹", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- CONSTANTES & PROFILS ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .report-card { 
        background: linear-gradient(135deg, #1e3a8a, #581c87); 
        padding: 40px; border-radius: 20px; text-align: center; color: white; margin-bottom: 20px;
    }
    .metric-box {
        background: #1a1c24; border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR AUDIO ---

def apply_filters(y, sr):
    """Combine Pré-emphase (C1) et Butterworth (C2)."""
    y = librosa.effects.preemphasis(y)
    nyq = 0.5 * sr
    b, a = butter(4, [100/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y)

def solve_key(chroma_vector):
    """Trouve la meilleure tonalité selon plusieurs profils."""
    best_score = -1
    res = {"key": "", "score": 0}
    # Normalisation
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                if score > best_score:
                    best_score = score
                    res = {"key": f"{NOTES_LIST[i]} {mode}", "score": score}
    return res

# --- ANALYSE ---

@st.cache_data(show_spinner=False)
def analyze_absolute(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050)
    
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_filters(y, sr)
    
    # Analyse par segments (Stabilité)
    step = 8
    timeline = []
    votes = Counter()
    
    for start in range(0, int(duration) - step, step):
        seg = y_filt[int(start*sr):int((start+step)*sr)]
        if np.max(np.abs(seg)) < 0.02: continue
        
        # CQT Chroma amélioré
        chroma = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, bins_per_octave=24)
        chroma_avg = np.mean(chroma**2, axis=1)
        
        result = solve_key(chroma_avg)
        votes[result['key']] += int(result['score'] * 100)
        timeline.append({"Temps": start, "Note": result['key'], "Conf": result['score']})

    final_key = votes.most_common(1)[0][0]
    avg_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == final_key]) * 100)
    
    # Global Chroma pour Radar
    global_chroma = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return {
        "key": final_key,
        "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": avg_conf,
        "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline,
        "chroma": global_chroma,
        "name": file_name
    }

# --- UI INTERFACE ---
st.title("🎧 Ultra Key Detector PRO")
st.write("Fusion des moteurs RCDJ228 M3 & Musical Ear")

uploaded_file = st.file_uploader("Déposez un fichier audio", type=['mp3','wav','flac'])

if uploaded_file:
    file_bytes = uploaded_file.read()
    
    with st.spinner("Analyse spectrale en cours..."):
        data = analyze_absolute(file_bytes, uploaded_file.name)
        
    # --- AFFICHAGE RÉSULTAT ---
    st.markdown(f"""
        <div class="report-card">
            <h3 style="opacity:0.7;">TONALITÉ DÉTECTÉE</h3>
            <h1 style="font-size:5em; margin:0;">{data['key'].upper()}</h1>
            <p style="font-size:1.5em;">CAMELOT: <b>{data['camelot']}</b> | CONFIANCE: <b>{data['conf']}%</b></p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:1.5em;'>{data['tempo']} BPM</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:1.5em;'>{data['tuning']} Hz</span></div>", unsafe_allow_html=True)
    with col3:
        # Mini Oscillateur de test
        n, m = data['key'].split()
        js_id = "test_audio"
        components.html(f"""
            <button id="{js_id}" style="width:100%; height:55px; background:#6366F1; color:white; border:none; border-radius:10px; cursor:pointer; font-weight:bold;">🔊 TESTER L'ACCORD</button>
            <script>
            document.getElementById('{js_id}').onclick = function() {{
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
                const intervals = '{m}' === 'minor' ? [0, 3, 7] : [0, 4, 7];
                intervals.forEach(i => {{
                    const o = ctx.createOscillator(); const g = ctx.createGain();
                    o.frequency.value = freqs['{n}'] * Math.pow(2, i/12);
                    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 1.5);
                    o.connect(g); g.connect(ctx.destination);
                    o.start(); o.stop(ctx.currentTime + 1.5);
                }});
            }}
            </script>""", height=70)

    # --- GRAPHES ---
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("📈 Stabilité Temporelle")
        df_tl = pd.DataFrame(data['timeline'])
        fig_line = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
        st.plotly_chart(fig_line, use_container_width=True)
        
    with c_right:
        st.subheader("🌀 Empreinte Harmonique")
        fig_radar = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#00FFAA'))
        fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- TELEGRAM ---
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            msg = f"🎹 *ANALYSE ABSOLUE*\n📂 `{data['name']}`\n\n*Résultat:* {data['key']}\n*Camelot:* {data['camelot']}\n*Confiance:* {data['conf']}%\n*Tempo:* {data['tempo']} BPM"
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        except:
            pass

else:
    st.info("Téléchargez un morceau pour lancer l'analyse combinée.")
