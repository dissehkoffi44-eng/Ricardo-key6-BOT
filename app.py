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

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="Absolute Key Detector V4", page_icon="🎼", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
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
        background: linear-gradient(135deg, #0f172a, #1e1b4b); 
        padding: 40px; border-radius: 25px; text-align: center; color: white; 
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.1); color: #f87171;
        padding: 10px; border-radius: 10px; border: 1px solid #ef4444;
        margin-top: 15px; font-weight: bold;
    }
    .metric-box {
        background: #1a1c24; border-radius: 15px; padding: 15px; text-align: center; border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE TRAITEMENT ---

def apply_filters(y, sr):
    y = librosa.effects.preemphasis(y)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 6000/nyq], btype='band')
    return lfilter(b, a, y)

def solve_key(chroma_vector):
    best_score = -1
    res = {"key": "Inconnu", "score": 0}
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                if score > best_score:
                    best_score = score
                    res = {"key": f"{NOTES_LIST[i]} {mode}", "score": score}
    return res

@st.cache_data(show_spinner=False)
def analyze_full_engine(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050, mono=True)
    
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_filters(y, sr)
    
    # Analyse segmentée
    step = 6 
    timeline = []
    votes = Counter()
    
    for start in range(0, int(duration) - step, step):
        seg = y_filt[int(start*sr):int((start+step)*sr)]
        if np.max(np.abs(seg)) < 0.02: continue
        
        chroma = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, bins_per_octave=36)
        chroma_avg = np.mean(chroma**2, axis=1)
        
        result = solve_key(chroma_avg)
        votes[result['key']] += int(result['score'] * 100)
        timeline.append({"Temps": start, "Note": result['key'], "Conf": result['score']})

    # Détection de la tonalité principale et de la modulation
    most_common = votes.most_common(2)
    main_key = most_common[0][0]
    
    target_key = None
    modulation_detected = False
    
    # Si on a une deuxième tonalité qui représente une part significative (>30% du score de la principale)
    if len(most_common) > 1:
        second_key = most_common[1][0]
        # On vérifie si cette clé apparaît de manière consistante dans la timeline
        unique_keys_in_flow = [t['Note'] for t in timeline]
        if unique_keys_in_flow.count(second_key) > (len(timeline) * 0.15):
            modulation_detected = True
            target_key = second_key

    avg_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == main_key]) * 100)
    
    # Tempo & Chromagramme global
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    full_chroma = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
    
    output = {
        "key": main_key, "camelot": CAMELOT_MAP.get(main_key, "??"),
        "conf": avg_conf, "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline, "chroma": full_chroma,
        "modulation": modulation_detected, 
        "target_key": target_key,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "name": file_name
    }
    
    del y, y_filt
    gc.collect()
    return output

# --- INTERFACE UTILISATEUR ---

st.title("🎧 ABSOLUTE KEY DETECTOR V4")
st.subheader("Moteur Hybride : Détection de Modulation de Précision")

uploaded_files = st.file_uploader("📂 Glissez vos fichiers audio ici", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        with st.spinner(f"Analyse de {f.name}..."):
            data = analyze_full_engine(f.read(), f.name)
        
        with st.expander(f"📊 RÉSULTATS : {data['name']}", expanded=True):
            # Affichage Principal
            bg_color = "linear-gradient(135deg, #1e1b4b, #312e81)" if not data['modulation'] else "linear-gradient(135deg, #1e1b4b, #450a0a)"
            
            modulation_html = ""
            if data['modulation']:
                modulation_html = f"""
                <div class='modulation-alert'>
                    ⚠️ MODULATION DÉTECTÉE <br>
                    <span style='font-size:1.2em;'>Vers : {data['target_key'].upper()} ({data['target_camelot']})</span>
                </div>
                """

            st.markdown(f"""
                <div class="report-card" style="background:{bg_color};">
                    <p style="text-transform:uppercase; letter-spacing:2px; opacity:0.7;">Tonalité Dominante</p>
                    <h1 style="font-size:6em; margin:10px 0;">{data['key'].upper()}</h1>
                    <p style="font-size:1.8em;">CAMELOT : <b>{data['camelot']}</b> | CONFIANCE : <b>{data['conf']}%</b></p>
                    {modulation_html}
                </div>
            """, unsafe_allow_html=True)

            st.write("---")
            
            # Métriques
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:1.8em;'>{data['tempo']} BPM</span></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-box'><b>ACCORDAGE (REF)</b><br><span style='font-size:1.8em;'>{data['tuning']} Hz</span></div>", unsafe_allow_html=True)
            with m3:
                # Oscillateur
                n, mode = data['key'].split()
                uid = f.name.replace(".","").replace(" ","")
                components.html(f"""
                    <button id="btn_{uid}" style="width:100%; height:65px; background:linear-gradient(90deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:12px; cursor:pointer; font-weight:bold; font-size:1.1em; box-shadow:0 4px 15px rgba(0,0,0,0.3);">🔊 TESTER L'ACCORD</button>
                    <script>
                    document.getElementById('btn_{uid}').onclick = function() {{
                        const ctx = new (window.AudioContext || window.webkitAudioContext)();
                        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
                        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
                        intervals.forEach((step, i) => {{
                            const o = ctx.createOscillator(); const g = ctx.createGain();
                            o.type = 'triangle';
                            o.frequency.setValueAtTime(freqs['{n}'] * Math.pow(2, step/12), ctx.currentTime);
                            g.gain.setValueAtTime(0, ctx.currentTime);
                            g.gain.linearRampToValueAtTime(0.2, ctx.currentTime + 0.1);
                            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
                            o.connect(g); g.connect(ctx.destination);
                            o.start(); o.stop(ctx.currentTime + 2.0);
                        }});
                    }}
                    </script>""", height=85)

            # Visualisations
            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.markdown("#### 📈 Stabilité et Courbe de Modulation")
                df_tl = pd.DataFrame(data['timeline'])
                fig_line = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", 
                                   category_orders={"Note": NOTES_ORDER}, color_discrete_sequence=['#818cf8'])
                fig_line.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300)
                st.plotly_chart(fig_line, use_container_width=True)
                
            with c_right:
                st.markdown("#### 🌀 Profil Harmonique")
                fig_radar = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#818cf8'))
                fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)), margin=dict(l=30,r=30,t=30,b=30), height=300)
                st.plotly_chart(fig_radar, use_container_width=True)

            # Envoi Telegram
            if TELEGRAM_TOKEN and CHAT_ID:
                try:
                    mod_txt = f"⚠️ Modulation vers {data['target_key']}" if data['modulation'] else "✅ Stable"
                    msg = (f"🎹 *RAPPORT ABSOLUTE V4*\n"
                           f"📂 `{data['name']}`\n\n"
                           f"*Tonalité:* `{data['key']}`\n"
                           f"*Camelot:* `{data['camelot']}`\n"
                           f"*Stabilité:* {mod_txt}\n"
                           f"*Confiance:* `{data['conf']}%`\n"
                           f"*Tempo:* `{data['tempo']} BPM`")
                    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                                  json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
                except: pass

    if st.sidebar.button("🧹 Vider le cache mémoire"):
        st.cache_data.clear()
        st.rerun()
else:
    st.info("En attente de fichiers audio pour commencer l'analyse absolue.")
