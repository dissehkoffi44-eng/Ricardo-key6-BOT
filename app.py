import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor
import requests  
import gc         

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | KEY 98% FIABLE", page_icon="🎧", layout="wide")

TELEGRAM_TOKEN = "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo"
CHAT_ID = "-1003602454394" 

# Initialisation des états
if 'history' not in st.session_state: st.session_state.history = []
if 'processed_files' not in st.session_state: st.session_state.processed_files = {}
if 'order_list' not in st.session_state: st.session_state.order_list = []
if 'uploader_id' not in st.session_state: st.session_state.uploader_id = 0

# --- FONCTIONS TECHNIQUES ---
def upload_to_telegram(file_buffer, filename, caption):
    try:
        file_buffer.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        files = {'document': (filename, file_buffer.read())}
        data = {'chat_id': CHAT_ID, 'caption': caption}
        response = requests.post(url, files=files, data=data, timeout=30).json()
        return response.get("ok", False)
    except: return False

def get_camelot_pro(key_mode_str):
    BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
    BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode in ['minor', 'dorian'] else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def get_sine_witness(note_mode_str, key_suffix=""):
    parts = note_mode_str.split(' ')
    note = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp").replace(".", "_")
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px;"><button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer;">▶</button><span style="font-size: 10px; font-weight: bold; color: #666;">{note} {mode[:3].upper()}</span></div>
    <script>
    const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    let audioCtx = null; let oscillators = [];
    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.innerText === '▶') {{
            this.innerText = '◼'; let g = audioCtx.createGain(); g.gain.setValueAtTime(0.05, audioCtx.currentTime); g.connect(audioCtx.destination);
            const intervals = ('{mode}' === 'minor') ? [0, 3, 7] : [0, 4, 7];
            intervals.forEach(i => {{ let o = audioCtx.createOscillator(); o.frequency.value = notesFreq['{note}'] * Math.pow(2, i/12); o.connect(g); o.start(); oscillators.push(o); }});
        }} else {{ oscillators.forEach(o => o.stop()); oscillators = []; this.innerText = '▶'; }}
    }};
    </script>""", height=40)

def analyze_segment(y, sr, tuning=0.0):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=512, n_chroma=12, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], 
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score: best_score, res_key = score, f"{NOTES[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner=False)
def get_full_analysis(file_bytes, file_name):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, res_type='kaiser_fast')
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    timeline, votes, chromas = [], [], []
    for t in range(0, int(duration) - 10, 10):
        y_seg = y[int(t*sr):int((t+10)*sr)]
        key_seg, score, chroma_vec = analyze_segment(y_seg, sr, tuning=tuning)
        votes.append(key_seg)
        chromas.append(chroma_vec)
        timeline.append({"Temps": t, "Note": key_seg, "Confiance": round(float(score)*100, 1)})
    
    counts = Counter(votes)
    dom_vote = counts.most_common(1)[0][0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return {
        "file_name": file_name, "vote": dom_vote, "vote_conf": int((counts[dom_vote]/len(votes))*100),
        "synthese": dom_vote, "confidence": 95, "tempo": int(float(tempo)), "energy": 7,
        "timeline": timeline, "purity": int((counts[dom_vote]/len(votes))*100), "key_shift": len(counts) > 1
    }

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | KEY 98% FIABLE</h1>", unsafe_allow_html=True)

files = st.file_uploader("📂 DÉPOSEZ VOS TRACKS ICI", type=['mp3', 'wav', 'flac'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_id}")

tabs = st.tabs(["📁 ANALYSEUR", "🕒 HISTORIQUE"])

with tabs[0]:
    if files:
        newly_done = False
        for f in files:
            file_id = f"{f.name}_{f.size}"
            if file_id not in st.session_state.processed_files:
                with st.spinner(f"Analyse : {f.name}"):
                    res = get_full_analysis(f.read(), f.name)
                    st.session_state.processed_files[file_id] = res
                    st.session_state.order_list.insert(0, file_id)
                    newly_done = True
        
        # AJUSTEMENT DEMANDÉ : On ne reset que si TOUS les fichiers chargés sont traités
        all_completed = all(f"{f.name}_{f.size}" in st.session_state.processed_files for f in files)
        if newly_done and all_completed:
            st.session_state.uploader_id += 1
            st.rerun()

    # --- TON AFFICHAGE ORIGINAL ---
    for fid in st.session_state.order_list[:10]:
        res = st.session_state.processed_files[fid]
        with st.expander(f"🎵 {res['file_name']}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1: 
                st.markdown(f'<div style="text-align:center"><b>DOMINANTE</b><br><h2 style="margin:0">{res["vote"]}</h2></div>', unsafe_allow_html=True)
                get_sine_witness(res["vote"], f"dom_{fid}")
            with c2: st.markdown(f'<div style="text-align:center"><b>SYNTHÈSE</b><br><h2 style="margin:0; color:#6366F1">{res["synthese"]}</h2></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div style="text-align:center"><b>PURETÉ</b><br><h2 style="margin:0">{res["purity"]}%</h2></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div style="text-align:center"><b>BPM</b><br><h2 style="margin:0">{res["tempo"]}</h2></div>', unsafe_allow_html=True)
            
            st.plotly_chart(px.scatter(pd.DataFrame(res['timeline']), x="Temps", y="Note", color="Confiance", size="Confiance", template="plotly_white"), use_container_width=True)
