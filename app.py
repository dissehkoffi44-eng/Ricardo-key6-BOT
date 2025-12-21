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

# --- IMPORT POUR LES TAGS MP3 (MUTAGEN) ---
try:
    from mutagen.id3 import ID3, TKEY
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | KEY 98% FIABLE", page_icon="🎧", layout="wide")

# Paramètres Telegram
TELEGRAM_TOKEN = "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo"
CHAT_ID = "-1003602454394" 

# --- INITIALISATION DES ÉTATS ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'order_list' not in st.session_state:
    st.session_state.order_list = []
# On initialise la clé de l'uploader si elle n'existe pas
if 'uploader_id' not in st.session_state:
    st.session_state.uploader_id = 0

# --- FONCTION STOCKAGE EXTERNE TELEGRAM ---
def upload_to_telegram(file_buffer, filename, caption):
    try:
        file_buffer.seek(0)
        url = "https://api.telegram.org/bot{}/sendDocument".format(TELEGRAM_TOKEN)
        files = {'document': (filename, file_buffer.read())}
        data = {'chat_id': CHAT_ID, 'caption': caption}
        response = requests.post(url, files=files, data=data, timeout=30).json()
        return response.get("ok", False)
    except Exception as e:
        st.error(f"Erreur Telegram : {e}")
        return False

# --- DESIGN CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #1A1A1A; }
    .diag-box { text-align:center; padding:10px; border-radius:10px; border:1px solid #EEE; background: white; }
    .stFileUploader { border: 2px dashed #6366F1; padding: 20px; border-radius: 15px; background: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR AUDIO JS ---
def get_sine_witness(note_mode_str, key_suffix=""):
    parts = note_mode_str.split(' ')
    note = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp").replace(".", "_")
    
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: sans-serif;">
        <button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 12px;">▶</button>
        <span style="font-size: 9px; font-weight: bold; color: #666;">{note} {mode[:3].upper()}</span>
    </div>
    <script>
    const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    const semitones = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    let audioCtx = null; let oscillators = []; let gainNode = null;
    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.innerText === '▶') {{
            this.innerText = '◼'; this.style.background = '#E74C3C';
            gainNode = audioCtx.createGain(); gainNode.gain.setValueAtTime(0.05, audioCtx.currentTime);
            gainNode.connect(audioCtx.destination);
            const rootIdx = semitones.indexOf('{note}');
            const intervals = ('{mode}' === 'minor' || '{mode}' === 'dorian') ? [0, 3, 7] : [0, 4, 7];
            intervals.forEach(interval => {{
                let osc = audioCtx.createOscillator(); osc.type = 'sine';
                let freq = notesFreq['{note}'] * Math.pow(2, interval / 12);
                if (!freq) freq = 440; 
                osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
                osc.connect(gainNode); osc.start(); oscillators.push(osc);
            }});
        }} else {{
            oscillators.forEach(o => o.stop()); oscillators = [];
            this.innerText = '▶'; this.style.background = '#6366F1';
        }}
    }};
    </script>
    """, height=40)

# --- MAPPING CAMELOT ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']: return BASE_CAMELOT_MINOR.get(key, "??")
        else: return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

# --- MOTEUR ANALYSE ---
def check_drum_alignment(y, sr):
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_max_mean = np.mean(np.max(chroma, axis=0))
    return flatness < 0.045 or chroma_max_mean > 0.75

def analyze_segment(y, sr, tuning=0.0):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=512, n_chroma=12, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], 
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], 
        "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17]
    }
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score: 
                best_score, res_key = score, f"{NOTES[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse Multi-Couches V6.1 Hybrid...", max_entries=20)
def get_full_analysis(file_bytes, file_name):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, res_type='kaiser_fast')
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    is_aligned = check_drum_alignment(y, sr)
    y_final, filter_applied = (y, False) if is_aligned else (librosa.effects.hpss(y)[0], True)
    
    duration = librosa.get_duration(y=y_final, sr=sr)
    timeline_data, votes, all_chromas = [], [], []
    
    for start_t in range(0, int(duration) - 10, 10):
        y_seg = y_final[int(start_t*sr):int((start_t+10)*sr)]
        key_seg, score_seg, chroma_vec = analyze_segment(y_seg, sr, tuning=tuning_offset)
        votes.append(key_seg)
        all_chromas.append(chroma_vec)
        timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": round(float(score_seg) * 100, 1)})
    
    counts = Counter(votes)
    dominante_vote = counts.most_common(1)[0][0]
    dominante_conf = int((counts[dominante_vote] / len(votes)) * 100)
    avg_chroma_global = np.mean(all_chromas, axis=0)
    
    PROFILES_SYNTH = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}
    best_synth_score, tonique_synth = -1, ""
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for mode, profile in PROFILES_SYNTH.items():
        for i in range(12):
            score = np.corrcoef(avg_chroma_global, np.roll(profile, i))[0, 1]
            if score > best_synth_score: best_synth_score, tonique_synth = score, f"{NOTES[i]} {mode}"

    top_votes = Counter(votes).most_common(2)
    purity = int((top_votes[0][1] / len(votes)) * 100)
    key_shift_detected = True if len(top_votes) > 1 and (top_votes[1][1] / len(votes)) > 0.25 else False
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = int(np.clip(np.mean(librosa.feature.rms(y=y))*35 + (float(tempo)/160), 1, 10))

    return {
        "file_name": file_name,
        "vote": dominante_vote, "vote_conf": dominante_conf, 
        "synthese": tonique_synth, "confidence": int(best_synth_score*100), "tempo": int(float(tempo)), 
        "energy": energy, "timeline": timeline_data, "purity": purity, 
        "key_shift": key_shift_detected, "secondary": top_votes[1][0] if len(top_votes)>1 else top_votes[0][0],
        "is_filtered": filter_applied
    }

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | KEY 98% FIABLE</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ MAINTENANCE")
    if st.button("🧹 VIDER TOUT (REDÉMARRAGE)"):
        st.session_state.history = []
        st.session_state.processed_files = {}
        st.session_state.order_list = []
        st.cache_data.clear()
        gc.collect()
        st.rerun()

# --- ZONE D'IMPORTATION ---
# Utilisation de la clé dynamique pour réinitialiser l'uploader
files = st.file_uploader("📂 DÉPOSEZ VOS TRACKS ICI", type=['mp3', 'wav', 'flac'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_id}")

tabs = st.tabs(["📁 ANALYSEUR", "🕒 HISTORIQUE"])

with tabs[0]:
    if files:
        processed_something = False
        for f in files:
            file_id = f"{f.name}_{f.size}"
            if file_id not in st.session_state.processed_files:
                with st.spinner(f"Traitement : {f.name}"):
                    f_bytes = f.read()
                    res = get_full_analysis(f_bytes, f.name)
                    cam_val = get_camelot_pro(res['synthese'])
                    
                    df_tl_tg = pd.DataFrame(res['timeline']).sort_values(by="Confiance", ascending=False).reset_index()
                    n1_tg = df_tl_tg.loc[0, 'Note'] if not df_tl_tg.empty else "??"
                    c1_tg = df_tl_tg.loc[0, 'Confiance'] if not df_tl_tg.empty else 0
                    n2_tg = n1_tg
                    c2_tg = 0
                    if not df_tl_tg.empty:
                        for idx, row in df_tl_tg.iterrows():
                            if row['Note'] != n1_tg:
                                n2_tg = row['Note']
                                c2_tg = row['Confiance']
                                break

                    tg_caption = (f"🎵 {f.name}\n🥁 BPM: {res['tempo']}\n🎯 DOMINANTE: {res['vote']} ({get_camelot_pro(res['vote'])}) - {res['vote_conf']}%\n🧬 SYNTHÈSE: {res['synthese']} ({cam_val}) - {res['confidence']}%")

                    upload_to_telegram(io.BytesIO(f_bytes), f"[{cam_val}] {f.name}", tg_caption)
                    st.session_state.processed_files[file_id] = res
                    if file_id not in st.session_state.order_list:
                        st.session_state.order_list.insert(0, file_id)
                    processed_something = True
                    gc.collect()
        
        # --- RÉINITIALISATION DE LA ZONE DE DÉPÔT ---
        if processed_something:
            st.session_state.uploader_id += 1 # On change l'ID pour forcer Streamlit à créer un nouvel uploader vide
            st.rerun()

    # AFFICHAGE
    st.subheader("Analyses récentes")
    for fid in st.session_state.order_list[:10]:
        res = st.session_state.processed_files[fid]
        file_name = res['file_name']
        with st.expander(f"🎵 {file_name}", expanded=True):
            cam_final = get_camelot_pro(res['synthese'])
            if not any(h['Fichier'] == file_name for h in st.session_state.history): 
                st.session_state.history.insert(0, {"Date": datetime.now().strftime("%d/%m %H:%M"), "Fichier": file_name, "Note": res['synthese'], "Camelot": cam_final, "BPM": res['tempo']})

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f'<div class="metric-container"><div class="label-custom">DOMINANTE</div><div class="value-custom">{res["vote"]}</div><div>{get_camelot_pro(res["vote"])}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #6366F1;"><div class="label-custom">SYNTHÈSE</div><div class="value-custom">{res["synthese"]}</div><div>{cam_final}</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-container"><div class="label-custom">PURETÉ</div><div class="value-custom">{res["purity"]}%</div></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="metric-container"><div class="label-custom">BPM</div><div class="value-custom">{res["tempo"]}</div></div>', unsafe_allow_html=True)
            
            st.plotly_chart(px.scatter(pd.DataFrame(res['timeline']), x="Temps", y="Note", color="Confiance", template="plotly_white"), use_container_width=True)

with tabs[1]:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else: st.info("Historique vide.")
