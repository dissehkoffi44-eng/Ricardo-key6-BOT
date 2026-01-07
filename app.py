import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import streamlit.components.v1 as components
import requests  
import gc                

# --- CONFIGURATION & CSS ---
st.set_page_config(page_title="KEY V6.2 Perceptual Pro", page_icon="🎧", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #1A1A1A; }
    
    .final-decision-box { 
        background: linear-gradient(135deg, #1e1e2f 0%, #6366F1 100%); 
        color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px;
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.2); border: 1px solid #6366F1;
    }
    .stFileUploader { border: 2px dashed #6366F1; padding: 20px; border-radius: 15px; background: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION API ---
TELEGRAM_TOKEN = "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo"
CHAT_ID = "-1003602454394" 

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'order_list' not in st.session_state:
    st.session_state.order_list = []

# --- FONCTIONS TECHNIQUES ---
def upload_to_telegram(file_buffer, filename, caption):
    try:
        file_buffer.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        files = {'document': (filename, file_buffer.read())}
        data = {'chat_id': CHAT_ID, 'caption': caption}
        response = requests.post(url, files=files, data=data, timeout=30).json()
        return response.get("ok", False)
    except:
        return False

def get_sine_witness(note_mode_str, key_suffix=""):
    parts = note_mode_str.split(' ')
    note = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp").replace(".", "_")
    
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: sans-serif;">
        <button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 12px;">▶</button>
        <span style="font-size: 9px; font-weight: bold; color: #666;">{note} {mode[:3].upper()} PIANO</span>
    </div>
    <script>
    const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    let audioCtx = null;
    let masterGain = null;

    function playPianoTone(freq, startTime) {{
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, startTime);
        gain.gain.setValueAtTime(0, startTime);
        gain.gain.linearRampToValueAtTime(0.15, startTime + 0.01); 
        gain.gain.exponentialRampToValueAtTime(0.001, startTime + 2.0);
        osc.connect(gain);
        gain.connect(masterGain);
        osc.start(startTime);
        osc.stop(startTime + 2.0);
    }}

    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.innerText === '▶') {{
            this.innerText = '◼'; this.style.background = '#E74C3C';
            masterGain = audioCtx.createGain();
            masterGain.connect(audioCtx.destination);
            const isMinor = '{mode}' === 'minor' || '{mode}' === 'dorian';
            const intervals = isMinor ? [0, 3, 7, 12] : [0, 4, 7, 12];
            const now = audioCtx.currentTime;
            intervals.forEach((interval, index) => {{
                let freq = notesFreq['{note}'] * Math.pow(2, interval / 12);
                playPianoTone(freq, now + (index * 0.04));
            }});
            setTimeout(() => {{
                this.innerText = '▶'; this.style.background = '#6366F1';
            }}, 2000);
        }} else {{
            if(masterGain) masterGain.gain.setTargetAtTime(0, audioCtx.currentTime, 0.1);
            this.innerText = '▶'; this.style.background = '#6366F1';
        }}
    }};
    </script>
    """, height=40)

# Base Camelot (F# Minor = 11A selon vos instructions)
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']: return BASE_CAMELOT_MINOR.get(key, "??")
        else: return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

# --- MOTEUR ANALYSE PERCEPTUELLE ---
def analyze_segment_perceptual(y, sr, tuning=0.0):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 1. Constant-Q Transform (CQT) : correspond à l'oreille humaine
    cqt = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=84, tuning=tuning))
    
    # 2. Pondération A (A-weighting) sur les fréquences CQT
    freqs = librosa.cqt_frequencies(n_bins=84, fmin=librosa.note_to_hz('C1'))
    weights = librosa.A_weighting(freqs)
    cqt_weighted = cqt * librosa.db_to_amplitude(weights[:, np.newaxis])
    
    # 3. Chroma à partir du CQT pondéré
    chroma = librosa.feature.chroma_cqt(C=cqt_weighted, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], 
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
    
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score: 
                best_score, res_key = score, f"{NOTES[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse Psychoacoustique...", max_entries=10)
def get_full_analysis(file_bytes, file_name):
    # Charge à 22050Hz pour économiser la RAM sur Streamlit Cloud
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    
    # HPSS : On garde uniquement la partie harmonique (mélodique) pour la clé
    y_harm = librosa.effects.hpss(y)[0]
    
    duration = librosa.get_duration(y=y, sr=sr)
    timeline_data, votes, all_chromas = [], [], []
    
    # Analyse par segments de 10s
    for start_t in range(0, int(duration) - 10, 10):
        y_seg = y_harm[int(start_t*sr):int((start_t+10)*sr)]
        if len(y_seg) < 1000: continue
        
        key_seg, score_seg, chroma_vec = analyze_segment_perceptual(y_seg, sr, tuning=tuning_offset)
        votes.append(key_seg)
        all_chromas.append(chroma_vec)
        timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": round(float(score_seg) * 100, 1)})
    
    # Résultats statistiques
    counts = Counter(votes)
    dominante_vote = counts.most_common(1)[0][0]
    dominante_conf = int((counts[dominante_vote] / len(votes)) * 100)
    
    # Synthèse globale par moyenne des chromas
    avg_chroma_global = np.mean(all_chromas, axis=0)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    best_synth_score, tonique_synth = -1, ""
    for mode in ["major", "minor"]:
        profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88] if mode=="major" else [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        for i in range(12):
            score = np.corrcoef(avg_chroma_global, np.roll(profile, i))[0, 1]
            if score > best_synth_score: best_synth_score, tonique_synth = score, f"{NOTES[i]} {mode}"

    # Calcul Energie/BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # Loudness simplifié basé sur le spectrogramme
    S = np.abs(librosa.stft(y, n_fft=2048))
    rms = librosa.feature.rms(S=S)
    energy_score = int(np.clip(np.mean(rms)*60 + (float(tempo)/160), 1, 10))

    # Classement stabilité
    df_tl = pd.DataFrame(timeline_data)
    n1 = dominante_vote; c1_val = dominante_conf; n2 = "??"; c2_val = 0
    if not df_tl.empty:
        df_s = df_tl.sort_values(by="Confiance", ascending=False)
        n1 = df_s.iloc[0]['Note']; c1_val = df_s.iloc[0]['Confiance']
        for _, row in df_s.iterrows():
            if row['Note'] != n1:
                n2 = row['Note']; c2_val = row['Confiance']; break

    # Décision finale (recommandation)
    candidates = [{"n": dominante_vote, "c": dominante_conf}, {"n": tonique_synth, "c": int(best_synth_score*100)}, {"n": n1, "c": c1_val}]
    final_res = max(candidates, key=lambda x: x['c'])

    return {
        "file_name": file_name, "vote": dominante_vote, "vote_conf": dominante_conf, 
        "synthese": tonique_synth, "confidence": int(best_synth_score*100), "tempo": int(float(tempo)), 
        "energy": energy_score, "timeline": timeline_data, "n1": n1, "c1": c1_val, "n2": n2, "c2": c2_val, "recommended": final_res
    }

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 KEY V6.2 PERCEPTUAL</h1>", unsafe_allow_html=True)

files = st.file_uploader("📂 ANALYSE PSYCHOACOUSTIQUE", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)
tabs = st.tabs(["📁 ANALYSEUR", "🕒 HISTORIQUE"])

with tabs[0]:
    if files:
        for f in files:
            file_id = f"{f.name}_{f.size}"
            if file_id not in st.session_state.processed_files:
                with st.spinner(f"Analyse humaine : {f.name}"):
                    f_bytes = f.read()
                    res = get_full_analysis(f_bytes, f.name)
                    
                    tg_caption = (
                        f"🎵 {res['file_name']}\n"
                        f"🥁 BPM: {res['tempo']} | E: {res['energy']}/10\n"
                        f"━━━━━━━━━━━━━━━\n"
                        f"👂 OREILLE: {res['recommended']['n']} ({get_camelot_pro(res['recommended']['n'])}) • {res['recommended']['c']}%\n"
                        f"💎 SYNTHÈSE: {res['synthese']} • {res['confidence']}%\n"
                        f"📊 DOMINANTE: {res['vote']} • {res['vote_conf']}%"
                    )
                    upload_to_telegram(io.BytesIO(f_bytes), f.name, tg_caption)
                    st.session_state.processed_files[file_id] = res
                    st.session_state.order_list.insert(0, file_id)

        for fid in st.session_state.order_list[:10]:
            res = st.session_state.processed_files[fid]
            with st.expander(f"🎵 {res['file_name']}", expanded=True):
                st.markdown(f"""
                    <div class="final-decision-box">
                        <div style="font-size: 1em; opacity: 0.8; letter-spacing: 1px;">NOTE RECOMMANDÉE (PERCEPTION)</div>
                        <div style="font-size: 3.8em; font-weight: 900; margin: 5px 0; line-height:1;">{res['recommended']['n']}</div>
                        <div style="font-size: 1.6em; font-weight: 700; color: #F1C40F;">{get_camelot_pro(res['recommended']['n'])} • {res['recommended']['c']}% FIABILITÉ</div>
                    </div>
                """, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">DOMINANTE</div><div class="value-custom">{res["vote"]}</div><div>{get_camelot_pro(res["vote"])} • {res["vote_conf"]}%</div></div>', unsafe_allow_html=True)
                    get_sine_witness(res["vote"], f"dom_{fid}")
                with c2:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">SYNTHÈSE</div><div class="value-custom">{res["synthese"]}</div><div>{get_camelot_pro(res["synthese"])} • {res["confidence"]}%</div></div>', unsafe_allow_html=True)
                    get_sine_witness(res["synthese"], f"synth_{fid}")
                with c3:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">STABILITÉ</div><div style="font-size:0.85em; margin-top:5px;">🥇 {res["n1"]} <b>{res["c1"]}%</b></div><div style="font-size:0.85em;">🥈 {res["n2"]} <b>{res["c2"]}%</b></div></div>', unsafe_allow_html=True)
                    col_s1, col_s2 = st.columns(2)
                    with col_s1: get_sine_witness(res["n1"], f"s1_{fid}")
                    with col_s2: get_sine_witness(res["n2"], f"s2_{fid}")
                with c4:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">BPM & ENERGIE</div><div class="value-custom">{res["tempo"]}</div><div>E. Perceptuelle: {res["energy"]}/10</div></div>', unsafe_allow_html=True)

                st.plotly_chart(px.scatter(pd.DataFrame(res['timeline']), x="Temps", y="Note", color="Confiance", size="Confiance", template="plotly_white"), use_container_width=True)

with tabs[1]:
    if st.session_state.processed_files:
        hist_data = [{"Fichier": r["file_name"], "Note": r["recommended"]["n"], "Camelot": get_camelot_pro(r["recommended"]["n"]), "BPM": r["tempo"]} for r in st.session_state.processed_files.values()]
        st.dataframe(pd.DataFrame(hist_data), use_container_width=True)
