import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import tempfile
import base64
import os
import time
import matplotlib.pyplot as plt
import whisper
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
from io import BytesIO

# ======================
# SETUP
# ======================
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

model = tf.keras.models.load_model("emotion_model.keras", compile=False)
stt_model = whisper.load_model("base")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ======================
# SESSION STATE
# ======================
defaults = {
    "messages": [],
    "started": False,
    "processed": False,
    "audio_np": None,
    "user_text": "",
    "ask_music": False,
    "show_youtube": False,
    "music_prompt_said": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================
# TTS
# ======================
def speak(text):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def autoplay(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

# ======================
# WAVEFORM
# ======================
def plot_waveform(audio_np):
    fig, ax = plt.subplots()
    ax.plot(audio_np, linewidth=1)
    ax.set_title("Waveform Audio")
    st.pyplot(fig)

# ======================
# FEATURE EXTRACTION
# ======================
def extract_features(audio_np, sr=22050):

    audio_np = audio_np[:sr * 3]

    mel = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < 128:
        pad = 128 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')

    mel_db = mel_db[:, :128]
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    return mel_db

# ======================
# WHISPER
# ======================
def speech_to_text(path):
    return stt_model.transcribe(path)["text"]

st.title("🤖 Emotion Chatbot")

# ======================
# START
# ======================
if not st.session_state.started:

    if st.button("▶️ START"):
        st.session_state.started = True
        st.session_state.messages = []
        st.session_state.processed = False
        st.rerun()

    st.stop()

# ======================
# INTRO BOT
# ======================
if len(st.session_state.messages) == 0:
    msg = "How are you feeling?"
    st.session_state.messages.append({
        "role": "assistant",
        "content": msg
    })
    autoplay(speak(msg))

# ======================
# CHAT DISPLAY
# ======================
for i, m in enumerate(st.session_state.messages):

    with st.chat_message(m["role"]):
        st.write(m["content"])

        if "audio_bytes" in m:
            st.audio(m["audio_bytes"])

        if "audio_np" in m:
            if st.button("📊 Waveform", key=f"wave_{i}"):
                plot_waveform(m["audio_np"])

        if "probs" in m:

            if "show_probs" not in st.session_state:
                st.session_state.show_probs = {}

            if i not in st.session_state.show_probs:
                st.session_state.show_probs[i] = False

            if st.button("📊 Emotion Probabilities", key=f"prob_{i}"):
                st.session_state.show_probs[i] = not st.session_state.show_probs[i]

            if st.session_state.show_probs[i]:
                with st.expander("Confidence", expanded=True):
                    for idx, val in enumerate(m["probs"]):
                        st.write(f"{le.classes_[idx]}: {val*100:.2f}%")

# ======================
# INPUT
# ======================
if not st.session_state.processed:

    mode = st.radio("Input:", ["🎤 Record", "📁 Upload"])

    audio_np = None
    audio_bytes = None

    # RECORD
    if mode == "🎤 Record":

        audio = mic_recorder(
            start_prompt="🎤 Start",
            stop_prompt="⏹ Stop"
        )

        if audio and "bytes" in audio:

            audio_bytes = audio["bytes"]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                path = tmp.name

            text = speech_to_text(path)
            audio_np, sr = librosa.load(path, sr=22050)

    # UPLOAD
    else:

        uploaded = st.file_uploader("Upload WAV", type=["wav"])

        if uploaded:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.read())
                path = tmp.name

            text = speech_to_text(path)
            audio_np, sr = librosa.load(path, sr=22050)

            audio_bytes = uploaded.getvalue()

    # ======================
    # PREDICTION
    # ======================
    if audio_np is not None:

        st.session_state.audio_np = audio_np
        st.session_state.user_text = text

        features = extract_features(audio_np).reshape(1, 128, 128, 1)

        pred = model.predict(features, verbose=0)
        probs = pred[0]

        emotion = le.inverse_transform([np.argmax(pred[0])])[0]
        st.session_state.last_emotion = emotion

        response = f"You said: '{text}'. I think you are {emotion}"

        st.session_state.messages.append({
            "role": "user",
            "content": text,
            "audio_np": audio_np,
            "audio_bytes": audio_bytes
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "probs": probs
        })

        autoplay(speak(response))

        # ======================
        # ASK MUSIC (WITH VOICE)
        # ======================
        st.session_state.ask_music = True

        st.session_state.processed = True
        st.rerun()

# ======================
# MUSIC QUESTION + VOICE (CLEAN UI)
# ======================
if st.session_state.ask_music:

    if not st.session_state.music_prompt_said:
        prompt_text = "Do you want music recommendations from YouTube based on your mood?"
        autoplay(speak(prompt_text))
        st.session_state.music_prompt_said = True

    # CARD UI BIAR KONSISTEN
    st.markdown("""
    <div style="
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #2a2a2a;
        background-color: #111111;
        margin-top: 10px;
        margin-bottom: 10px;
    ">
        <h4 style="margin-bottom:10px;">🤖 AI Music Assistant</h4>
        <p style="margin:0; font-size:15px;">
            Reccomending music based on your mood can be a great way to enhance your emotional experience.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        if st.button("👍 Yes, show me music", use_container_width=True):
            st.session_state.show_youtube = True
            st.session_state.ask_music = False

    with col2:
        if st.button("👎 No thanks", use_container_width=True):
            st.session_state.ask_music = False
# ======================
# YOUTUBE POPUP
# ======================
from yt_dlp import YoutubeDL

def get_youtube_embed(query):
    ydl_opts = {
        'quiet': True,
        'skip_download': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f"ytsearch1:{query}", download=False)
        video = result['entries'][0]
        video_id = video['id']

    return f"https://www.youtube.com/embed/{video_id}"

# ======================
# YOUTUBE POPUP (DYNAMIC)
# ======================
if st.session_state.show_youtube:

    emotion = st.session_state.last_emotion

    # mapping emosi → keyword YouTube
    youtube_query_map = {
        "happy": "happy upbeat music playlist",
        "sad": "sad emotional piano music",
        "angry": "intense aggressive workout music",
        "neutral": "lofi chill relaxing music",
        "fear": "calming anxiety relief music",
        "surprise": "uplifting inspirational music"
    }

    query = youtube_query_map.get(emotion, "lofi music")

    # ambil video otomatis dari YouTube
    try:
        url = get_youtube_embed(query)
    except:
        url = "https://www.youtube.com/embed/5qap5aO4i9A"  # fallback

    with st.expander("🎵 Your Mood Music", expanded=True):

        st.markdown(f"""
        <iframe width="100%" height="400"
        src="{url}"
        frameborder="0"
        allow="autoplay; encrypted-media"
        allowfullscreen>
        </iframe>
        """, unsafe_allow_html=True)

        if st.button("Close"):
            st.session_state.show_youtube = False
            st.rerun()
# ======================
# RESET
# ======================
if st.session_state.started and st.session_state.processed:

    if st.button("🔄 RESET"):

        for k in defaults.keys():
            st.session_state[k] = defaults[k]

        st.rerun()