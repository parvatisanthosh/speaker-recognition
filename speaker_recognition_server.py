"""
Dhwani Speaker Recognition Server
Uses Pyannote.audio for speaker embedding and recognition
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import torch
import io
import wave
from datetime import datetime
import sqlite3
from scipy.spatial.distance import cosine

from pyannote.audio import Model, Inference

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

PYANNOTE_AUTH_TOKEN = os.environ.get("PYANNOTE_AUTH_TOKEN", "")
DB_FILE = "speakers.db"

MODEL_NAME = "pyannote/embedding"
SIMILARITY_THRESHOLD = 0.7

# ============================================================
# LOAD MODEL
# ============================================================

print("🔊 Loading Pyannote model...")

model = None
inference = None

try:
    model = Model.from_pretrained(
        MODEL_NAME,
        use_auth_token=PYANNOTE_AUTH_TOKEN
    )
    inference = Inference(model, window="whole")
    print("✅ Pyannote model loaded")
except Exception as e:
    print("❌ Model load failed:", e)

# ============================================================
# DATABASE
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            name TEXT PRIMARY KEY,
            embedding BLOB,
            emoji TEXT,
            updated_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============================================================
# AUDIO HELPERS
# ============================================================

def bytes_to_audio(audio_bytes):
    try:
        with io.BytesIO(audio_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                sr = wf.getframerate()
                audio = wf.readframes(wf.getnframes())
                audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                return audio, sr
    except Exception as e:
        print("❌ Audio decode error:", e)
        return None, None

def get_embedding(audio, sr):
    try:
        waveform = torch.tensor(audio)
        if sr != 16000:
            import torchaudio.transforms as T
            waveform = T.Resample(sr, 16000)(waveform)
        emb = inference({"waveform": waveform, "sample_rate": 16000})
        return emb.cpu().numpy()
    except Exception as e:
        print("❌ Embedding error:", e)
        return None

def similarity(a, b):
    return 1 - cosine(a, b)

# ============================================================
# API
# ============================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": inference is not None
    })

@app.route("/enroll", methods=["POST"])
def enroll():
    if inference is None:
        return jsonify({"error": "model not loaded"}), 500

    audio = request.files.get("audio")
    name = request.form.get("name")
    emoji = request.form.get("emoji", "👤")

    if not audio or not name:
        return jsonify({"error": "audio and name required"}), 400

    data, sr = bytes_to_audio(audio.read())
    emb = get_embedding(data, sr)

    if emb is None:
        return jsonify({"error": "embedding failed"}), 500

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "REPLACE INTO speakers VALUES (?, ?, ?, ?)",
        (name, emb.tobytes(), emoji, datetime.now())
    )
    conn.commit()
    conn.close()

    return jsonify({"success": True, "name": name})

@app.route("/recognize", methods=["POST"])
def recognize():
    if inference is None:
        return jsonify({"error": "model not loaded"}), 500

    audio = request.files.get("audio")
    data, sr = bytes_to_audio(audio.read())
    emb = get_embedding(data, sr)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name, embedding, emoji FROM speakers")
    rows = c.fetchall()
    conn.close()

    best, best_score = None, 0.0

    for name, blob, emoji in rows:
        stored = np.frombuffer(blob, dtype=np.float32)
        score = similarity(emb, stored)
        if score > best_score:
            best_score = score
            best = {"name": name, "emoji": emoji}

    if best and best_score >= SIMILARITY_THRESHOLD:
        return jsonify({"identified": True, **best, "confidence": best_score})

    return jsonify({"identified": False, "confidence": best_score})

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
