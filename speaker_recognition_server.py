"""
Dhwani Speaker Recognition Server
Uses Pyannote.audio for speaker embedding and recognition
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from pyannote.audio import Model, Inference
import torch
import io
import wave
import json
from datetime import datetime
import sqlite3
from scipy.spatial.distance import cosine

app = Flask(__name__)
CORS(app)  # Allow Flutter app to connect

# ============================================================
# CONFIGURATION
# ============================================================

# Get Pyannote auth token from environment variable
PYANNOTE_AUTH_TOKEN = os.environ.get('PYANNOTE_AUTH_TOKEN', '')

# Database file
DB_FILE = 'speakers.db'

# Model configuration
MODEL_NAME = "pyannote/embedding"
EMBEDDING_DIM = 512

# Recognition threshold (lower = stricter)
SIMILARITY_THRESHOLD = 0.7

# ============================================================
# INITIALIZE PYANNOTE MODEL
# ============================================================

print("🔊 Loading Pyannote model...")
try:
    # Load the embedding model
    model = Model.from_pretrained(
        MODEL_NAME,
        use_auth_token=PYANNOTE_AUTH_TOKEN
    )
    inference = Inference(model, window="whole")
    print("✅ Pyannote model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("⚠️ Make sure PYANNOTE_AUTH_TOKEN is set correctly")
    model = None
    inference = None

# ============================================================
# DATABASE SETUP
# ============================================================

def init_database():
    """Initialize SQLite database for storing speaker embeddings"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            emoji TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

init_database()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def bytes_to_audio(audio_bytes):
    """Convert audio bytes to numpy array"""
    try:
        # Read WAV file from bytes
        with io.BytesIO(audio_bytes) as audio_io:
            with wave.open(audio_io, 'rb') as wav_file:
                # Get audio parameters
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Read audio data
                audio_data = wav_file.readframes(n_frames)
                
                # Convert to numpy array
                if sample_width == 2:  # 16-bit
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                else:
                    audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                
                # Convert to float32 and normalize
                audio_array = audio_array.astype(np.float32) / 32768.0
                
                # Convert to mono if stereo
                if n_channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                
                return audio_array, framerate
    except Exception as e:
        print(f"❌ Error converting audio: {e}")
        return None, None

def get_embedding(audio_array, sample_rate):
    """Extract speaker embedding from audio"""
    if inference is None:
        return None
    
    try:
        # Convert to torch tensor
        waveform = torch.from_numpy(audio_array).float()
        
        # Get embedding
        embedding = inference({"waveform": waveform, "sample_rate": sample_rate})
        
        # Convert to numpy array
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"❌ Error extracting embedding: {e}")
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def save_speaker(name, embedding, emoji='👤'):
    """Save speaker to database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Convert embedding to bytes
        embedding_bytes = embedding.tobytes()
        
        # Insert or update speaker
        cursor.execute('''
            INSERT INTO speakers (name, embedding, emoji, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                embedding = excluded.embedding,
                emoji = excluded.emoji,
                updated_at = excluded.updated_at
        ''', (name, embedding_bytes, emoji, datetime.now()))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"❌ Error saving speaker: {e}")
        return False
    finally:
        conn.close()

def get_all_speakers():
    """Get all enrolled speakers"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT name, emoji, created_at FROM speakers ORDER BY name')
    speakers = cursor.fetchall()
    
    conn.close()
    
    return [
        {
            'name': name,
            'emoji': emoji,
            'enrolledAt': created_at
        }
        for name, emoji, created_at in speakers
    ]

def recognize_speaker(embedding):
    """Recognize speaker from embedding"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT name, embedding, emoji FROM speakers')
    speakers = cursor.fetchall()
    
    conn.close()
    
    if not speakers:
        return None, 0.0
    
    best_match = None
    best_similarity = 0.0
    
    for name, stored_embedding_bytes, emoji in speakers:
        # Convert bytes back to numpy array
        stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32)
        
        # Calculate similarity
        similarity = calculate_similarity(embedding, stored_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = {'name': name, 'emoji': emoji}
    
    # Check if similarity meets threshold
    if best_similarity >= SIMILARITY_THRESHOLD:
        return best_match, best_similarity
    else:
        return None, best_similarity

def delete_speaker(name):
    """Delete speaker from database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM speakers WHERE name = ?', (name,))
    deleted = cursor.rowcount > 0
    
    conn.commit()
    conn.close()
    
    return deleted

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/enroll', methods=['POST'])
def enroll_speaker():
    """
    Enroll a new speaker
    
    Expected: multipart/form-data
    - audio: WAV file (16-bit, mono/stereo, 16kHz recommended)
    - name: Speaker name
    - emoji: Optional emoji (default: 👤)
    """
    if inference is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    name = request.form.get('name', '').strip()
    emoji = request.form.get('emoji', '👤')
    
    if not name:
        return jsonify({'error': 'Speaker name required'}), 400
    
    print(f"📝 Enrolling speaker: {name}")
    
    # Read audio
    audio_bytes = audio_file.read()
    audio_array, sample_rate = bytes_to_audio(audio_bytes)
    
    if audio_array is None:
        return jsonify({'error': 'Invalid audio format'}), 400
    
    # Check audio length (minimum 3 seconds)
    duration = len(audio_array) / sample_rate
    if duration < 3:
        return jsonify({'error': 'Audio too short (minimum 3 seconds)'}), 400
    
    print(f"🎵 Audio duration: {duration:.2f}s, sample rate: {sample_rate}Hz")
    
    # Extract embedding
    embedding = get_embedding(audio_array, sample_rate)
    
    if embedding is None:
        return jsonify({'error': 'Failed to extract embedding'}), 500
    
    print(f"🎯 Embedding shape: {embedding.shape}")
    
    # Save to database
    success = save_speaker(name, embedding, emoji)
    
    if not success:
        return jsonify({'error': 'Failed to save speaker'}), 500
    
    print(f"✅ Speaker {name} enrolled successfully!")
    
    return jsonify({
        'success': True,
        'name': name,
        'emoji': emoji,
        'duration': duration,
        'embedding_dim': len(embedding)
    })

@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Recognize a speaker from audio
    
    Expected: multipart/form-data
    - audio: WAV file (16-bit, mono/stereo, 16kHz recommended)
    """
    if inference is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Read audio
    audio_bytes = audio_file.read()
    audio_array, sample_rate = bytes_to_audio(audio_bytes)
    
    if audio_array is None:
        return jsonify({'error': 'Invalid audio format'}), 400
    
    # Extract embedding
    embedding = get_embedding(audio_array, sample_rate)
    
    if embedding is None:
        return jsonify({'error': 'Failed to extract embedding'}), 500
    
    # Recognize speaker
    match, confidence = recognize_speaker(embedding)
    
    if match:
        print(f"✅ Recognized: {match['name']} ({confidence:.2%})")
        return jsonify({
            'identified': True,
            'name': match['name'],
            'emoji': match['emoji'],
            'confidence': float(confidence)
        })
    else:
        print(f"❓ Unknown speaker (best match: {confidence:.2%})")
        return jsonify({
            'identified': False,
            'confidence': float(confidence)
        })

@app.route('/speakers', methods=['GET'])
def list_speakers():
    """Get list of all enrolled speakers"""
    speakers = get_all_speakers()
    
    return jsonify({
        'speakers': speakers,
        'count': len(speakers)
    })

@app.route('/speakers/<name>', methods=['DELETE'])
def delete_speaker_endpoint(name):
    """Delete a speaker"""
    success = delete_speaker(name)
    
    if success:
        print(f"🗑️ Deleted speaker: {name}")
        return jsonify({'success': True, 'name': name})
    else:
        return jsonify({'error': 'Speaker not found'}), 404

@app.route('/speakers/reset', methods=['POST'])
def reset_speakers():
    """Delete all speakers"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM speakers')
    deleted_count = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    print(f"🗑️ Deleted all {deleted_count} speakers")
    
    return jsonify({'success': True, 'deleted_count': deleted_count})

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Dhwani Speaker Recognition Server on port {port}")
    print(f"📍 Endpoints:")
    print(f"   - POST /enroll - Enroll new speaker")
    print(f"   - POST /recognize - Recognize speaker")
    print(f"   - GET /speakers - List all speakers")
    print(f"   - DELETE /speakers/<name> - Delete speaker")
    print(f"   - GET /health - Health check")
    
    app.run(host='0.0.0.0', port=port, debug=False)