# app_lucky.py
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import os

# Configuration de la page
st.set_page_config(page_title="Lucky Norm Analyzer", layout="wide")
st.title("🎵 Analyse de Lucky Norm.wav")

# Chemin du fichier fixe
FILE_PATH = "lucky_norm.wav"

# Fonction d'analyse (identique)
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, duration=30)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    return {
        "signal": y,
        "sr": sr,
        "duration": duration,
        "tempo": tempo,
        "spectral_centroid": spectral_centroid,
        "rms": rms,
        "zero_crossing_rate": zero_crossing_rate
    }

if os.path.exists(FILE_PATH):
    # Lire les métadonnées
    audio = AudioSegment.from_wav(FILE_PATH)
    channels = "Mono" if audio.channels == 1 else "Stereo"
    
    # Afficher les métadonnées
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Métadonnées")
        st.write("Format: WAV")
        st.write(f"Canaux: {channels}")
        st.write(f"Sample Rate: {audio.frame_rate} Hz")
        st.write(f"Bit Depth: {audio.sample_width * 8} bits")
    
    # Analyse avancée
    analysis = analyze_audio(FILE_PATH)
    
    with col2:
        st.subheader("📊 Caractéristiques Audio")
        st.write(f"Durée: {analysis['duration']:.2f} secondes")
        st.write(f"Tempo: {analysis['tempo']:.2f} BPM")
        st.write(f"Centroid Spectral: {analysis['spectral_centroid']:.2f} Hz")
        st.write(f"Volume RMS: {analysis['rms']:.2f}")
        st.write(f"Zero Crossing Rate: {analysis['zero_crossing_rate']:.2f}")

    # Visualisations
    st.subheader("🎨 Visualisations")
    
    # Waveform
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(analysis['signal'], sr=analysis['sr'], ax=ax1)
    ax1.set_title("Forme d'Onde")
    st.pyplot(fig1)
    
    # Spectrogramme
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(analysis['signal'])), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=analysis['sr'], ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Spectrogramme")
    st.pyplot(fig2)
    
    # Jouer l'audio
    st.subheader("🎧 Écouter le Fichier")
    st.audio(FILE_PATH)

else:
    st.error("❌ Fichier non trouvé - Placez 'lucky_norm.wav' dans le même dossier que ce script")

