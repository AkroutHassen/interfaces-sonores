# app.py
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

# Page configuration
st.set_page_config(page_title="Audio Analyzer", layout="wide")
st.title("🎵 Audio File Visualization")

# Function to analyze audio
def analyze_audio(y, sr):
    """Analyze audio features using the provided time series and sample rate."""
    duration = librosa.get_duration(y=y, sr=sr)
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]  # Returns an array, take first element
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    return {
        "duration": duration,
        "tempo": tempo,
        "spectral_centroid": spectral_centroid,
        "rms": rms,
        "zero_crossing_rate": zero_crossing_rate
    }

# Streamlit interface
uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3" if uploaded_file.type == "audio/mpeg" else ".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Load audio with librosa
    y, sr = librosa.load(tmp_path, mono=False, duration=30)
    
    # Determine number of channels
    channels = "Mono" if y.ndim == 1 else "Stereo"
    
    # Display basic metadata
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Metadata")
        st.write(f"Format: {uploaded_file.type}")
        st.write(f"Channels: {channels}")
        st.write(f"Sample Rate: {sr} Hz")
    
    # Perform advanced analysis
    analysis = analyze_audio(y, sr)
    
    with col2:
        st.subheader("📊 Audio Features")
        st.write(f"Duration: {analysis['duration']:.2f} seconds")
        st.write(f"Tempo: {analysis['tempo']:.2f} BPM")
        st.write(f"Spectral Centroid: {analysis['spectral_centroid']:.2f} Hz")
        st.write(f"RMS Volume: {analysis['rms']:.2f}")
        st.write(f"Zero Crossing Rate: {analysis['zero_crossing_rate']:.2f}")

    # Visualizations
    st.subheader("🎨 Visualizations")
    
    # Waveform
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    if y.ndim == 1:
        librosa.display.waveshow(y, sr=sr, ax=ax1)
    else:
        # Plot each channel for stereo audio
        for i in range(y.shape[0]):
            librosa.display.waveshow(y[i], sr=sr, ax=ax1, alpha=0.5, label=f"Channel {i+1}")
        ax1.legend()
    ax1.set_title("Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)
    
    # Spectrogram
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    if y.ndim == 1:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    else:
        # For stereo, compute spectrogram on the mean of channels
        y_mono = np.mean(y, axis=0)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_mono)), ref=np.max)
    img = librosa.display.specshow(D, y_axis="log", x_axis="time", sr=sr, ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Spectrogram")
    st.pyplot(fig2)
    
    # Play the audio
    st.subheader("🎧 Listen to the File")
    st.audio(uploaded_file)
    
    # Clean up temporary file
    os.unlink(tmp_path)

else:
    st.info("⬆️ Please upload an audio file to start the analysis")