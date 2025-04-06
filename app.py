# app.py
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import joblib
import pandas as pd
import math

# Page configuration
st.set_page_config(page_title="Audio Analyzer", layout="wide")
st.title("🎵 Audio Analyzer")

# ------------------------------
# Sidebar: File Upload, Audio Player & Prediction Button
# ------------------------------
st.sidebar.header("Upload & Controls")
uploaded_file = st.sidebar.file_uploader("Upload an audio file (MP3/WAV)", type=["wav", "mp3"])

# Sidebar: Chunk Duration Selection
st.sidebar.subheader("Chunk Duration")
chunk_duration = st.sidebar.slider(
    "Select the chunk duration (in seconds):",
    min_value=1,
    max_value=30,
    value=10,  # Default value
    step=1
)

# Initialize prediction flag
do_predictions = False

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3" if uploaded_file.type == "audio/mpeg" else ".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load audio with librosa
    y, sr = librosa.load(tmp_path, mono=False)
    # Convert to mono if needed for processing over time
    audio_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # Sidebar Audio Player
    st.sidebar.audio(uploaded_file)
    
    # Sidebar Prediction Button
    if st.sidebar.button("Make Predictions"):
        do_predictions = True

# ------------------------------
# Helper Functions
# ------------------------------
def analyze_audio(y, sr):
    """Extract audio features from a time series and sample rate."""
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    features = {}
    features["length"] = len(y)
    chroma_hop_length = 512
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=chroma_hop_length)
    features["chroma_stft_mean"] = chromagram.mean()
    features["chroma_stft_var"] = chromagram.var()
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = rms.mean()
    features["rms_var"] = rms.var()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = spectral_centroid.mean()
    features["spectral_centroid_var"] = spectral_centroid.var()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = spectral_bandwidth.mean()
    features["spectral_bandwidth_var"] = spectral_bandwidth.var()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["rolloff_mean"] = rolloff.mean()
    features["rolloff_var"] = rolloff.var()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    features["zero_crossing_rate_mean"] = zero_crossing_rate.mean()
    features["zero_crossing_rate_var"] = zero_crossing_rate.var()
    harmony, perceptr = librosa.effects.hpss(y=y)
    features["harmony_mean"] = harmony.mean()
    features["harmony_var"] = harmony.var()
    features["perceptr_mean"] = perceptr.mean()
    features["perceptr_var"] = perceptr.var()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(tempo)
    num_mfcc = 20
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    for i in range(num_mfcc):
        features[f"mfcc{i+1}_mean"] = mfcc[i].mean()
        features[f"mfcc{i+1}_var"] = mfcc[i].var()
    return features

def build_features_array(features):
    """Build a features array in the same order used during model training."""
    feature_list = [
        features["length"],
        features["chroma_stft_mean"],
        features["chroma_stft_var"],
        features["rms_mean"],
        features["rms_var"],
        features["spectral_centroid_mean"],
        features["spectral_centroid_var"],
        features["spectral_bandwidth_mean"],
        features["spectral_bandwidth_var"],
        features["rolloff_mean"],
        features["rolloff_var"],
        features["zero_crossing_rate_mean"],
        features["zero_crossing_rate_var"],
        features["harmony_mean"],
        features["harmony_var"],
        features["perceptr_mean"],
        features["perceptr_var"],
        features["tempo"],
    ]
    for i in range(20):
        feature_list.append(features[f"mfcc{i+1}_mean"])
        feature_list.append(features[f"mfcc{i+1}_var"])
    return np.array(feature_list).reshape(1, -1)

# ------------------------------
# Main Page: Visualizations & Data Display
# ------------------------------
if uploaded_file is not None:
    # Determine audio channel info
    channels = "Mono" if y.ndim == 1 else "Stereo"
    duration = librosa.get_duration(y=y, sr=sr)

    # Display Metadata
    st.subheader("📋 Metadata")
    st.write(f"**File Name:** {uploaded_file.name}")
    st.write(f"**File Size:** {uploaded_file.size/1024:.2f} KB")
    st.write(f"**Format:** {uploaded_file.type}")
    st.write(f"**Channels:** {channels}")
    st.write(f"**Sample Rate:** {sr} Hz")
    st.write(f"**Duration:** {duration:.2f} seconds")
    st.write(f"**Number of Samples:** {len(y)}")

    # Full track feature analysis
    full_features = analyze_audio(y, sr)

    # Waveform Visualization
    st.subheader("🎨 Visualizations")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    if y.ndim == 1:
        librosa.display.waveshow(y, sr=sr, ax=ax1)
    else:
        for i in range(y.shape[0]):
            librosa.display.waveshow(y[i], sr=sr, ax=ax1, alpha=0.5, label=f"Channel {i+1}")
        ax1.legend()
    ax1.set_title("Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # Spectrogram Visualization
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    if y.ndim == 1:
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    else:
        y_mono = np.mean(y, axis=0)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_mono)), ref=np.max)
    img = librosa.display.specshow(D, y_axis="log", x_axis="time", sr=sr, ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Spectrogram")
    st.pyplot(fig2)

    # ------------------------------
    # Prediction Section (Triggered from Sidebar)
    # ------------------------------
    if do_predictions:
        with st.spinner("Processing predictions..."):
            # Load the model (ensure xgb_model2.pkl is in the same directory)
            with open("xgb_model2.pkl", "rb") as model_file:
                model = joblib.load(model_file)

            # Overall Prediction on full track
            features_array = build_features_array(full_features)
            overall_probabilities = model.predict_proba(features_array, validate_features=False)[0]
            class_labels = model.classes_

            st.subheader("🎯 Overall Predictions")
            overall_df = pd.DataFrame({
                "Genre": class_labels,
                "Probability (%)": overall_probabilities * 100
            }).sort_values(by="Probability (%)", ascending=False)
            st.dataframe(overall_df)

            # ------------------------------
            # Genre Prediction Over Time
            # ------------------------------
            st.subheader("⏱️ Genre Predictions Over Time")
            st.write(f"The track is split into {chunk_duration}-second segments and predictions are plotted over time.")
            num_samples_per_chunk = int(chunk_duration * sr)
            num_chunks = int(math.ceil(len(audio_mono) / num_samples_per_chunk))

            time_points = []  # midpoints for each chunk
            predictions_list = []  # probabilities for each chunk

            for i in range(num_chunks):
                start = i * num_samples_per_chunk
                end = min((i + 1) * num_samples_per_chunk, len(audio_mono))
                chunk = audio_mono[start:end]
                if len(chunk) < num_samples_per_chunk:
                    pad_width = num_samples_per_chunk - len(chunk)
                    chunk = np.pad(chunk, (0, pad_width), 'constant')
                chunk_features = analyze_audio(chunk, sr)
                features_array_chunk = build_features_array(chunk_features)
                chunk_probabilities = model.predict_proba(features_array_chunk, validate_features=False)[0]
                predictions_list.append(chunk_probabilities)
                midpoint = (start + end) / 2 / sr  # midpoint in seconds
                time_points.append(midpoint)

            predictions_df = pd.DataFrame(predictions_list, columns=class_labels)
            predictions_df["Time (s)"] = time_points

            # Plot predictions over time for each genre
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            for genre in class_labels:
                ax3.plot(predictions_df["Time (s)"], predictions_df[genre] * 100,
                         marker='o', label=genre)
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Probability (%)")
            ax3.set_title("Genre Predictions Over Time")
            ax3.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
            st.pyplot(fig3)

            st.subheader("Prediction Data (per segment)")
            st.dataframe(predictions_df)

    # Clean up temporary file when done
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
else:
    st.info("⬆️ Please upload an audio file from the sidebar to start the analysis.")
