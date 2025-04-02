# app.py
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import joblib

# Page configuration
st.set_page_config(page_title="Audio Analyzer", layout="wide")
st.title("🎵 Audio File Visualization")


def analyze_audio(y, sr):
    """Analyze audio features using the provided time series and sample rate."""
    # If stereo, average the channels to convert to mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)  # Mean of each MFCC
    mfcc_vars = np.var(mfccs, axis=1)    # Variance of each MFCC

    # Extract chroma features
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)

    # Extract spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Extract tonnetz (tonal centroid features)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    # Extract tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # Extract perceptual features
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    perceptual_features = librosa.power_to_db(mel_spectrogram, ref=np.max)
    perceptr_mean = np.mean(perceptual_features)
    perceptr_var = np.var(perceptual_features)

    # Extract other features
    features = {
        "length": len(y),
        "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        "chroma_stft_var": np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        "rms_mean": np.mean(librosa.feature.rms(y=y)),
        "rms_var": np.var(librosa.feature.rms(y=y)),
        "spectral_centroid_mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_centroid_var": np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "spectral_bandwidth_var": np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "rolloff_var": np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "zero_crossing_rate_mean": np.mean(librosa.feature.zero_crossing_rate(y=y)),
        "zero_crossing_rate_var": np.var(librosa.feature.zero_crossing_rate(y=y)),
        "harmony_mean": np.mean(tonnetz),
        "harmony_var": np.var(tonnetz),
        "tempo": float(tempo),
        "chroma_cqt_mean": np.mean(chroma_cqt),
        "chroma_cqt_var": np.var(chroma_cqt),
        "chroma_cens_mean": np.mean(chroma_cens),
        "chroma_cens_var": np.var(chroma_cens),
        "spectral_contrast_mean": np.mean(spectral_contrast),
        "spectral_contrast_var": np.var(spectral_contrast),
        "perceptr_mean": perceptr_mean,
        "perceptr_var": perceptr_var,
    }

    # Add MFCC features
    num_mfccs = mfccs.shape[0]
    for i in range(num_mfccs):
        features[f"mfcc{i+1}_mean"] = mfcc_means[i]
        features[f"mfcc{i+1}_var"] = mfcc_vars[i]

    # Add spectral contrast features
    num_contrast_bands = spectral_contrast.shape[0]
    for i in range(num_contrast_bands):
        features[f"spectral_contrast_band{i+1}_mean"] = np.mean(spectral_contrast[i])
        features[f"spectral_contrast_band{i+1}_var"] = np.var(spectral_contrast[i])

    return features

# Streamlit interface
uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3" if uploaded_file.type == "audio/mpeg" else ".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Load audio with librosa
    y, sr = librosa.load(tmp_path, mono=False)
    
    # Determine number of channels
    channels = "Mono" if y.ndim == 1 else "Stereo"
    
    # Display basic metadata
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Metadata")
        st.write(f"File Name: {uploaded_file.name}")
        st.write(f"File Size: {uploaded_file.size / 1024:.2f} KB")
        st.write(f"Format: {uploaded_file.type}")
        st.write(f"Channels: {channels}")
        st.write(f"Sample Rate: {sr} Hz")
        st.write(f"Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
        st.write(f"Number of Samples: {len(y)}")
    
    # Perform advanced analysis
    analysis = analyze_audio(y, sr)
    
    with col2:
        st.subheader("📊 Audio Features")
        st.write(f"Length: {analysis['length']} samples")
        st.write(f"Tempo: {analysis['tempo']:.2f} BPM")
        st.write(f"Chroma STFT Mean: {analysis['chroma_stft_mean']:.2f}")
        st.write(f"Chroma STFT Variance: {analysis['chroma_stft_var']:.2f}")
        st.write(f"RMS Mean: {analysis['rms_mean']:.2f}")
        st.write(f"RMS Variance: {analysis['rms_var']:.2f}")
        st.write(f"Spectral Centroid Mean: {analysis['spectral_centroid_mean']:.2f} Hz")
        st.write(f"Spectral Centroid Variance: {analysis['spectral_centroid_var']:.2f}")
        st.write(f"Spectral Bandwidth Mean: {analysis['spectral_bandwidth_mean']:.2f} Hz")
        st.write(f"Spectral Bandwidth Variance: {analysis['spectral_bandwidth_var']:.2f}")
        st.write(f"Rolloff Mean: {analysis['rolloff_mean']:.2f} Hz")
        st.write(f"Rolloff Variance: {analysis['rolloff_var']:.2f}")
        st.write(f"Zero Crossing Rate Mean: {analysis['zero_crossing_rate_mean']:.2f}")
        st.write(f"Zero Crossing Rate Variance: {analysis['zero_crossing_rate_var']:.2f}")
        st.write(f"Harmony Mean: {analysis['harmony_mean']:.2f}")
        st.write(f"Harmony Variance: {analysis['harmony_var']:.2f}")

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
    
    with open("xgb_model.pkl", "rb") as model_file:
        model = joblib.load(model_file)
    
    if uploaded_file is not None:
        features = analyze_audio(y, sr)
        features_array = np.array([
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
            features["perceptr_mean"],  # Add perceptual mean
            features["perceptr_var"],  # Add perceptual variance
            features["tempo"],
            *[val for i in range(20) for val in (features[f"mfcc{i+1}_mean"], features[f"mfcc{i+1}_var"])],
            # *[features[f"spectral_contrast_band{i+1}_mean"] for i in range(7)],  # Add spectral contrast bands
            # *[features[f"spectral_contrast_band{i+1}_var"] for i in range(7)],   # Add spectral contrast bands
        ]).reshape(1, -1)
        # Predict probabilities for all classes
        probabilities = model.predict_proba(features_array,validate_features=False)[0]
        
        # Get class labels (genres)
        class_labels = model.classes_
        st.write(f"Class labels: {class_labels}")
        
        # Combine labels and probabilities, then sort by probability in descending order
        top_5 = sorted(zip(class_labels, probabilities), key=lambda x: x[1], reverse=True)[:5]
        
        # Extract genres and probabilities for the top 5 predictions
        genres, probs = zip(*top_5)

        # Display the top 5 predictions with probabilities
        st.subheader("🎼 Top 5 Predicted Music Styles")
        for genre, prob in top_5:
            st.write(f"**{genre}**: {prob * 100:.2f}%")
        
        fig, ax = plt.subplots( figsize=(8, 6))
        ax.pie(probs, labels=genres, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        ax.set_title("Top 5 Predicted Music Styles")

    # Display the pie chart in Streamlit
    st.pyplot(fig)
    
    # Clean up temporary file
    os.unlink(tmp_path)

else:
    st.info("⬆️ Please upload an audio file to start the analysis")