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
st.title("🎵 Audio File Visualization")

def analyze_audio(y, sr):
    """Analyze audio features using the provided time series and sample rate."""
    # If stereo, average the channels to convert to mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Initialize feature dictionary
    features = {}
    
    # Length of the audio signal
    features["length"] = len(y)

    # Chromagram
    chroma_hop_length = 512
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=chroma_hop_length)
    features["chroma_stft_mean"] = chromagram.mean()
    features["chroma_stft_var"] = chromagram.var()

    # Root Mean Square Energy
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = rms.mean()
    features["rms_var"] = rms.var()

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = spectral_centroid.mean()
    features["spectral_centroid_var"] = spectral_centroid.var()

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = spectral_bandwidth.mean()
    features["spectral_bandwidth_var"] = spectral_bandwidth.var()

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["rolloff_mean"] = rolloff.mean()
    features["rolloff_var"] = rolloff.var()

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    features["zero_crossing_rate_mean"] = zero_crossing_rate.mean()
    features["zero_crossing_rate_var"] = zero_crossing_rate.var()

    # Harmonics and Perceptual Features
    harmony, perceptr = librosa.effects.hpss(y=y)
    features["harmony_mean"] = harmony.mean()
    features["harmony_var"] = harmony.var()
    features["perceptr_mean"] = perceptr.mean()
    features["perceptr_var"] = perceptr.var()

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(tempo)

    # MFCCs (Mean and Variance)
    num_mfcc = 20
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    for i in range(num_mfcc):
        features[f"mfcc{i+1}_mean"] = mfcc[i].mean()
        features[f"mfcc{i+1}_var"] = mfcc[i].var()

    return features

def build_features_array(features):
    """Build the feature array in the same order as used in the model training."""
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
    # Append MFCC features (mean and variance for each)
    for i in range(20):
        feature_list.append(features[f"mfcc{i+1}_mean"])
        feature_list.append(features[f"mfcc{i+1}_var"])
    
    return np.array(feature_list).reshape(1, -1)

# Streamlit interface
uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3" if uploaded_file.type == "audio/mpeg" else ".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Load audio with librosa
    y, sr = librosa.load(tmp_path, mono=False)  # Allow multi-channel
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
    
    # Perform advanced analysis on the entire track
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
        y_mono = np.mean(y, axis=0)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_mono)), ref=np.max)
    img = librosa.display.specshow(D, y_axis="log", x_axis="time", sr=sr, ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Spectrogram")
    st.pyplot(fig2)
    
    # Play the audio
    st.subheader("🎧 Listen to the File")
    st.audio(uploaded_file)
    
    # Load the prediction model
    with open("xgb_model2.pkl", "rb") as model_file:
        model = joblib.load(model_file)
    
    # Use the features from the full track for an overall prediction
    features = analysis
    st.subheader("📋 Features Comparison: Extracted vs CSV File")
    st.write("Below is the comparison of feature values extracted from the audio file and the corresponding values from the CSV file:")

    # Convert the extracted features dictionary to a DataFrame
    extracted_features_df = pd.DataFrame(features.items(), columns=["Feature", "Extracted Value"])

    # Load the CSV file and filter the row for the uploaded file
    csv_file_path = "features_30_sec.csv"  # Update the path if necessary
    features_csv = pd.read_csv(csv_file_path)
    selected_row = features_csv[features_csv["filename"] == "blues.00000.wav"]

    # Convert the selected row to a dictionary for comparison
    if not selected_row.empty:
        csv_features_dict = selected_row.iloc[0].to_dict()
        csv_features_df = pd.DataFrame(csv_features_dict.items(), columns=["Feature", "CSV Value"])
        
        # Merge the extracted features and CSV features into a single DataFrame
        comparison_df = pd.merge(extracted_features_df, csv_features_df, on="Feature", how="left")
        
        # Display the comparison table
        st.dataframe(comparison_df)
    else:
        st.warning("The file `blues.00000.wav` was not found in the CSV file.")

    features_array = build_features_array(features)
    st.write("Overall Features Array:")
    st.write(features_array)
    
    # Predict probabilities for all classes on the full track
    probabilities = model.predict_proba(features_array, validate_features=False)[0]
    class_labels = model.classes_
    st.write(f"Class labels: {class_labels}")

    st.subheader("🎯 Predictions for All Classes")
    all_predictions_df = pd.DataFrame({
        "Genre": class_labels,
        "Probability (%)": probabilities * 100
    }).sort_values(by="Probability (%)", ascending=False)
    st.dataframe(all_predictions_df)
    
    top_5 = sorted(zip(class_labels, probabilities), key=lambda x: x[1], reverse=True)[:5]
    st.subheader("🎼 Top 5 Predicted Music Styles")
    for genre, prob in top_5:
        st.write(f"**{genre}**: {prob * 100:.2f}%")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    genres_top5, probs_top5 = zip(*top_5)
    ax.pie(probs_top5, labels=genres_top5, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
    ax.axis('equal')
    ax.set_title("Top 5 Predicted Music Styles")
    st.pyplot(fig)
    
    #############################################################################
    st.subheader("⏱️ Genre Prediction Over Time")
    st.write("The track is split into 10-second segments and the predicted genre probabilities are plotted over time.")

    # For segmentation, use mono audio
    audio_mono = librosa.to_mono(y) if y.ndim > 1 else y
    chunk_duration = 10  # seconds
    num_samples_per_chunk = int(chunk_duration * sr)
    num_chunks = int(math.ceil(len(audio_mono) / num_samples_per_chunk))

    time_points = []  # midpoints of each chunk
    predictions_list = []  # list of predicted probabilities arrays

    # Loop over each chunk
    for i in range(num_chunks):
        start = i * num_samples_per_chunk
        end = min((i + 1) * num_samples_per_chunk, len(audio_mono))
        chunk = audio_mono[start:end]
        # Pad the last chunk if it's shorter than 10 seconds
        if len(chunk) < num_samples_per_chunk:
            pad_width = num_samples_per_chunk - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), 'constant')
        # Extract features for the chunk and build the features array
        chunk_features = analyze_audio(chunk, sr)
        features_array_chunk = build_features_array(chunk_features)
        # Predict probabilities for the chunk
        chunk_probabilities = model.predict_proba(features_array_chunk, validate_features=False)[0]
        predictions_list.append(chunk_probabilities)
        # The midpoint of the chunk (e.g., 5 sec for the first chunk, 15 sec for the second, etc.)
        midpoint = (start + end) / 2 / sr
        time_points.append(midpoint)
    
    # Convert the list of predictions into a DataFrame for plotting
    predictions_df = pd.DataFrame(predictions_list, columns=class_labels)
    predictions_df["Time (s)"] = time_points

    # Plot the probabilities over time for each genre
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for genre in class_labels:
        ax3.plot(predictions_df["Time (s)"], predictions_df[genre] * 100, marker='o', label=genre)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Probability (%)")
    ax3.set_title("Genre Predictions Over Time")
    ax3.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    st.pyplot(fig3)
    
    # Optionally, display the raw prediction data
    st.subheader("Prediction Data (per 10-second segment)")
    st.dataframe(predictions_df)
    
    # Clean up temporary file
    os.unlink(tmp_path)

else:
    st.info("⬆️ Please upload an audio file to start the analysis")
