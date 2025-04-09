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
from scipy.signal import find_peaks

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
    y, sr = librosa.load(tmp_path, mono=False, sr=None)
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
    st.markdown("### Waveform Visualization")
    st.markdown("""
The waveform shows how the amplitude of the audio signal varies over time. Each point on the plot corresponds to a sample in the audio file. The X-axis represents time in seconds, while the Y-axis represents the amplitude of the signal. 
If the audio contains multiple channels, each channel's waveform will be plotted separately.
""")
    
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

    # Fourier Transform Visualization Title and Description
    st.markdown("### Fourier Transform Visualization")
    st.markdown("""
    The Fourier Transform decomposes the signal into its frequency components.  
    This graph shows which frequencies are present and their intensity.
    """)
    if y.ndim == 2:
        y = np.mean(y, axis=0)

    # Compute the FFT
    n = len(y)
    fft = np.fft.fft(y)
    fft_mag = np.abs(fft)[:n // 2]  # Keep only positive frequencies
    freqs = np.fft.fftfreq(n, d=1/sr)[:n // 2]

    # Plot the FFT
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, fft_mag, color='c')
    ax.set_title("Fourier Transform (Frequency Domain)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, sr // 2)
    st.pyplot(fig)


    # Spectrogram Visualization
    # Spectrogram Visualization Title and Description
    st.markdown("### Spectrogram Visualization")
    st.markdown("""
    The spectrogram represents the frequency content of the audio signal over time. It is generated by applying the Short-Time Fourier Transform (STFT) to the audio signal. The X-axis shows time, while the Y-axis shows frequency. The color intensity represents the magnitude of the frequencies at each point in time. 
    The spectrogram is often used to visualize the frequency spectrum and how it changes over time, and it is commonly used in music and speech processing.
    """)
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

    # Mel Spectrogram Visualization Title and Description
    st.markdown("### Mel Spectrogram Visualization")
    st.markdown("""
    The Mel spectrogram is a representation of the audio signal's frequency content, but it uses a Mel scale instead of the linear frequency scale. The Mel scale is a perceptual scale of pitches that approximates the way humans perceive sound. This visualization is useful for speech and music analysis.
    """)
    if y.ndim == 2:
        st.write("Stereo audio detected. Processing each channel separately.")

        # Process the left and right channels separately
        S_left = librosa.feature.melspectrogram(y=y[0], sr=sr)  # Left channel (index 0)
        S_right = librosa.feature.melspectrogram(y=y[1], sr=sr)  # Right channel (index 1)

        # Convert both to decibels (log scale)
        S_left_db = librosa.power_to_db(S_left, ref=np.max)
        S_right_db = librosa.power_to_db(S_right, ref=np.max)

        # Create subplots for left and right channels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left channel plot
        img_left = librosa.display.specshow(S_left_db, y_axis='mel', x_axis='time', sr=sr, ax=ax1, cmap="cool")
        ax1.set_title("Left Channel Mel Spectrogram", fontsize=14)
        fig.colorbar(img_left, ax=ax1, format="%+2.0f dB")

        # Right channel plot
        img_right = librosa.display.specshow(S_right_db, y_axis='mel', x_axis='time', sr=sr, ax=ax2, cmap="cool")
        ax2.set_title("Right Channel Mel Spectrogram", fontsize=14)
        fig.colorbar(img_right, ax=ax2, format="%+2.0f dB")

        # Display the plots in Streamlit
        st.pyplot(fig)

    else:
        # If mono audio (y is 1D), compute and display the spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Plot the Mel Spectrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_db, y_axis='mel', x_axis='time', sr=sr, ax=ax, cmap="cool")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel Spectrogram", fontsize=14)

        # Display the plot in Streamlit
        st.pyplot(fig)


    # Chroma Feature Visualization Title and Description
    st.markdown("### Chroma Feature Visualization")
    st.markdown("""
    Chroma features represent the 12 different pitch classes in music (e.g., A, B, C, etc.). This visualization shows how the intensity of these pitch classes varies over time and is useful for analyzing harmony and chord progressions.
    """)

    if y.ndim == 2:
        # Stereo audio: y has shape (2, n_samples), process each channel separately
        st.write("Stereo audio detected. Processing each channel separately.")
        
        # Compute Chroma features for both channels (left and right)
        chroma_left = librosa.feature.chroma_stft(y=y[0], sr=sr)
        chroma_right = librosa.feature.chroma_stft(y=y[1], sr=sr)

        # Plot Chroma Features for Left and Right Channels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left channel plot
        librosa.display.specshow(chroma_left, y_axis='chroma', x_axis='time', ax=ax1, cmap="cool")
        ax1.set_title("Chroma Features - Left Channel", fontsize=14)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pitch Class")
        fig.colorbar(librosa.display.specshow(chroma_left, ax=ax1, cmap="cool"), ax=ax1, format="%+2.0f dB")

        # Right channel plot
        librosa.display.specshow(chroma_right, y_axis='chroma', x_axis='time', ax=ax2, cmap="cool")
        ax2.set_title("Chroma Features - Right Channel", fontsize=14)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Pitch Class")
        fig.colorbar(librosa.display.specshow(chroma_right, ax=ax2, cmap="cool"), ax=ax2, format="%+2.0f dB")

        # Display the plots in Streamlit
        st.pyplot(fig)

    else:       
        # Compute Chroma features for mono audio
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # Plot Chroma Features
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=ax, cmap="cool")
        ax.set_title("Chroma Features", fontsize=14)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pitch Class")
        fig.colorbar(librosa.display.specshow(chroma, ax=ax, cmap="cool"), ax=ax, format="%+2.0f dB")

        # Display the plot in Streamlit
        st.pyplot(fig)

    # Zero-Crossing Rate Visualization Title and Description
    st.markdown("### Zero-Crossing Rate Visualization")
    st.markdown("""
    The zero-crossing rate is the rate at which the audio signal changes its sign (crosses zero). This is a simple feature used in speech and music analysis to differentiate between voiced and unvoiced speech, or between noisy and clean signals.
    """)

    if y.ndim == 2:
        # Stereo audio: y has shape (2, n_samples), process each channel separately       
        # Compute Zero Crossing Rate for both channels (left and right)
        zcr_left = librosa.feature.zero_crossing_rate(y=y[0])
        zcr_right = librosa.feature.zero_crossing_rate(y=y[1])

        # Plot Zero Crossing Rate for Left and Right Channels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left channel plot
        ax1.plot(librosa.times_like(zcr_left), zcr_left[0], color='b')
        ax1.set_title("Zero Crossing Rate - Left Channel", fontsize=14)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Zero Crossing Rate")
        
        # Right channel plot
        ax2.plot(librosa.times_like(zcr_right), zcr_right[0], color='r')
        ax2.set_title("Zero Crossing Rate - Right Channel", fontsize=14)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Zero Crossing Rate")

        # Display the plots in Streamlit
        st.pyplot(fig)

    else:
        # Compute Zero Crossing Rate for mono audio
        zcr = librosa.feature.zero_crossing_rate(y=y)

        # Plot Zero Crossing Rate
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(librosa.times_like(zcr), zcr[0], color='g')
        ax.set_title("Zero Crossing Rate", fontsize=14)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Zero Crossing Rate")
        
        # Display the plot in Streamlit
        st.pyplot(fig)
    
    st.markdown("### spectral Centroid Visualization")
    st.markdown("""
    The spectral centroid indicates the "brightness" of a sound. A higher centroid means that the energy is more concentrated in higher frequencies.
    """)
    if y.ndim == 2:
        y_mono = np.mean(y, axis=0)
    else:
        y_mono = y

    # Compute Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)

    # Time axis for plotting
    times = librosa.times_like(spectral_centroid)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, spectral_centroid[0], color='m')
    ax.set_title("Spectral Centroid Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    st.pyplot(fig)


   
    

    st.markdown("### Harmonic-Percussive Source Separation")
    st.markdown("""
    - The **harmonic component** contains pitched sounds (e.g., vocals, instruments).
    - The **percussive component** contains transients and beats (e.g., drums).
    """)
    st.markdown("""
    This graph overlays the harmonic (blue) and percussive (orange) components of the audio.  
    It's useful for visualizing how each component contributes to the overall signal.
    """)
    if y.ndim == 2:
        y = np.mean(y, axis=0)

    # Separate harmonic and percussive signals
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Plot both on the same graph
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y_harmonic, sr=sr, alpha=0.6, label="Harmonic", color="blue", ax=ax)
    librosa.display.waveshow(y_percussive, sr=sr, alpha=0.6, label="Percussive", color="orange", ax=ax)
    ax.set(title="Harmonic and Percussive Signals", xlabel="Time (s)", ylabel="Amplitude")
    ax.legend()
    st.pyplot(fig)

   
    st.markdown("## Constant-Q Transform (CQT)")
    st.markdown("""
    **🎧 Perceptual Feature (CQT):**  
    CQT gives a logarithmic frequency scale similar to how we perceive pitch, making it useful for analyzing harmonic content in music.
    """)
    # Perceptual Feature: Constant-Q Transform
    CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)

    fig_cqt, ax_cqt = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(CQT, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax_cqt, cmap='magma')
    ax_cqt.set(title='Constant-Q Transform (CQT)')
    fig_cqt.colorbar(img, ax=ax_cqt, format="%+2.0f dB")
    st.pyplot(fig_cqt)

    st.markdown("### Tempo and Beat Tracking")
    st.markdown("""
    Tempo is the speed of the beat in music, measured in BPM (Beats Per Minute).  
    The red dashed lines indicate the estimated beats in your audio.
    """)

    if y.ndim == 2:
        y = np.mean(y, axis=0)

    # Estimate tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    st.write(f"Tempo: {tempo[0]:.2f} BPM")

    st.markdown("#### Beat Detection")
    st.markdown("""
    This visualization shows the **beats** detected within the waveform, with **red circles** around each beat in the specified time range.  
    You can adjust the **start time** and **end time** to zoom into a specific section of the audio.
    """)
    if y.ndim == 2:
        y = np.mean(y, axis=0)

    # Total duration of the audio
    total_duration = librosa.get_duration(y=y, sr=sr)

    start_time = st.number_input("Start time (in seconds):", min_value=0.0, max_value=total_duration, value=0.0, step=0.1)
    end_time = st.number_input("End time (in seconds):", min_value=start_time, max_value=total_duration, value=total_duration, step=0.1)
    end_time = min(end_time, total_duration)  # Ensure end time does not exceed total duration
    start_sample = int(start_time * sr)
    end_sample = int((end_time+1) * sr)
    y_cut = y[start_sample:end_sample]

    onset_env = librosa.onset.onset_strength(y=y_cut, sr=sr)
    peaks, _ = find_peaks(onset_env, height=0.1)  # Adjustable height parameter

    fig, ax = plt.subplots(figsize=(10, 4))

    librosa.display.waveshow(y_cut, sr=sr, ax=ax)

    for peak in peaks:
        peak_time = librosa.frames_to_time(peak, sr=sr)  # Convert frame to time
        if start_time <= peak_time <= end_time:  # Ensure beat is within the time range
            ax.plot(peak_time, y_cut[peak], 'ro', markersize=8)  # Red dot at the beat position
            ax.add_patch(plt.Circle((peak_time, y_cut[peak]), radius=0.02, color='r', fill=False, linewidth=2))  # Circle around the beat

    # Set labels and title
    ax.set(title="Waveform with Beats Highlighted", xlabel="Time (s)", ylabel="Amplitude")

    # Adjust the x-axis to start from start_time and end at end_time
    ax.set_xlim(start_time, end_time)

    # Show the plot
    st.pyplot(fig)
    

    


    

    


    

    




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
