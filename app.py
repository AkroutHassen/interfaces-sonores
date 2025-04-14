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

# Configuration de la page
st.set_page_config(page_title="Analyseur Audio", layout="wide")
st.title("🎵 Analyseur Audio")

# ------------------------------
# Barre latérale : Téléchargement de fichier, lecteur audio et bouton de prédiction
# ------------------------------
st.sidebar.header("Téléchargement et Contrôles")
uploaded_file = st.sidebar.file_uploader("Téléchargez un fichier audio (MP3/WAV)", type=["wav", "mp3"])

# Barre latérale : Sélection de la durée des segments
st.sidebar.subheader("Durée des segments")
chunk_duration = st.sidebar.slider(
    "Sélectionnez la durée des segments (en secondes) :",
    min_value=1,
    max_value=30,
    value=10,  # Valeur par défaut
    step=1
)

# Initialisation du drapeau de prédiction
do_predictions = False

if uploaded_file is not None:
    # Sauvegarder temporairement le fichier téléchargé
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3" if uploaded_file.type == "audio/mpeg" else ".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Charger l'audio avec librosa
    y, sr = librosa.load(tmp_path, mono=False, sr=None)
    # Convertir en mono si nécessaire pour le traitement
    audio_mono = librosa.to_mono(y) if y.ndim > 1 else y

    # Lecteur audio dans la barre latérale
    st.sidebar.audio(uploaded_file)
    
    # Bouton de prédiction dans la barre latérale
    if st.sidebar.button("Faire des prédictions"):
        do_predictions = True

    if st.sidebar.button("Visualisations"):
        do_predictions = False

# ------------------------------
# Fonctions auxiliaires
# ------------------------------
def analyze_audio(y, sr):
    """Extraire les caractéristiques audio d'une série temporelle et d'une fréquence d'échantillonnage."""
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

# Continuez à traduire les autres sections de votre code de manière similaire.

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
    # Affichage des métadonnées
    st.subheader("📋 Métadonnées")
    st.write(f"**Nom du fichier :** {uploaded_file.name}")
    st.write(f"**Taille du fichier :** {uploaded_file.size/1024:.2f} KB")
    st.write(f"**Format :** {uploaded_file.type}")
    st.write(f"**Canaux :** {channels}")
    st.write(f"**Fréquence d'échantillonnage :** {sr} Hz")
    st.write(f"**Durée :** {duration:.2f} secondes")
    st.write(f"**Nombre d'échantillons :** {len(y)}")

    # Full track feature analysis
    full_features = analyze_audio(y, sr)
    if(do_predictions == False):
        # Waveform Visualization
        st.subheader("🎨 Visualizations")
        st.markdown("### Visualisation de la forme d'onde")
        st.markdown("""
        La forme d'onde montre comment l'amplitude du signal audio varie dans le temps.  
        L'axe X représente le temps (en secondes) et l'axe Y représente l'amplitude du signal.
        """)
        
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        if y.ndim == 1:
            librosa.display.waveshow(y, sr=sr, ax=ax1)
        else:
            for i in range(y.shape[0]):
                librosa.display.waveshow(y[i], sr=sr, ax=ax1, alpha=0.5, label=f"Channel {i+1}")
            ax1.legend()
        ax1.set_title("Waveform")
        ax1.set_xlabel("Temp (s)")
        ax1.set_ylabel("Amplitude")
        st.pyplot(fig1)

        # Fourier Transform Visualization Title and Description
        st.markdown("### Visualisation de la Transformée de Fourier")
        st.markdown("""
        La Transformée de Fourier décompose le signal en ses composantes fréquentielles.  
        Ce graphique montre les fréquences présentes et leur intensité.
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
        st.markdown("### Visualisation du Spectrogramme")
        st.markdown("""
        Le spectrogramme représente le contenu fréquentiel du signal audio dans le temps.  
        L'axe X montre le temps, l'axe Y montre les fréquences, et l'intensité des couleurs représente l'amplitude.
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
        ax2.set_xlabel("Temps (s)")
        st.pyplot(fig2)

        # Mel Spectrogram Visualization Title and Description
        st.markdown("### Visualisation du Spectrogramme Mel")
        st.markdown("""
        Le spectrogramme Mel utilise une échelle perceptuelle des fréquences, adaptée à la perception humaine.  
        Il est utile pour l'analyse de la parole et de la musique.
        """)
        if y.ndim == 2:

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
            ax1.set_ylabel("Mel")
            fig.colorbar(img_left, ax=ax1, format="%+2.0f dB")

            # Right channel plot
            img_right = librosa.display.specshow(S_right_db, y_axis='mel', x_axis='time', sr=sr, ax=ax2, cmap="cool")
            ax2.set_title("Right Channel Mel Spectrogram", fontsize=14)
            ax2.set_ylabel("Mel")
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
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Mel")
            ax.set_title("Mel Spectrogram", fontsize=14)

            # Display the plot in Streamlit
            st.pyplot(fig)


        # Visualisation des Caractéristiques Chroma
        st.markdown("### Visualisation des Caractéristiques Chroma")
        st.markdown("""
        Les caractéristiques Chroma représentent les 12 classes de hauteur musicale (par exemple, Do, Ré, Mi, etc.).  
        Cette visualisation montre comment l'intensité de ces classes varie dans le temps.
        """)

        if y.ndim == 2:  # Stéréo
            # Calcul des caractéristiques Chroma pour les deux canaux
            chroma_left = librosa.feature.chroma_stft(y=y[0], sr=sr)
            chroma_right = librosa.feature.chroma_stft(y=y[1], sr=sr)

            # Création des sous-graphiques pour les canaux gauche et droit
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Canal gauche
            img_left = librosa.display.specshow(chroma_left, y_axis='chroma', x_axis='time', sr=sr, ax=ax1, cmap="cool")
            ax1.set_title("Caractéristiques Chroma - Canal Gauche", fontsize=14)
            ax1.set_xlabel("Temps (s)")
            ax1.set_ylabel("Classe de Hauteur (Do, Ré, Mi, etc.)")
            fig.colorbar(img_left, ax=ax1, format="%+2.0f dB")

            # Canal droit
            img_right = librosa.display.specshow(chroma_right, y_axis='chroma', x_axis='time', sr=sr, ax=ax2, cmap="cool")
            ax2.set_title("Caractéristiques Chroma - Canal Droit", fontsize=14)
            ax2.set_xlabel("Temps (s)")
            ax2.set_ylabel("Classe de Hauteur (Do, Ré, Mi, etc.)")
            fig.colorbar(img_right, ax=ax2, format="%+2.0f dB")

            # Affichage des graphiques dans Streamlit
            st.pyplot(fig)

        else:  # Mono
            # Calcul des caractéristiques Chroma pour l'audio mono
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=ax, cmap="cool")
            ax.set_title("Caractéristiques Chroma", fontsize=14)
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Classe de Hauteur (Do, Ré, Mi, etc.)")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

            # Affichage du graphique dans Streamlit
            st.pyplot(fig)

        # Zero-Crossing Rate Visualization Title and Description
        st.markdown("### Visualisation du Taux de Passage par Zéro")
        st.markdown("""
        Le taux de passage par zéro mesure la fréquence à laquelle le signal audio change de signe.  
        Il est utilisé pour différencier les sons vocaux et non vocaux, ou les signaux bruyants et propres.
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
            ax1.set_xlabel("Temps (s)")
            ax1.set_ylabel("Zero Crossing Rate")
            
            # Right channel plot
            ax2.plot(librosa.times_like(zcr_right), zcr_right[0], color='r')
            ax2.set_title("Zero Crossing Rate - Right Channel", fontsize=14)
            ax2.set_xlabel("Temps (s)")
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
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Zero Crossing Rate")
            
            # Display the plot in Streamlit
            st.pyplot(fig)
        
        st.markdown("### Visualisation du Centroïde Spectral")
        st.markdown("""
        Le centroïde spectral indique la "brillance" d'un son. Un centroïde plus élevé signifie que l'énergie est davantage concentrée dans les hautes fréquences.
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
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Hz")
        st.pyplot(fig)


    
        

        st.markdown("### Séparation des Sources Harmoniques et Percussives")
        st.markdown("""
        - La **composante harmonique** contient les sons avec hauteur définie (par exemple, voix, instruments).
        - La **composante percussive** contient les transitoires et les battements (par exemple, percussions).
        """)
        st.markdown("""
        Ce graphique superpose les composantes harmoniques (bleu) et percussives (orange) du signal audio.  
        Il est utile pour visualiser comment chaque composante contribue au signal global.
        """)
        if y.ndim == 2:
            y = np.mean(y, axis=0)

        # Separate harmonic and percussive signals
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Plot both on the same graph
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y_harmonic, sr=sr, alpha=0.6, label="Harmonic", color="blue", ax=ax)
        librosa.display.waveshow(y_percussive, sr=sr, alpha=0.6, label="Percussive", color="orange", ax=ax)
        ax.set(title="Harmonic and Percussive Signals", xlabel="Temps (s)", ylabel="Amplitude")
        ax.legend()
        st.pyplot(fig)

    
        st.markdown("## Transformée Constant-Q (CQT)")
        st.markdown("""
        **🎧 Caractéristique perceptuelle (CQT) :**  
        La CQT utilise une échelle logarithmique des fréquences, similaire à la façon dont nous percevons la hauteur.  
        Elle est utile pour analyser le contenu harmonique dans la musique.
        """)
        # Perceptual Feature: Constant-Q Transform
        CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)

        fig_cqt, ax_cqt = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(CQT, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax_cqt, cmap='magma')
        ax_cqt.set(title='Constant-Q Transform (CQT)')
        fig_cqt.colorbar(img, ax=ax_cqt, format="%+2.0f dB")
        st.pyplot(fig_cqt)

        st.markdown("### Suivi du Tempo et des Battements")
        st.markdown("""
        Le tempo correspond à la vitesse des battements dans la musique, mesurée en BPM (Battements Par Minute).  
        Les lignes rouges en pointillés indiquent les battements estimés dans votre audio.
        """)

        if y.ndim == 2:
            y = np.mean(y, axis=0)

        # Estimate tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        st.write(f"Tempo: {tempo[0]:.2f} BPM")

        # st.markdown("#### Détection des Battements")
        st.markdown("""
        Cette visualisation montre les **battements** détectés dans la forme d'onde, avec des **cercles rouges** autour de chaque battement dans la plage de temps spécifiée.  
        Vous pouvez ajuster le **temps de début** et le **temps de fin** pour zoomer sur une section spécifique de l'audio.
        """)
        if y.ndim == 2:
            y = np.mean(y, axis=0)

        # Total duration of the audio
        total_duration = librosa.get_duration(y=y, sr=sr)

        start_time = st.number_input("Temps de début (en secondes) :", min_value=0.0, max_value=total_duration, value=0.0, step=0.1)
        end_time = st.number_input("Temps de fin (en secondes) :", min_value=start_time, max_value=total_duration, value=total_duration, step=0.1)        
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
        ax.set(title="Forme d'onde avec les battements mis en évidence", xlabel="Temps (s)", ylabel="Amplitude")
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
            with open("results/xgb_model6.pkl", "rb") as model_file:
                model = joblib.load(model_file)

            # Overall Prediction on full track
            features_array = build_features_array(full_features)
            overall_probabilities = model.predict_proba(features_array, validate_features=False)[0]
            class_labels = model.classes_

            st.subheader("🎯 Prévisions générales")
            overall_df = pd.DataFrame({
                "Genre": class_labels,
                "Probability (%)": overall_probabilities * 100
            }).sort_values(by="Probability (%)", ascending=False)
            st.dataframe(overall_df)

            # ------------------------------
            # Genre Prediction Over Time
            # ------------------------------
            st.subheader("⏱️ Prédictions des Genres au Fil du Temps")
            st.write(f"La piste est divisée en segments de {chunk_duration} secondes, et les prédictions sont tracées au fil du temps.")            
            num_samples_per_chunk = int(chunk_duration * sr)
            num_chunks = int(math.ceil(len(audio_mono) / num_samples_per_chunk))

            time_points = [] 
            predictions_list = []  

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
            ax3.set_xlabel("Temps (s)")
            ax3.set_ylabel("Probabilité (%)")
            ax3.set_title("Prédictions des Genres au Fil du Temps")
            ax3.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
            st.pyplot(fig3)

            st.subheader("Données de Prédiction (par segment)")
            st.dataframe(predictions_df)

    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
else:
    st.info("⬆️ Veuillez télécharger un fichier audio depuis la barre latérale pour commencer l'analyse.")