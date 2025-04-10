import os
import glob
import numpy as np
import pandas as pd
import librosa

# Définir le chemin vers le dossier principal qui contient les sous-dossiers par genre
base_dir = "/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original"

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

# Liste pour stocker les dictionnaires de caractéristiques
features_list = []

# Parcourir les sous-dossiers par genre
for genre in os.listdir(base_dir):
    genre_path = os.path.join(base_dir, genre)
    if os.path.isdir(genre_path):
        # Récupérer tous les fichiers wav dans le sous-dossier
        wav_files = glob.glob(os.path.join(genre_path, "*.wav"))
        for file in wav_files:
            try:
                y, sr = librosa.load(file)
                features = analyze_audio(y, sr)
                features['genre'] = genre
                features['filename'] = os.path.basename(file)
                features_list.append(features)
            except Exception as e:
                print(f"Erreur sur le fichier {file}: {e}")

features_df = pd.DataFrame(features_list)

features_df.to_csv("extracted_features.csv", index=False)

