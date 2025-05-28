import librosa
import numpy as np
import os
from config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, N_SAMPLES, CLASES

def extraer_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mfcc = mfcc.T  # Transponer a [tiempo, características]
    return mfcc

def cargar_mfcc_desde_archivo(ruta_audio, n_mfcc=13):
    y, sr = librosa.load(ruta_audio, sr=SAMPLE_RATE)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))  # Rellenar
    else:
        y = y[:N_SAMPLES]  # Recortar
    mfcc = extraer_mfcc(y, sr, n_mfcc=n_mfcc)
    return mfcc

def cargar_dataset(ruta_base='dataset', n_mfcc=13):
    X, y = [], []
    for i, clase in enumerate(CLASES):
        carpeta = os.path.join(ruta_base, clase)
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.wav'):
                ruta = os.path.join(carpeta, archivo)
                mfcc = cargar_mfcc_desde_archivo(ruta, n_mfcc=n_mfcc)
                X.append(mfcc)
                y.append(i)
    X = np.array(X)[..., np.newaxis]  # (n, 63, 13, 1)
    y = np.array(y)
    return X, y
