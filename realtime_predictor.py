import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
from config import SAMPLE_RATE, DURATION, N_FFT, HOP_LENGTH, N_MELS, CLASES

def grabar_audio():
    print("🎙 Grabando audio de 1 segundo...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio.flatten()

def extraer_mfcc(audio):
    if len(audio) < int(SAMPLE_RATE * DURATION):
        audio = np.pad(audio, (0, int(SAMPLE_RATE * DURATION) - len(audio)))
    else:
        audio = audio[:int(SAMPLE_RATE * DURATION)]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return mfcc.T  # (63, 13)

def preparar_entrada(mel_db):
    mel_db = mel_db[np.newaxis, ..., np.newaxis]  # Forma: (1, alto, ancho, 1)
    return mel_db.astype(np.float32)

@tf.function(reduce_retracing=True)
def predecir(modelo, entrada):
    return modelo(entrada, training=False)

def main():
    modelo = tf.keras.models.load_model("modelo_entrenado.h5")
    print("✅ Modelo cargado. Presiona ENTER para predecir una grabación o Ctrl+C para salir.\n")

    while True:
        input("Presiona ENTER para grabar...")
        audio = grabar_audio()
        mfcc = extraer_mfcc(audio)
        entrada = preparar_entrada(mfcc)

        pred = predecir(modelo, entrada).numpy()
        clase_idx = np.argmax(pred)
        confianza = np.max(pred)

        print(f"🔊 Predicción: {CLASES[clase_idx]} ({confianza*100:.2f}% de confianza)\n")
        print(f"Forma de entrada para predicción: {entrada.shape}")