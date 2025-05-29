import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
import threading

from config import SAMPLE_RATE, DURATION, N_FFT, HOP_LENGTH, CLASES

def grabar_audio():
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio.flatten()

def extraer_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return mfcc.T  # Forma (frames, 13)

def preparar_entrada(mel_db):
    return mel_db[np.newaxis, ..., np.newaxis].astype(np.float32)

@tf.function(reduce_retracing=True)
def predecir(modelo, entrada):
    return modelo(entrada, training=False)

# ───── Interfaz gráfica ─────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de voz: ¿Quién dijo 'Hola'?")
        self.root.geometry("400x250")

        self.label_resultado = tk.Label(root, text="Presiona el botón para grabar", font=("Helvetica", 14))
        self.label_resultado.pack(pady=20)

        self.label_confianza = tk.Label(root, text="", font=("Helvetica", 12))
        self.label_confianza.pack()

        self.boton_grabar = tk.Button(root, text="🎙 Grabar", font=("Helvetica", 12), command=self.iniciar_prediccion)
        self.boton_grabar.pack(pady=20)

        self.modelo = tf.keras.models.load_model("modelo_entrenado.h5")

    def iniciar_prediccion(self):
        threading.Thread(target=self.procesar_audio).start()

    def procesar_audio(self):
        self.label_resultado.config(text="🎙 Grabando...")
        audio = grabar_audio()
        mel_db = extraer_mfcc(audio)


        # 🔧 Asegurar forma (63, 13)
        if mel_db.shape[0] < 63:
            mel_db = np.pad(mel_db, ((0, 63 - mel_db.shape[0]), (0, 0)))
        elif mel_db.shape[0] > 63:
            mel_db = mel_db[:63, :]

        entrada = preparar_entrada(mel_db)
        pred = predecir(self.modelo, entrada).numpy()
        idx = np.argmax(pred)
        confianza = np.max(pred)

        if confianza < 0.90:
            self.label_resultado.config(text="❌ No se reconoció la palabra 'Hola'.")
            self.label_confianza.config(text=f"Confianza: {confianza * 100:.2f}%")
        else:
            self.label_resultado.config(text=f"✅ {CLASES[idx]}")
            self.label_confianza.config(text=f"Confianza: {confianza * 100:.2f}%")

# ───── Main ─────
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Saliendo del programa...")
