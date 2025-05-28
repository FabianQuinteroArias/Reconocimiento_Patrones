import os
import sounddevice as sd
import soundfile as sf
import keyboard
from config import SAMPLE_RATE, DURATION

import os
import sounddevice as sd
import soundfile as sf
import keyboard
from config import SAMPLE_RATE, DURATION

def grabar_muestras_persona(persona, base_dir='dataset', num_muestras=100):
    clase_dir = os.path.join(base_dir, persona)
    try:
        os.makedirs(clase_dir, exist_ok=True)
        print(f"\n🎙 Carpeta para guardar: {clase_dir}")
    except Exception as e:
        print(f"Error creando carpeta: {e}")
        return

    for i in range(num_muestras):
        print(f"  Esperando tecla 'r' para grabar muestra {i+1}/{num_muestras}...")
        keyboard.wait('r')

        print("  Grabando 1 segundo...")
        try:
            audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()
            if audio is None or len(audio) == 0:
                print("  ¡Error! Audio vacío o nulo.")
                continue
        except Exception as e:
            print(f"  Error al grabar audio: {e}")
            continue

        filename = os.path.join(clase_dir, f"{persona}_{i+1}.wav")
        try:
            sf.write(filename, audio, SAMPLE_RATE)
            print(f"  Guardado: {filename}")
        except Exception as e:
            print(f"  Error al guardar archivo: {e}")

def main():
    persona = input("Ingrese el nombre o ID de la persona para grabar audios: ")
    grabar_muestras_persona(persona, num_muestras=100)

if __name__ == "__main__":
    main()