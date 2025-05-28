from modelo import entrenar_modelo
from preprocesador import cargar_dataset
import tensorflow as tf


def entrenar_y_guardar_modelo():
    print("Cargando dataset...")
    X, y = cargar_dataset()
    print(f"Dataset cargado con {len(X)} muestras.")

    print("Entrenando modelo...")
    modelo = entrenar_modelo(X, y, epochs=30, batch_size=8)

    print("Guardando modelo entrenado...")
    modelo.save('modelo_entrenado.h5')
    print("Modelo guardado en 'modelo_entrenado.h5'.")

def main():
    entrenar_y_guardar_modelo()

if __name__ == "__main__":
    main()

