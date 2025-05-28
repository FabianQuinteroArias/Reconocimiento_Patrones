from entrenar_modelo import entrenar_y_guardar_modelo
from realtime_predictor import main as predecir_en_tiempo_real
from grabador import grabar_muestras_persona  # importar la función corregida

def main():
    while True:
        print("Elige una opción:")
        print("1. Grabar audios para una persona")
        print("2. Entrenar y guardar modelo")
        print("3. Predecir en tiempo real")
        print("4. Salir")
        opcion = input("Opción: ")

        if opcion == '1':
            persona = input("Ingrese el nombre o ID de la persona que va a grabar: ")
            print(f"🎙 Grabando audios para la persona: {persona}")
            grabar_muestras_persona(persona, num_muestras=100)
            print("✅ Grabación finalizada.\n")

        elif opcion == '2':
            print("🧠 Cargando dataset y entrenando modelo...")
            entrenar_y_guardar_modelo()
            print("✅ Entrenamiento finalizado.\n")

        elif opcion == '3':
            print("🔍 Iniciando predicción en tiempo real...")
            predecir_en_tiempo_real()
            print("🎯 Predicción finalizada.\n")

        elif opcion == '4':
            print("👋 Saliendo del programa.")
            break
        else:
            print("⚠️ Opción inválida. Intenta de nuevo.\n")

if __name__ == "__main__":
    main()
