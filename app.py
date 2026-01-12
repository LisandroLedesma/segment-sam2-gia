"""
Punto de entrada principal de la aplicación SAM2.1
Inicializa y lanza la interfaz de usuario.
"""
import os
from config import MODELS_INFO, DEVICE, SERVER_CONFIG
from ui import create_interface


def verify_files():
    """Verifica que los archivos de configuración y checkpoints existan"""
    print("\n📋 Verificando archivos...")
    for model_name, info in MODELS_INFO.items():
        cfg_exists = "✅" if os.path.exists(info["config"]) else "❌"
        ckpt_exists = "✅" if os.path.exists(info["checkpoint"]) else "❌"
        print(f"  {model_name}:")
        print(f"    Config: {cfg_exists} {info['config']}")
        print(f"    Checkpoint: {ckpt_exists} {info['checkpoint']}")


def main():
    """Función principal que inicia la aplicación"""
    print("="*70)
    print("🚀 Iniciando SAM 2.1 - Segmentación de Imágenes")
    print("="*70)
    print(f"📱 Dispositivo: {DEVICE.upper()}")
    print(f"🧠 Modelos disponibles: {len(MODELS_INFO)}")
    print(f"🔍 Directorio actual: {os.getcwd()}")
    print("="*70)
    
    # Verificar archivos antes de iniciar
    verify_files()
    
    print("\n" + "="*70)
    
    # Crear y lanzar interfaz
    demo = create_interface()
    demo.launch(**SERVER_CONFIG, debug=True)


if __name__ == "__main__":
    main()
