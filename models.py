"""
Gestión de modelos SAM2
Maneja la carga y gestión del estado de los modelos.
"""
import os
from sam2.build_sam import build_sam2
from config import MODELS_INFO, DEVICE

# Estado global del modelo
_current_model = None
_current_model_name = None


def get_current_model():
    """Obtiene el modelo actualmente cargado"""
    return _current_model


def get_current_model_name():
    """Obtiene el nombre del modelo actualmente cargado"""
    return _current_model_name


def is_model_loaded(model_name):
    """Verifica si un modelo específico está cargado"""
    return _current_model_name == model_name and _current_model is not None


def load_model(model_name):
    """
    Carga el modelo SAM2 seleccionado.
    
    Args:
        model_name: Nombre del modelo a cargar
        
    Returns:
        str: Mensaje de estado de la operación
    """
    global _current_model, _current_model_name

    print(f"[load_model] Pedido cargar modelo: {model_name}")
    
    try:
        # Verificar si el modelo ya está cargado
        if is_model_loaded(model_name):
            msg = f"✅ Modelo '{model_name}' ya está cargado (device={DEVICE.upper()})"
            print("[load_model]", msg)
            return msg

        # Obtener información del modelo
        if model_name not in MODELS_INFO:
            msg = f"❌ Modelo '{model_name}' no encontrado en la configuración"
            print("[load_model]", msg)
            return msg

        model_info = MODELS_INFO[model_name]
        cfg_path = model_info["config"]
        ckpt_path = model_info["checkpoint"]

        print(f"[load_model] Config: {cfg_path}")
        print(f"[load_model] Checkpoint: {ckpt_path}")

        # Verificar que existen los archivos
        if not os.path.exists(cfg_path):
            msg = f"❌ No se encontró el archivo de configuración: {cfg_path}"
            print("[load_model]", msg)
            return msg

        if not os.path.exists(ckpt_path):
            msg = f"❌ No se encontró el checkpoint: {ckpt_path}"
            print("[load_model]", msg)
            return msg

        print(f"[load_model] Cargando modelo en {DEVICE.upper()} ...")

        # Cargar el modelo
        sam2_model = build_sam2(
            cfg_path,
            ckpt_path,
            device=DEVICE
        )

        # Actualizar estado global
        _current_model = sam2_model
        _current_model_name = model_name

        msg = f"✅ Modelo '{model_name}' cargado correctamente en {DEVICE.upper()}"
        print("[load_model]", msg)
        return msg

    except Exception as e:
        msg = f"❌ Error al cargar modelo: {repr(e)}"
        print("[load_model]", msg)
        import traceback
        traceback.print_exc()
        return msg


def ensure_model_loaded(model_name):
    """
    Asegura que el modelo especificado esté cargado.
    Si no lo está, intenta cargarlo.
    
    Args:
        model_name: Nombre del modelo a asegurar
        
    Returns:
        tuple: (success: bool, message: str)
    """
    if is_model_loaded(model_name):
        return True, f"Modelo '{model_name}' ya está cargado"
    
    status = load_model(model_name)
    if "Error" in status or "❌" in status:
        return False, status
    
    return True, status

