"""
Configuración del proyecto SAM2.1
Define constantes, información de modelos y configuración general.
"""
import os
import torch

# Configuración de dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Información de modelos disponibles
MODELS_INFO = {
    "Tiny (Rápido)": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
        "description": "Más rápido, menor precisión"
    },
    "Small (Balanceado)": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_small.pt",
        "description": "Balance entre velocidad y precisión"
    },
    "Base+ (Recomendado)": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_base_plus.pt",
        "description": "Buena precisión, velocidad aceptable"
    },
    "Large (Máxima calidad)": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
        "description": "Máxima precisión, más lento"
    }
}

# Configuración de servidor Gradio
SERVER_CONFIG = {
    "server_name": "127.0.0.1",
    "server_port": 7860,
    "share": False,
    "show_error": True
}

# Parámetros por defecto de segmentación
DEFAULT_SEGMENTATION_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95
}

