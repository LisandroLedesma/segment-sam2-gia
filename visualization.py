"""
Funciones de visualización
Renderiza máscaras y resultados de segmentación.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io


def get_mask_colors(num_masks):
    """
    Genera colores vibrantes y únicos para cada máscara usando HSV.
    
    Args:
        num_masks: Número de máscaras
        
    Returns:
        list: Lista de colores RGB normalizados (0-1)
    """
    colors = []
    
    if num_masks == 0:
        return colors
    
    # Usar colores equidistantes en el espacio HSV para mejor diferenciación
    # H (Hue): 0-360 grados, distribuimos uniformemente
    # S (Saturation): 0.8-1.0 (alta saturación para colores vibrantes)
    # V (Value): 0.8-1.0 (alto brillo para colores radiantes)
    
    import colorsys
    
    for i in range(num_masks):
        # Distribuir el matiz (hue) uniformemente
        hue = (i * 360.0 / num_masks) % 360.0
        
        # Variar ligeramente la saturación y el valor para más variedad
        # pero mantenerlos altos para colores vibrantes
        saturation = 0.85 + (i % 3) * 0.05  # Entre 0.85 y 1.0
        value = 0.85 + ((i * 7) % 3) * 0.05  # Entre 0.85 y 1.0
        
        # Convertir HSV a RGB
        rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
        colors.append(np.array(rgb))
    
    return colors


def visualize_masks(image, masks, selected_indices=None, point_coords=None, point_labels=None):
    """
    Visualiza las máscaras sobre la imagen original usando numpy/PIL (optimizado).
    
    Args:
        image: Imagen original en formato numpy array (RGB)
        masks: Lista de diccionarios con información de máscaras
        selected_indices: Lista de índices de máscaras a mostrar. Si es None, muestra todas.
        point_coords: Lista opcional de coordenadas de puntos [(x, y), ...]
        point_labels: Lista opcional de etiquetas (1 para foreground, 0 para background)
        
    Returns:
        PIL.Image: Imagen con máscaras visualizadas
    """
    if masks is None or len(masks) == 0:
        # Si no hay máscaras pero hay puntos, dibujar los puntos
        if point_coords and point_labels and len(point_coords) > 0:
            result_image = image.copy()
            # Convertir a BGR para OpenCV
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            
            for coord, label in zip(point_coords, point_labels):
                x, y = int(coord[0]), int(coord[1])
                # Foreground: verde, Background: rojo
                color = (0, 255, 0) if label == 1 else (0, 0, 255)  # BGR: verde o rojo
                # Dibujar círculo relleno
                cv2.circle(result_bgr, (x, y), 8, color, -1)
                # Dibujar borde blanco
                cv2.circle(result_bgr, (x, y), 8, (255, 255, 255), 2)
            
            # Convertir de vuelta a RGB
            result_image = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_image)
        # Si no hay máscaras ni puntos, devolver la imagen original
        return Image.fromarray(image)
    
    # Ordenar máscaras por área (de mayor a menor) y mantener índices originales
    sorted_indices = sorted(range(len(masks)), key=lambda i: masks[i]['area'], reverse=True)
    sorted_masks = [masks[i] for i in sorted_indices]
    
    # Si no se especifican índices, mostrar todas
    if selected_indices is None:
        masks_to_show = sorted_masks
        mask_indices_to_show = sorted_indices
    else:
        # Filtrar máscaras según selección
        # selected_indices son los índices en la lista ordenada
        masks_to_show = [sorted_masks[i] for i in selected_indices if i < len(sorted_masks)]
        mask_indices_to_show = [sorted_indices[i] for i in selected_indices if i < len(sorted_indices)]
    
    if len(masks_to_show) == 0:
        # Si no hay máscaras para mostrar, devolver imagen original
        return Image.fromarray(image)
    
    # Convertir imagen a float para operaciones
    result = image.astype(np.float32) / 255.0
    
    # Generar colores para todas las máscaras (para mantener consistencia)
    all_colors = get_mask_colors(len(masks))
    
    # Superponer máscaras directamente sobre la imagen (mucho más rápido que matplotlib)
    for idx, mask_data in enumerate(masks_to_show):
        mask = mask_data['segmentation']
        
        # Asegurarse de que la máscara sea booleana para usarla como índice
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        original_idx = mask_indices_to_show[idx]
        color = all_colors[original_idx]
        
        # Crear máscara con transparencia
        # Mezclar color con imagen original usando alpha blending
        alpha = 0.5
        
        # Aplicar color solo donde está la máscara
        result[mask] = result[mask] * (1 - alpha) + color * alpha
    
    # Convertir de vuelta a uint8
    result = (result * 255).astype(np.uint8)
    
    # Dibujar contornos sobre la imagen resultante
    result_with_contours = result.copy()
    for idx, mask_data in enumerate(masks_to_show):
        mask = mask_data['segmentation']
        original_idx = mask_indices_to_show[idx]
        color = all_colors[original_idx]
        
        # Encontrar contornos
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Dibujar contornos directamente en la imagen
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))  # RGB a BGR
        cv2.drawContours(result_with_contours, contours, -1, color_bgr, 2)
    
    # Dibujar puntos si están disponibles
    if point_coords and point_labels and len(point_coords) > 0:
        # Convertir a BGR para OpenCV
        result_bgr = cv2.cvtColor(result_with_contours, cv2.COLOR_RGB2BGR)
        
        for coord, label in zip(point_coords, point_labels):
            x, y = int(coord[0]), int(coord[1])
            # Foreground: verde, Background: rojo
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # BGR: verde o rojo
            # Dibujar círculo relleno
            cv2.circle(result_bgr, (x, y), 8, color, -1)
            # Dibujar borde blanco
            cv2.circle(result_bgr, (x, y), 8, (255, 255, 255), 2)
        
        # Convertir de vuelta a RGB
        result_with_contours = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_with_contours)


def get_mask_info_list(masks):
    """
    Genera una lista de información de máscaras para mostrar en la UI.
    
    Args:
        masks: Lista de diccionarios con información de máscaras
        
    Returns:
        list: Lista de strings con información de cada máscara
    """
    if masks is None or len(masks) == 0:
        return []
    
    # Ordenar máscaras por área (de mayor a menor)
    sorted_indices = sorted(range(len(masks)), key=lambda i: masks[i]['area'], reverse=True)
    
    info_list = []
    for idx, original_idx in enumerate(sorted_indices):
        mask = masks[original_idx]
        area = mask.get('area', 0)
        bbox = mask.get('bbox', [0, 0, 0, 0])
        score = mask.get('stability_score', 0)
        
        info = f"Máscara {idx + 1} (área: {area:,} px², score: {score:.3f})"
        info_list.append(info)
    
    return info_list

