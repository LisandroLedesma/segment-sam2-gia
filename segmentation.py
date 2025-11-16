"""
Lógica de segmentación de imágenes
Procesa imágenes y genera máscaras usando SAM2.
"""
import numpy as np
import cv2
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from models import get_current_model, ensure_model_loaded


def preprocess_image(image):
    """
    Preprocesa la imagen para la segmentación.
    
    Args:
        image: Imagen en formato PIL o numpy array
        
    Returns:
        numpy.ndarray: Imagen en formato BGR para OpenCV
    """
    # Convertir imagen a numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Asegurar formato BGR para OpenCV
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np

    return image_np, image_bgr


def segment_image_with_points(image, model_name, point_coords, point_labels):
    """
    Segmenta la imagen usando SAM2 con puntos de prompt.
    
    Args:
        image: Imagen a segmentar (PIL Image o numpy array)
        model_name: Nombre del modelo a usar
        point_coords: Lista de coordenadas de puntos [(x, y), ...]
        point_labels: Lista de etiquetas (1 para foreground, 0 para background)
        
    Returns:
        tuple: (imagen_resultado, mensaje_estado, máscaras, imagen_original) o (None, mensaje_error, None, None)
    """
    print("[segment_image_with_points] Inicio")
    print(f"[segment_image_with_points] Puntos: {len(point_coords)}")
    
    if image is None:
        print("[segment_image_with_points] image is None")
        return None, "⚠️ Por favor, sube una imagen primero", None, None
    
    if not point_coords or len(point_coords) == 0:
        return None, "⚠️ Por favor, selecciona al menos un punto", None, None
    
    # Asegurar que el modelo esté cargado
    success, status = ensure_model_loaded(model_name)
    if not success:
        print("[segment_image_with_points] error al cargar modelo")
        return None, status, None, None
    
    try:
        # Preprocesar imagen
        print("[segment_image_with_points] preprocesando imagen")
        image_np, image_bgr = preprocess_image(image)
        
        # Obtener modelo actual
        current_model = get_current_model()
        if current_model is None:
            return None, "❌ Error: Modelo no disponible", None, None
        
        # Crear predictor
        print("[segment_image_with_points] creando SAM2ImagePredictor")
        predictor = SAM2ImagePredictor(current_model)
        
        # Configurar la imagen
        predictor.set_image(image_bgr)
        
        # Convertir puntos a formato numpy
        # point_coords debe ser array de shape (N, 2) con coordenadas (x, y)
        # point_labels debe ser array de shape (N,) con valores 1 (foreground) o 0 (background)
        point_coords_np = np.array(point_coords, dtype=np.float32)
        point_labels_np = np.array(point_labels, dtype=np.int32)
        
        print(f"[segment_image_with_points] Coordenadas: {point_coords_np.shape}")
        print(f"[segment_image_with_points] Labels: {point_labels_np.shape}")
        
        # Predecir máscaras
        print("[segment_image_with_points] prediciendo máscaras...")
        masks, scores, logits = predictor.predict(
            point_coords=point_coords_np,
            point_labels=point_labels_np,
            multimask_output=True
        )
        
        print(f"[segment_image_with_points] máscaras generadas: {len(masks)}")
        
        # Convertir máscaras al formato esperado
        # Seleccionar la máscara con mejor score
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        # Crear estructura de máscara compatible con el resto del código
        mask_data = {
            'segmentation': best_mask,
            'area': int(np.sum(best_mask)),
            'bbox': [0, 0, image_np.shape[1], image_np.shape[0]],  # Bounding box aproximado
            'stability_score': float(scores[best_mask_idx]),
            'predicted_iou': float(scores[best_mask_idx])
        }
        masks_list = [mask_data]
        
        # Visualizar máscaras con puntos
        from visualization import visualize_masks
        result_image = visualize_masks(image_np, masks_list, point_coords=point_coords, point_labels=point_labels)
        status = f"✅ Segmentación completada: 1 máscara generada (score: {scores[best_mask_idx]:.3f})"
        
        print("[segment_image_with_points] OK, devolviendo resultado")
        return result_image, status, masks_list, image_np
        
    except Exception as e:
        msg = f"❌ Error durante la segmentación: {repr(e)}"
        print("[segment_image_with_points]", msg)
        import traceback
        traceback.print_exc()
        return None, msg, None, None


def segment_image(image, model_name, points_per_side, pred_iou_thresh, stability_score_thresh, point_coords=None, point_labels=None):
    """
    Segmenta la imagen usando SAM2.
    
    Args:
        image: Imagen a segmentar (PIL Image o numpy array)
        model_name: Nombre del modelo a usar
        points_per_side: Número de puntos por lado para la generación de máscaras
        pred_iou_thresh: Umbral IoU para filtrar predicciones
        stability_score_thresh: Umbral de estabilidad para filtrar máscaras
        point_coords: Lista opcional de coordenadas de puntos [(x, y), ...]
        point_labels: Lista opcional de etiquetas (1 para foreground, 0 para background)
        
    Returns:
        tuple: (imagen_resultado, mensaje_estado, máscaras, imagen_original) o (None, mensaje_error, None, None)
    """
    print("[segment_image] Inicio")
    print(f"[segment_image] model_name={model_name}")
    
    # Si hay puntos seleccionados, usar segmentación con puntos
    if point_coords and len(point_coords) > 0 and point_labels and len(point_labels) > 0:
        print("[segment_image] Usando segmentación con puntos")
        return segment_image_with_points(image, model_name, point_coords, point_labels)
    
    # Si no hay puntos, usar segmentación automática
    print("[segment_image] Usando segmentación automática")
    
    if image is None:
        print("[segment_image] image is None")
        return None, "⚠️ Por favor, sube una imagen primero", None, None

    # Asegurar que el modelo esté cargado
    success, status = ensure_model_loaded(model_name)
    if not success:
        print("[segment_image] error al cargar modelo")
        return None, status, None, None

    try:
        # Preprocesar imagen
        print("[segment_image] preprocesando imagen")
        image_np, image_bgr = preprocess_image(image)

        # Obtener modelo actual
        current_model = get_current_model()
        if current_model is None:
            return None, "❌ Error: Modelo no disponible", None, None

        # Crear generador de máscaras
        print("[segment_image] creando SAM2AutomaticMaskGenerator")
        mask_generator = SAM2AutomaticMaskGenerator(
            model=current_model,
            points_per_side=int(points_per_side),
            pred_iou_thresh=float(pred_iou_thresh),
            stability_score_thresh=float(stability_score_thresh),
        )

        # Generar máscaras
        print("[segment_image] generando máscaras...")
        masks = mask_generator.generate(image_bgr)
        print(f"[segment_image] máscaras generadas: {len(masks)}")

        # La visualización se hace en el módulo de visualización
        from visualization import visualize_masks
        
        # Visualizar todas las máscaras por defecto (mostrar puntos si existen aunque no se usen)
        result_image = visualize_masks(image_np, masks, point_coords=point_coords, point_labels=point_labels)
        status = f"✅ Segmentación completada: {len(masks)} máscaras detectadas"

        print("[segment_image] OK, devolviendo resultado")
        # Devolver también las máscaras y la imagen original para poder re-visualizarlas después
        return result_image, status, masks, image_np

    except Exception as e:
        msg = f"❌ Error durante la segmentación: {repr(e)}"
        print("[segment_image]", msg)
        import traceback
        traceback.print_exc()
        return None, msg, None, None

