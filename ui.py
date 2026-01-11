"""
Interfaz de usuario con Gradio
Define y configura la interfaz gráfica de la aplicación.
"""
import gradio as gr
import numpy as np
import json
import tempfile
import os
from PIL import Image
from config import MODELS_INFO, DEFAULT_SEGMENTATION_PARAMS
from models import load_model
from segmentation import segment_image, preprocess_image
from visualization import visualize_masks, get_mask_info_list


def segment_image_wrapper(image, model_name, points_per_side, pred_iou_thresh, stability_score_thresh, point_coords=None, point_labels=None):
    """
    Wrapper para segment_image que maneja el retorno de máscaras.
    
    Returns:
        tuple: (imagen_resultado, mensaje_estado, máscaras, imagen_original, lista_info_máscaras)
    """
    if image is None:
        return None, "⚠️ Por favor, sube una imagen primero", None, None, []
    
    result = segment_image(image, model_name, points_per_side, pred_iou_thresh, stability_score_thresh, point_coords, point_labels)
    
    if result[0] is None:
        # Error en la segmentación
        return result[0], result[1], None, None, []
    
    # result contiene: (imagen, status, masks, image_np)
    result_image, status, masks, image_np = result
    
    # Generar lista de información de máscaras
    mask_info_list = get_mask_info_list(masks) if masks else []
    
    return result_image, status, masks, image_np, mask_info_list


def update_visualization(masks, image_np, selected_masks, point_coords=None, point_labels=None):
    """
    Actualiza la visualización basada en las máscaras seleccionadas.
    
    Args:
        masks: Lista de máscaras
        image_np: Imagen original en formato numpy
        selected_masks: Lista de strings de máscaras seleccionadas del CheckboxGroup
        point_coords: Lista opcional de coordenadas de puntos
        point_labels: Lista opcional de etiquetas de puntos
        
    Returns:
        PIL.Image: Imagen actualizada
    """
    if masks is None or image_np is None:
        return None
    
    if not selected_masks or len(selected_masks) == 0:
        # Si no hay selección, mostrar solo la imagen original (sin máscaras)
        # Pero mostrar puntos si hay
        if point_coords and len(point_coords) > 0:
            return visualize_masks(image_np, [], point_coords=point_coords, point_labels=point_labels)
        return Image.fromarray(image_np)
    
    # Convertir strings a índices enteros
    # selected_masks viene como lista de strings como "Máscara 1 (área: ...)"
    # Necesitamos extraer el número de máscara
    try:
        selected_indices = []
        for mask_str in selected_masks:
            # Extraer el número de la máscara del string
            # Formato: "Máscara 1 (área: ...)"
            parts = mask_str.split()
            if len(parts) >= 2:
                mask_num = int(parts[1]) - 1  # Convertir a índice 0-based
                if 0 <= mask_num < len(masks):
                    selected_indices.append(mask_num)
    except Exception as e:
        print(f"[update_visualization] Error procesando selección: {e}")
        print(f"[update_visualization] selected_masks: {selected_masks}")
        # Si hay error, mostrar todas
        return visualize_masks(image_np, masks, point_coords=point_coords, point_labels=point_labels)
    
    if not selected_indices:
        # Si no se pudieron extraer índices válidos, mostrar todas
        return visualize_masks(image_np, masks, point_coords=point_coords, point_labels=point_labels)
    
    return visualize_masks(image_np, masks, selected_indices, point_coords=point_coords, point_labels=point_labels)


def mask_to_rle(mask):
    """
    Convierte una máscara binaria a formato RLE (Run Length Encoding).
    
    Args:
        mask: Array numpy binario (2D)
        
    Returns:
        dict: Diccionario con 'size' y 'counts' en formato RLE
    """
    # Aplanar la máscara en orden de fila
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return {
        'size': list(mask.shape),
        'counts': runs.tolist()
    }


def export_masks_to_json(masks, selected_masks, image_shape=None, use_rle=True):
    """
    Exporta las máscaras seleccionadas a formato JSON.
    
    Args:
        masks: Lista completa de máscaras
        selected_masks: Lista de strings de máscaras seleccionadas
        image_shape: Tupla con las dimensiones de la imagen (height, width)
        use_rle: Si True, usa RLE (compacto). Si False, exporta array completo (más legible pero más grande)
        
    Returns:
        str: Ruta al archivo JSON temporal o None si hay error
    """
    if masks is None or len(masks) == 0:
        return None
    
    if not selected_masks or len(selected_masks) == 0:
        return None
    
    # Ordenar máscaras por área (igual que en visualization)
    sorted_indices = sorted(range(len(masks)), key=lambda i: masks[i]['area'], reverse=True)
    
    # Extraer índices de máscaras seleccionadas
    selected_indices = []
    for mask_str in selected_masks:
        parts = mask_str.split()
        if len(parts) >= 2:
            try:
                mask_num = int(parts[1]) - 1  # Convertir a índice 0-based
                if 0 <= mask_num < len(sorted_indices):
                    selected_indices.append(mask_num)
            except ValueError:
                continue
    
    if not selected_indices:
        return None
    
    # Preparar datos para exportación
    export_data = {
        'metadata': {
            'total_masks': len(masks),
            'exported_masks': len(selected_indices),
            'image_shape': image_shape if image_shape else None
        },
        'masks': []
    }
    
    # Exportar cada máscara seleccionada
    for idx in selected_indices:
        original_idx = sorted_indices[idx]
        mask_data = masks[original_idx]
        
        # Obtener máscara
        mask_array = mask_data['segmentation']
        
        # Preparar datos de la máscara
        mask_export = {
            'id': idx + 1,  # ID basado en orden visual
            'original_index': int(original_idx),
            'area': int(mask_data.get('area', 0)),
            'bbox': [float(x) for x in mask_data.get('bbox', [0, 0, 0, 0])],
            'stability_score': float(mask_data.get('stability_score', 0.0)),
            'predicted_iou': float(mask_data.get('predicted_iou', 0.0)) if 'predicted_iou' in mask_data else None,
        }
        
        # Exportar en formato RLE o array completo según preferencia
        if use_rle:
            rle = mask_to_rle(mask_array)
            mask_export['segmentation'] = {
                'format': 'rle',
                'rle': rle,
                'size': list(mask_array.shape)
            }
        else:
            # Exportar array completo (más legible pero más grande)
            mask_export['segmentation'] = {
                'format': 'array',
                'size': list(mask_array.shape),
                'data': mask_array.tolist()  # Convertir numpy array a lista de Python
            }
        
        export_data['masks'].append(mask_export)
    
    # Crear archivo temporal
    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(export_data, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"[export_masks_to_json] Error: {e}")
        return None


def create_interface():
    """
    Crea y configura la interfaz de Gradio.
    
    Returns:
        gr.Blocks: Interfaz de Gradio configurada
    """
    with gr.Blocks(title="SAM 2.1 - Segmentación de Imágenes", theme=gr.themes.Soft()) as demo:
        
        # Estados para almacenar datos entre llamadas
        masks_state = gr.State(value=None)
        image_np_state = gr.State(value=None)
        mask_choices_state = gr.State(value=[])  # Guardar las opciones disponibles
        point_coords_state = gr.State(value=[])  # Guardar coordenadas de puntos
        point_labels_state = gr.State(value=[])  # Guardar etiquetas de puntos
        original_image_state = gr.State(value=None)  # Guardar imagen original sin puntos
        
        # Título
        gr.Markdown(
            """
            # 🔬 GIA - Segmentación Automática de Imágenes
            ### Segment Anything Model 2.1 - Meta AI
            """
        )
        
        # Selector de modelo
        with gr.Row():
            with gr.Column(scale=4):
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS_INFO.keys()),
                    value="Base+ (Recomendado)",
                    label="🧠 Seleccionar Modelo",
                    info="Elige el modelo según tu hardware y necesidades"
                )
            
            with gr.Column(scale=1, min_width=150):
                load_btn = gr.Button("🔄 Cargar Modelo", variant="secondary", size="lg")
        
        # Información del modelo
        model_info = gr.Textbox(
            label="ℹ️ Estado del Modelo",
            value="Selecciona un modelo y haz clic en 'Cargar Modelo'",
            interactive=False
        )
        
        # Parámetros avanzados (colapsables)
        with gr.Accordion("⚙️ Parámetros Avanzados", open=False):
            with gr.Row():
                points_per_side = gr.Slider(
                    minimum=8,
                    maximum=64,
                    value=DEFAULT_SEGMENTATION_PARAMS["points_per_side"],
                    step=8,
                    label="Puntos por lado",
                    info="Más puntos = más máscaras detectadas"
                )
                
                pred_iou_thresh = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=DEFAULT_SEGMENTATION_PARAMS["pred_iou_thresh"],
                    step=0.01,
                    label="Umbral IoU",
                    info="Filtro de calidad de predicción"
                )
                
                stability_score_thresh = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=DEFAULT_SEGMENTATION_PARAMS["stability_score_thresh"],
                    step=0.01,
                    label="Umbral de estabilidad",
                    info="Filtro de estabilidad de máscaras"
                )
        
        gr.Markdown("---")
        
        # Panel principal: Imagen de entrada y resultado
        with gr.Row(equal_height=True):
            # Panel izquierdo: Imagen de entrada
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Imagen de Entrada")
                
                # Radio buttons para seleccionar tipo de punto
                point_type = gr.Radio(
                    choices=["Foreground", "Background"],
                    value="Foreground",
                    label="🎯 Tipo de Punto",
                )
                
                input_image = gr.Image(
                    label="Sube tu imagen (haz click para agregar puntos)",
                    type="pil",
                    height=500,
                    interactive=True
                )
                
                # Botón para limpiar puntos
                clear_points_btn = gr.Button(
                    "🗑️ Limpiar Puntos",
                    variant="secondary",
                    size="sm"
                )
                
                segment_btn = gr.Button(
                    "🎯 SEGMENTAR",
                    variant="primary",
                    size="lg"
                )
            
            # Panel derecho: Resultado
            with gr.Column(scale=1):
                gr.Markdown("### 🎨 Resultado de Segmentación")
                
                # Espaciador para alinear con el radio button de la izquierda
                gr.HTML("<div style='height: 80px;'></div>")
                
                output_image = gr.Image(
                    label="Máscaras detectadas",
                    type="pil",
                    height=500
                )
                
                # Espaciadores para alinear los botones
                gr.HTML("<div style='height: 20px;'></div>")
                gr.HTML("<div style='height: 50px;'></div>")
        
        # Panel de selección de máscaras (inicialmente oculto)
        with gr.Accordion("🎛️ Seleccionar Máscaras", open=False) as mask_selection_panel:
            mask_checkboxes = gr.CheckboxGroup(
                label="Máscaras detectadas",
                choices=[],
                value=[],
                info="Selecciona las máscaras que deseas visualizar. Deja vacío para ver todas."
            )
            
            with gr.Row():
                select_all_btn = gr.Button("✅ Seleccionar Todas", variant="secondary", size="sm")
                deselect_all_btn = gr.Button("❌ Deseleccionar Todas", variant="secondary", size="sm")
            
            with gr.Row():
                export_format = gr.Radio(
                    choices=["RLE", "Original"],
                    value="RLE",
                    label="📋 Formato de exportación",
                )
                export_btn = gr.Button("💾 Exportar JSON", variant="primary", size="sm")
            
            export_file = gr.File(
                label="📥 Archivo JSON exportado",
                visible=False,
                interactive=False
            )
        
        # Estado de la segmentación
        status_text = gr.Textbox(
            label="📊 Estado",
            value="Esperando imagen...",
            interactive=False
        )
        
        # ====================================================================
        # EVENTOS
        # ====================================================================
        
        # Cargar modelo
        load_btn.click(
            fn=load_model,
            inputs=[model_dropdown],
            outputs=[model_info]
        )
        
        # Función para guardar imagen original cuando se sube
        def save_original_image(image):
            """Guarda la imagen original cuando se sube"""
            # Solo guardar en el estado, no actualizar el input_image para evitar loops
            if image is not None:
                return image  # Solo devolver para original_image_state
            return None
        
        # Función para manejar clicks en la imagen
        def handle_image_click(evt: gr.SelectData, original_image, point_type_value, current_coords, current_labels):
            """Maneja los clicks en la imagen para agregar puntos"""
            if original_image is None:
                return original_image, current_coords, current_labels, original_image  # Mantener original sin puntos
            
            # Obtener coordenadas del click
            x, y = evt.index[0], evt.index[1]
            
            # Determinar label según el tipo de punto seleccionado
            label = 1 if point_type_value == "Foreground" else 0
            
            # Agregar punto a las listas
            # Asegurarse de que current_coords y current_labels sean listas
            if current_coords is None:
                current_coords = []
            if current_labels is None:
                current_labels = []
            
            new_coords = current_coords + [[x, y]]
            new_labels = current_labels + [label]
            
            # Dibujar puntos sobre una COPIA de la imagen original (NO modificar la original)
            from PIL import ImageDraw
            img_with_points = original_image.copy()  # Copia para mostrar, NO modifica original_image
            draw = ImageDraw.Draw(img_with_points)
            
            # Dibujar todos los puntos
            for coord, lbl in zip(new_coords, new_labels):
                color = "green" if lbl == 1 else "red"
                # Dibujar círculo
                draw.ellipse(
                    [coord[0] - 8, coord[1] - 8, coord[0] + 8, coord[1] + 8],
                    fill=color,
                    outline="white",
                    width=2
                )
            
            # Devolver: imagen con puntos para mostrar, nuevas coordenadas, nuevas labels, y la imagen ORIGINAL sin modificar
            return img_with_points, new_coords, new_labels, original_image
        
        # Función para limpiar puntos
        def clear_points(original_image, current_coords, current_labels):
            """Limpia todos los puntos seleccionados"""
            if original_image is None:
                return original_image, [], [], original_image
            # Devolver imagen original sin puntos (mantener original_image_state intacto)
            return original_image, [], [], original_image
        
        # Función para mostrar imagen con puntos (sin segmentar)
        def show_image_with_points(original_image, coords, labels):
            """Muestra la imagen con los puntos dibujados"""
            if original_image is None:
                return original_image
            
            if not coords or len(coords) == 0:
                return original_image
            
            from PIL import ImageDraw
            img_with_points = original_image.copy()
            draw = ImageDraw.Draw(img_with_points)
            
            # Dibujar todos los puntos
            for coord, lbl in zip(coords, labels):
                color = "green" if lbl == 1 else "red"
                # Dibujar círculo
                draw.ellipse(
                    [coord[0] - 8, coord[1] - 8, coord[0] + 8, coord[1] + 8],
                    fill=color,
                    outline="white",
                    width=2
                )
            
            return img_with_points
        
        # Segmentar imagen
        def segment_and_store(original_image, model_name, points_per_side, pred_iou_thresh, stability_score_thresh, point_coords, point_labels):
            """Segmenta y almacena los resultados en el estado"""
            # Si hay puntos, se usarán automáticamente. Si no hay puntos, será segmentación automática.
            # La función segment_image ya maneja esto internamente
            
            # Usar la imagen original sin puntos para la segmentación
            result_image, status, masks, image_np, mask_info_list = segment_image_wrapper(
                original_image, model_name, points_per_side, pred_iou_thresh, stability_score_thresh, point_coords, point_labels
            )
            
            # Si hay máscaras, seleccionar la primera (mejor) por defecto
            # Si es segmentación con puntos, habrá 3 máscaras, seleccionar la mejor (primera)
            if mask_info_list and len(mask_info_list) > 0:
                # Seleccionar la primera máscara (la mejor) por defecto
                selected_value = [mask_info_list[0]]
            else:
                selected_value = []
            
            # Actualizar el CheckboxGroup con nuevas opciones y valores
            # En Gradio, para actualizar choices y value, usamos gr.update()
            checkbox_update = gr.update(choices=mask_info_list, value=selected_value)
            
            return (
                result_image,  # output_image (ya tiene todas las máscaras visualizadas)
                status,  # status_text
                masks,  # masks_state
                image_np,  # image_np_state
                checkbox_update,  # mask_checkboxes (vacío por defecto)
                mask_info_list  # mask_choices_state
            )
        
        # Guardar imagen original cuando se sube (solo cuando cambia, no en cada actualización)
        # Usar upload en lugar de change para evitar que se ejecute cuando se actualiza la imagen con puntos
        def on_image_upload(image):
            """Se ejecuta cuando se sube una nueva imagen"""
            if image is not None:
                # Limpiar puntos cuando se sube una nueva imagen
                return image, [], []  # original_image_state, point_coords_state, point_labels_state
            return None, [], []
        
        # Usar upload en lugar de change para que solo se ejecute cuando se sube una imagen nueva
        input_image.upload(
            fn=on_image_upload,
            inputs=[input_image],
            outputs=[original_image_state, point_coords_state, point_labels_state]
        )
        
        # Evento para capturar clicks en la imagen
        input_image.select(
            fn=handle_image_click,
            inputs=[original_image_state, point_type, point_coords_state, point_labels_state],
            outputs=[input_image, point_coords_state, point_labels_state, original_image_state]
        )
        
        # Evento para limpiar puntos
        clear_points_btn.click(
            fn=clear_points,
            inputs=[original_image_state, point_coords_state, point_labels_state],
            outputs=[input_image, point_coords_state, point_labels_state, original_image_state]
        )
        
        # Actualizar imagen cuando cambian los puntos (para mostrar puntos sin segmentar)
        point_coords_state.change(
            fn=show_image_with_points,
            inputs=[original_image_state, point_coords_state, point_labels_state],
            outputs=[input_image]
        )
        
        segment_btn.click(
            fn=segment_and_store,
            inputs=[
                original_image_state,
                model_dropdown,
                points_per_side,
                pred_iou_thresh,
                stability_score_thresh,
                point_coords_state,
                point_labels_state
            ],
            outputs=[output_image, status_text, masks_state, image_np_state, mask_checkboxes, mask_choices_state]
        )
        
        # Actualizar visualización cuando cambian las selecciones
        def on_mask_selection_change(masks, image_np, selected_masks, point_coords, point_labels):
            """Actualiza la visualización cuando cambia la selección de máscaras"""
            if masks is None or image_np is None:
                return None
            # Mostrar puntos solo si fueron usados en la segmentación (si existen)
            return update_visualization(masks, image_np, selected_masks, point_coords, point_labels)
        
        mask_checkboxes.change(
            fn=on_mask_selection_change,
            inputs=[masks_state, image_np_state, mask_checkboxes, point_coords_state, point_labels_state],
            outputs=[output_image]
        )
        
        # Botones de selección rápida
        def select_all_masks(mask_choices):
            """Selecciona todas las máscaras"""
            if mask_choices:
                return mask_choices  # Devolver lista de strings
            return []
        
        def deselect_all_masks(mask_choices):
            """Deselecciona todas las máscaras"""
            return []  # Devolver lista vacía
        
        select_all_btn.click(
            fn=select_all_masks,
            inputs=[mask_choices_state],
            outputs=[mask_checkboxes]
        )
        
        deselect_all_btn.click(
            fn=deselect_all_masks,
            inputs=[mask_choices_state],
            outputs=[mask_checkboxes]
        )
        
        # Exportar máscaras a JSON
        def export_selected_masks(masks, selected_masks, image_np, format_choice):
            """Exporta las máscaras seleccionadas a JSON"""
            if masks is None or image_np is None:
                return gr.update(visible=False, value=None), "⚠️ No hay máscaras para exportar"
            
            if not selected_masks or len(selected_masks) == 0:
                return gr.update(visible=False, value=None), "⚠️ Por favor, selecciona al menos una máscara para exportar"
            
            # Determinar si usar RLE o array completo
            use_rle = format_choice == "RLE"
            
            # Obtener dimensiones de la imagen
            image_shape = image_np.shape[:2] if len(image_np.shape) >= 2 else None
            
            # Exportar
            json_file = export_masks_to_json(masks, selected_masks, image_shape, use_rle=use_rle)
            
            if json_file:
                num_exported = len(selected_masks)
                format_name = "RLE" if use_rle else "Array completo"
                return gr.update(visible=True, value=json_file), f"✅ Exportadas {num_exported} máscara(s) a JSON ({format_name})"
            else:
                return gr.update(visible=False, value=None), "❌ Error al exportar las máscaras"
        
        export_btn.click(
            fn=export_selected_masks,
            inputs=[masks_state, mask_checkboxes, image_np_state, export_format],
            outputs=[export_file, status_text]
        )
    
    return demo
