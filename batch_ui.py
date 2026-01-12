"""
Interfaz de usuario para procesamiento por lotes (batch)
Permite procesar múltiples imágenes y mostrar el paso a paso de cada una.
"""
import os
import gradio as gr
import numpy as np
from PIL import Image
from config import MODELS_INFO, DEFAULT_SEGMENTATION_PARAMS
from models import load_model, ensure_model_loaded, get_current_model
from segmentation import segment_image, preprocess_image, segment_single_point
from visualization import visualize_masks, get_mask_info_list


def create_batch_interface():
    """
    Crea y configura la interfaz de procesamiento por lotes.
    Devuelve el contenido para ser usado dentro de un Tab.
    
    Returns:
        tuple: (contenido_column, eventos_dict) donde eventos_dict contiene las funciones de eventos
    """
    # Estados para almacenar datos
    selected_model_state = gr.State(value=None)
    images_state = gr.State(value=[])  # Lista de imágenes cargadas
    image_points_state = gr.State(value={})  # Diccionario: {nombre_imagen: [(x, y), ...]}
    processing_results_state = gr.State(value={})  # Diccionario con resultados por imagen
    # Componentes Image dinámicos (máximo 10 imágenes)
    batch_image_components = []  # Lista para almacenar referencias a componentes Image
    batch_image_original_state = []  # Lista para almacenar imágenes originales sin puntos
    
    with gr.Column() as batch_content:
        # Sección 1: Selección de modelo
        gr.Markdown("### 📋 Paso 1: Seleccionar Modelo")
        with gr.Row():
            with gr.Column(scale=4):
                batch_model_dropdown = gr.Dropdown(
                    choices=list(MODELS_INFO.keys()),
                    value="Base+ (Recomendado)",
                    label="🧠 Seleccionar Modelo",
                    info="Elige el modelo según tu hardware y necesidades"
                )
            
            with gr.Column(scale=1, min_width=150):
                batch_load_btn = gr.Button("🔄 Cargar Modelo", variant="secondary", size="lg")
        
        batch_model_info = gr.Textbox(
            label="ℹ️ Estado del Modelo",
            value="Selecciona un modelo y haz clic en 'Cargar Modelo'",
            interactive=False
        )
        
        gr.Markdown("---")
        
        # Sección 2: Carga de imágenes
        gr.Markdown("### 📤 Paso 2: Cargar Imágenes")
        batch_image_upload = gr.File(
            label="Sube múltiples imágenes",
            file_count="multiple",
            file_types=["image"],
            type="filepath"
        )
        
        batch_images_info = gr.Textbox(
            label="📊 Estado de Imágenes",
            value="Esperando imágenes...",
            interactive=False
        )
        
        gr.Markdown("---")
        
        # Sección 2.5: Selección de puntos por imagen
        gr.Markdown("### 🎯 Paso 2.5: Seleccionar Puntos en las Imágenes")
        gr.Markdown("**Instrucciones:** Haz clic en cada imagen para agregar puntos. Cada punto generará una segmentación independiente (se guardará la tercera máscara de cada una).")
        
        # Crear componentes Image para hasta 10 imágenes (se pueden ocultar si no se usan)
        batch_image_components_list = []
        batch_original_images_state = gr.State(value=[])  # Guardar imágenes originales sin puntos
        
        with gr.Column(visible=False) as batch_images_container:
            for i in range(10):  # Máximo 10 imágenes
                img_component = gr.Image(
                    label=f"Imagen {i+1}",
                    type="pil",
                    height=300,
                    interactive=True,
                    visible=False
                )
                batch_image_components_list.append(img_component)
        
        batch_points_info = gr.Textbox(
            label="📊 Puntos Seleccionados",
            value="Carga imágenes primero",
            interactive=False
        )
        
        with gr.Row():
            batch_undo_point_btn = gr.Button("↩️ Deshacer Último Punto", variant="secondary")
            batch_clear_points_btn = gr.Button("🗑️ Limpiar Todos los Puntos", variant="secondary")
        
        gr.Markdown("---")
        
        # Sección 3: Parámetros de segmentación
        gr.Markdown("### ⚙️ Parámetros de Segmentación")
        with gr.Row():
            batch_points_per_side = gr.Slider(
                minimum=8,
                maximum=64,
                value=DEFAULT_SEGMENTATION_PARAMS["points_per_side"],
                step=8,
                label="Puntos por lado",
                info="Más puntos = más máscaras detectadas"
            )
            
            batch_pred_iou_thresh = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=DEFAULT_SEGMENTATION_PARAMS["pred_iou_thresh"],
                step=0.01,
                label="Umbral IoU",
                info="Filtro de calidad de predicción"
            )
            
            batch_stability_score_thresh = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=DEFAULT_SEGMENTATION_PARAMS["stability_score_thresh"],
                step=0.01,
                label="Umbral de estabilidad",
                info="Filtro de estabilidad de máscaras"
            )
        
        with gr.Row():
            batch_mask_index = gr.Slider(
                minimum=1,
                maximum=3,
                value=3,
                step=1,
                label="Máscara a seleccionar",
                info="1=Primera (mejor score), 2=Segunda, 3=Tercera"
            )
        
        gr.Markdown("---")
        
        # Sección 4: Procesamiento y resultados
        gr.Markdown("### 🚀 Paso 3: Procesar Imágenes")
        batch_process_btn = gr.Button(
            "▶️ PROCESAR TODAS LAS IMÁGENES",
            variant="primary",
            size="lg",
            visible=False
        )
        
        batch_status = gr.Textbox(
            label="📊 Estado del Procesamiento",
            value="Carga imágenes para comenzar",
            interactive=False
        )
        
        # Contenedor para resultados
        gr.Markdown("### 📁 Resultados por Imagen")
        batch_results_html = gr.HTML(
            visible=False
        )
        
        gr.Markdown("---")
        
        # Sección de exportación
        gr.Markdown("### 💾 Exportar Imágenes Finales")
        gr.Markdown("**Descarga las imágenes finales con todas las máscaras superpuestas en formato PNG de alta calidad.**")
        
        batch_export_files = gr.File(
            label="📥 Imágenes Finales Exportadas (PNG - Alta Calidad)",
            file_count="multiple",
            visible=False,
            interactive=False
        )
        
        batch_export_info = gr.Textbox(
            label="ℹ️ Información de Exportación",
            value="Las imágenes se generarán después del procesamiento",
            interactive=False,
            lines=8,
            max_lines=15
        )
    
    # Funciones de eventos
    def on_batch_model_load(model_name):
        """Carga el modelo seleccionado"""
        status = load_model(model_name)
        if "✅" in status:
            return status, model_name, gr.update(visible=True)
        return status, None, gr.update(visible=False)
    
    def on_images_upload(files):
        """Maneja la carga de múltiples imágenes y actualiza los componentes Image"""
        if files is None or len(files) == 0:
            updates = [gr.update(visible=False) for _ in range(10)]
            return "Esperando imágenes...", [], {}, [], *updates, gr.update(visible=False), "Carga imágenes primero"
        
        # Cargar imágenes
        images = []
        image_points = {}
        original_images = []
        
        for file_path in files:
            try:
                img = Image.open(file_path)
                img_name = os.path.basename(file_path)
                images.append({
                    'image': img,
                    'path': file_path,
                    'name': img_name
                })
                original_images.append(img.copy())  # Guardar copia original
                image_points[img_name] = []  # Inicializar sin puntos
            except Exception as e:
                print(f"Error cargando imagen {file_path}: {e}")
        
        if len(images) == 0:
            updates = [gr.update(visible=False) for _ in range(10)]
            return "⚠️ No se pudieron cargar las imágenes", [], {}, [], *updates, gr.update(visible=False), "Carga imágenes primero"
        
        # Actualizar componentes Image (mostrar solo las imágenes cargadas)
        updates = []
        for i in range(10):
            if i < len(images):
                img_data = images[i]
                updates.append(gr.update(
                    value=img_data['image'],
                    label=f"{img_data['name']} - Puntos: 0",
                    visible=True
                ))
            else:
                updates.append(gr.update(visible=False))
        
        info_text = f"✅ {len(images)} imagen(es) cargada(s) correctamente. Haz clic en las imágenes abajo para agregar puntos."
        return info_text, images, image_points, original_images, *updates, gr.update(visible=True), f"0 puntos seleccionados en total"
    
    def update_image_with_points(img_name, image, current_points):
        """Actualiza una imagen dibujando los puntos seleccionados"""
        if image is None:
            return None
        
        points = current_points.get(img_name, []) if current_points else []
        
        # Dibujar puntos en la imagen
        from PIL import ImageDraw
        img_with_points = image.copy()
        draw = ImageDraw.Draw(img_with_points)
        
        for point in points:
            x, y = point[0], point[1]
            draw.ellipse(
                [x - 8, y - 8, x + 8, y + 8],
                fill="green",
                outline="white",
                width=2
            )
        
        return img_with_points
    
    def create_image_click_handler(img_idx):
        """Crea un handler para clicks en una imagen específica"""
        def handler(evt: gr.SelectData, images_list, current_points, original_images):
            if images_list is None or len(images_list) == 0 or img_idx >= len(images_list):
                # Devolver valores por defecto para todos los componentes (10 imágenes)
                default_updates = [gr.update() for _ in range(10)]
                return current_points or {}, *default_updates, "Error: índice de imagen inválido"
            
            img_data = images_list[img_idx]
            img_name = img_data['name']
            original_image = original_images[img_idx] if img_idx < len(original_images) else img_data['image']
            
            # Obtener coordenadas del click
            x, y = evt.index[0], evt.index[1]
            
            # Actualizar puntos
            if current_points is None:
                current_points = {}
            
            if img_name not in current_points:
                current_points[img_name] = []
            
            current_points[img_name].append([int(x), int(y)])
            
            # Actualizar todos los componentes Image (siempre 10, incluso si hay menos imágenes)
            updates = []
            num_images = len(images_list) if images_list else 0
            
            for i in range(10):  # Siempre 10 componentes
                if i < num_images:
                    img_data_i = images_list[i]
                    img_name_i = img_data_i['name']
                    original_img_i = original_images[i] if i < len(original_images) else img_data_i['image']
                    img_with_points_i = update_image_with_points(img_name_i, original_img_i, current_points)
                    points_count = len(current_points.get(img_name_i, []))
                    updates.append(gr.update(
                        value=img_with_points_i,
                        label=f"{img_name_i} - Puntos: {points_count}",
                        visible=True
                    ))
                else:
                    # Para componentes no usados, mantener ocultos
                    updates.append(gr.update(visible=False))
            
            total_points = sum(len(points) for points in current_points.values())
            points_in_image = len(current_points[img_name])
            info = f"✅ Punto agregado a {img_name} ({points_in_image} puntos). Total: {total_points} puntos"
            
            return current_points, *updates, info
        return handler
    
    def undo_last_point_batch(current_points, images_list, original_images):
        """Elimina el último punto agregado (de cualquier imagen)"""
        if current_points is None or not current_points:
            updates = [gr.update(visible=False) for _ in range(10)]
            return current_points or {}, *updates, "No hay puntos para deshacer"
        
        if images_list is None or len(images_list) == 0:
            updates = [gr.update(visible=False) for _ in range(10)]
            return current_points, *updates, "No hay imágenes cargadas"
        
        # Encontrar la última imagen que tiene puntos (orden inverso)
        last_image_with_points = None
        for img_data in reversed(images_list):
            img_name = img_data['name']
            if img_name in current_points and len(current_points[img_name]) > 0:
                last_image_with_points = img_name
                break
        
        if last_image_with_points is None:
            updates = [gr.update(visible=False) for _ in range(10)]
            return current_points, *updates, "No hay puntos para deshacer"
        
        # Eliminar el último punto de esa imagen
        current_points[last_image_with_points] = current_points[last_image_with_points][:-1]
        
        # Actualizar todas las imágenes (siempre 10 componentes)
        updates = []
        num_images = len(images_list) if images_list else 0
        
        for i in range(10):  # Siempre 10 componentes
            if i < num_images and i < len(original_images) if original_images else False:
                img_data = images_list[i]
                img_name_i = img_data['name']
                original_img_i = original_images[i] if i < len(original_images) else img_data['image']
                img_with_points_i = update_image_with_points(img_name_i, original_img_i, current_points)
                points_count = len(current_points.get(img_name_i, []))
                updates.append(gr.update(
                    value=img_with_points_i,
                    label=f"{img_name_i} - Puntos: {points_count}",
                    visible=True
                ))
            else:
                # Para componentes no usados, mantener ocultos
                updates.append(gr.update(visible=False))
        
        total_points = sum(len(points) for points in current_points.values())
        info = f"↩️ Último punto eliminado de {last_image_with_points}. Total: {total_points} puntos"
        
        return current_points, *updates, info
    
    def clear_all_points(current_points, images_list, original_images):
        """Limpia todos los puntos seleccionados y actualiza las imágenes"""
        if current_points is None:
            current_points = {}
        else:
            for img_name in current_points:
                current_points[img_name] = []
        
        # Actualizar todas las imágenes sin puntos (siempre 10 componentes)
        updates = []
        num_images = len(images_list) if images_list else 0
        
        for i in range(10):  # Siempre 10 componentes
            if i < num_images and i < len(original_images) if original_images else False:
                img_data = images_list[i]
                updates.append(gr.update(
                    value=original_images[i],
                    label=f"{img_data['name']} - Puntos: 0",
                    visible=True
                ))
            else:
                # Para componentes no usados, mantener ocultos
                updates.append(gr.update(visible=False))
        
        return current_points, *updates, "Todos los puntos han sido limpiados"
    
    def process_batch_images(model_name, images, image_points_dict, points_per_side, pred_iou_thresh, stability_score_thresh, mask_index):
        """Procesa todas las imágenes en batch, segmentando por cada punto"""
        if not images or len(images) == 0:
            return "⚠️ No hay imágenes para procesar", {}, gr.update(visible=False), gr.update(visible=False), "No hay imágenes para procesar"
        
        if model_name is None:
            return "⚠️ Por favor, carga un modelo primero", {}, gr.update(visible=False), gr.update(visible=False), "Modelo no cargado"
        
        # Verificar que haya puntos seleccionados
        total_points = sum(len(points) for points in (image_points_dict or {}).values())
        if total_points == 0:
            return "⚠️ Por favor, selecciona al menos un punto en las imágenes", {}, gr.update(visible=False), gr.update(visible=False), "No hay puntos seleccionados"
        
        # Asegurar que el modelo esté cargado
        success, status = ensure_model_loaded(model_name)
        if not success:
            return f"❌ {status}", {}, gr.update(visible=False), gr.update(visible=False), f"Error: {status}"
        
        results = {}
        total_images = len(images)
        total_segmentations = 0
        
        # Procesar cada imagen
        for idx, img_data in enumerate(images):
            image = img_data['image']
            img_name = img_data['name']
            
            # Obtener puntos para esta imagen
            points_for_image = image_points_dict.get(img_name, [])
            
            if len(points_for_image) == 0:
                # Si no hay puntos para esta imagen, saltarla
                continue
            
            # Actualizar estado
            status_msg = f"🔄 Procesando imagen {idx + 1}/{total_images}: {img_name} ({len(points_for_image)} puntos)"
            yield status_msg, results, gr.update(visible=False), gr.update(visible=False), "Procesando..."
            
            # Preprocesar imagen una vez
            image_np, image_bgr = preprocess_image(image)
            
            # Lista para almacenar todas las máscaras (tercera de cada segmentación)
            all_masks = []
            all_points = []
            individual_masks = []  # Lista de máscaras individuales por punto: [{point: [x,y], mask: mask_data, image: PIL.Image}, ...]
            
            # Procesar cada punto individualmente
            for point_idx, point_coord in enumerate(points_for_image):
                status_msg = f"🔄 Imagen {idx + 1}/{total_images}: Procesando punto {point_idx + 1}/{len(points_for_image)}"
                yield status_msg, results, gr.update(visible=False), gr.update(visible=False), "Procesando..."
                
                # Segmentar con un solo punto y obtener la máscara seleccionada
                # mask_index viene como 1, 2 o 3, pero la función espera 0, 1 o 2
                mask_idx = int(mask_index) - 1
                mask, score = segment_single_point(image, model_name, point_coord, point_label=1, mask_index=mask_idx)
                
                if mask is not None:
                    # Convertir máscara al formato esperado
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if np.any(rows) and np.any(cols):
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                    else:
                        bbox = [0, 0, 0, 0]
                    
                    mask_data = {
                        'segmentation': mask,
                        'area': int(np.sum(mask)),
                        'bbox': bbox,
                        'stability_score': score,
                        'predicted_iou': score
                    }
                    all_masks.append(mask_data)
                    all_points.append(point_coord)
                    
                    # Visualizar esta máscara individual sobre la imagen original
                    individual_mask_list = [mask_data]
                    individual_point = [point_coord]
                    individual_label = [1]
                    individual_image = visualize_masks(image_np, individual_mask_list, selected_indices=None,
                                                      point_coords=individual_point, point_labels=individual_label)
                    
                    # Convertir a base64 para guardar
                    import base64
                    from io import BytesIO
                    buffered = BytesIO()
                    individual_image.save(buffered, format="PNG")
                    individual_img_b64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    individual_masks.append({
                        'point': point_coord,
                        'point_index': point_idx + 1,
                        'mask': mask_data,
                        'image_b64': individual_img_b64
                    })
                    
                    total_segmentations += 1
            
            # Si hay máscaras, guardar toda la información
            if len(all_masks) > 0:
                # Crear lista de labels para los puntos (todos foreground)
                point_labels = [1] * len(all_points)
                
                # Visualizar todas las máscaras superpuestas sobre la imagen original (sin puntos)
                final_image_all_masks = visualize_masks(image_np, all_masks, selected_indices=None, 
                                                        point_coords=None, point_labels=None)
                
                # Imagen original con puntos (sin máscaras)
                original_with_points = visualize_masks(image_np, [], point_coords=all_points, point_labels=point_labels)
                
                # Crear máscara binaria combinada (OR lógico de todas las máscaras)
                combined_binary_mask = None
                if len(all_masks) > 0:
                    # Inicializar máscara combinada con la primera máscara
                    combined_binary_mask = all_masks[0]['segmentation'].copy()
                    
                    # Combinar todas las máscaras usando OR lógico
                    for mask_data in all_masks[1:]:
                        mask = mask_data['segmentation']
                        # Asegurar que sea booleana
                        if mask.dtype != bool:
                            mask = mask.astype(bool)
                        # OR lógico: cualquier píxel True en cualquier máscara se mantiene True
                        combined_binary_mask = np.logical_or(combined_binary_mask, mask)
                    
                    # Convertir a uint8 (0 o 255) para crear imagen binaria
                    combined_binary_mask = (combined_binary_mask.astype(np.uint8) * 255)
                
                # Guardar imágenes como base64 y como archivos para exportación
                import base64
                import tempfile
                from io import BytesIO
                
                original_img_b64 = ""
                original_with_points_b64 = ""
                final_all_masks_b64 = ""
                final_image_file_path = None  # Ruta al archivo PNG de alta calidad (máscaras visuales)
                binary_mask_file_path = None  # Ruta al archivo PNG de máscara binaria
                
                try:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    original_img_b64 = base64.b64encode(buffered.getvalue()).decode()
                except:
                    pass
                
                try:
                    buffered = BytesIO()
                    original_with_points.save(buffered, format="PNG")
                    original_with_points_b64 = base64.b64encode(buffered.getvalue()).decode()
                except:
                    pass
                
                try:
                    if final_image_all_masks:
                        # Guardar como base64 para visualización
                        buffered = BytesIO()
                        final_image_all_masks.save(buffered, format="PNG")
                        final_all_masks_b64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Guardar como archivo PNG de alta calidad para exportación
                        # Usar el nombre de la imagen sin extensión + "_masks_unified.png"
                        img_name_base = os.path.splitext(img_name)[0]
                        temp_file = tempfile.NamedTemporaryFile(
                            mode='wb', 
                            suffix=f'_{img_name_base}_masks_unified.png', 
                            delete=False
                        )
                        # Guardar en PNG de alta calidad (sin compresión)
                        final_image_all_masks.save(temp_file.name, format="PNG", optimize=False)
                        temp_file.close()
                        final_image_file_path = temp_file.name
                except Exception as e:
                    print(f"Error guardando imagen final: {e}")
                    pass
                
                # Guardar máscara binaria combinada
                try:
                    if combined_binary_mask is not None:
                        # Convertir array numpy a imagen PIL en modo 'L' (escala de grises)
                        binary_mask_image = Image.fromarray(combined_binary_mask, mode='L')
                        
                        # Guardar como archivo PNG
                        img_name_base = os.path.splitext(img_name)[0]
                        temp_file = tempfile.NamedTemporaryFile(
                            mode='wb',
                            suffix=f'_{img_name_base}_binary_mask.png',
                            delete=False
                        )
                        # Guardar en PNG de alta calidad (sin compresión)
                        binary_mask_image.save(temp_file.name, format="PNG", optimize=False)
                        temp_file.close()
                        binary_mask_file_path = temp_file.name
                except Exception as e:
                    print(f"Error guardando máscara binaria: {e}")
                    pass
                
                results[img_name] = {
                    'original_image_b64': original_img_b64,
                    'original_with_points_b64': original_with_points_b64,
                    'final_all_masks_b64': final_all_masks_b64,
                    'final_image_file_path': final_image_file_path,  # Ruta al archivo PNG para exportación (máscaras visuales)
                    'binary_mask_file_path': binary_mask_file_path,  # Ruta al archivo PNG de máscara binaria
                    'image_path': img_data['path'],
                    'masks': all_masks,
                    'points': all_points,
                    'individual_masks': individual_masks,  # Lista de máscaras individuales por punto
                    'status': f"✅ {len(all_masks)} máscaras generadas de {len(points_for_image)} puntos",
                    'num_masks': len(all_masks),
                    'num_points': len(points_for_image)
                }
            else:
                # Error: no se generaron máscaras
                import base64
                from io import BytesIO
                
                original_img_b64 = ""
                try:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    original_img_b64 = base64.b64encode(buffered.getvalue()).decode()
                except:
                    pass
                
                results[img_name] = {
                    'original_image_b64': original_img_b64,
                    'result_image_b64': "",
                    'image_path': img_data['path'],
                    'masks': [],
                    'points': [],
                    'status': "❌ No se pudieron generar máscaras",
                    'num_masks': 0
                }
        
        # Actualizar estado final
        final_status = f"✅ Procesamiento completado: {total_segmentations} segmentación(es) de {total_points} punto(s) en {len(results)} imagen(es)"
        
        # Recopilar archivos de exportación
        export_files = []
        export_info_parts = []
        for img_name, result_data in results.items():
            # Agregar archivo de máscaras visuales (si existe)
            file_path = result_data.get('final_image_file_path')
            if file_path and os.path.exists(file_path):
                export_files.append(file_path)
            
            # Agregar archivo de máscara binaria (si existe)
            binary_path = result_data.get('binary_mask_file_path')
            if binary_path and os.path.exists(binary_path):
                export_files.append(binary_path)
            
            # Agregar información si hay al menos un archivo
            if (file_path and os.path.exists(file_path)) or (binary_path and os.path.exists(binary_path)):
                export_info_parts.append(f"✅ {img_name}")
        
        # Crear mensaje de información
        if export_files:
            num_images = len([name for name in export_info_parts])
            export_info = f"✅ {len(export_files)} archivo(s) lista(s) para descargar ({num_images} imagen(es)):\n" + "\n".join(export_info_parts)
            export_info += f"\n\n📋 Formato: PNG de alta calidad (sin pérdida)"
            export_info += f"\n📏 Resolución: Original de la imagen"
            export_info += f"\n🎨 Tipos de archivo:"
            export_info += f"\n   • Máscaras visuales: Imagen original con máscaras superpuestas en colores"
            export_info += f"\n   • Máscaras binarias: Imagen en blanco/negro (blanco = máscara, negro = fondo)"
            export_info += f"\n💡 Haz clic en los archivos arriba para descargarlos"
        else:
            export_info = "⚠️ No se generaron archivos de exportación"
        
        # Mostrar archivos de exportación si hay
        if export_files:
            yield final_status, results, gr.update(visible=True), gr.update(visible=True, value=export_files), export_info
        else:
            yield final_status, results, gr.update(visible=True), gr.update(visible=False), export_info
    
    def update_results_display(results_dict):
        """Actualiza la visualización de resultados con tabs dinámicos"""
        if not results_dict or len(results_dict) == 0:
            return gr.update(visible=False)
        
        # Debug: verificar que tenemos todas las imágenes
        print(f"[update_results_display] Procesando {len(results_dict)} imágenes: {list(results_dict.keys())}")
        for img_name, result_data in results_dict.items():
            print(f"  - Imagen: '{img_name}' (tipo: {type(img_name)}, len: {len(str(img_name)) if img_name else 0})")
        
        # Crear HTML con tabs usando JavaScript
        html_content = """
        <style>
            .batch-results-container {
                padding: 20px;
            }
            .batch-image-section {
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
                background: white;
            }
            .batch-image-section h3 {
                color: #333;
                margin-top: 0;
                padding-bottom: 15px;
                border-bottom: 2px solid #007bff;
            }
            .batch-image-container {
                margin: 20px 0;
            }
            .batch-image-box {
                text-align: center;
                margin: 20px 0;
            }
            .batch-image-box img {
                max-width: 100%;
                height: auto;
                border: 2px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .batch-info {
                background: #f9f9f9;
                color: #333 !important;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .batch-info p {
                color: #333 !important;
                margin: 5px 0;
            }
            .batch-info strong {
                color: #333 !important;
            }
            .batch-info table {
                color: #333 !important;
            }
            .batch-info th, .batch-info td {
                color: #333 !important;
            }
        </style>
        <div class="batch-results-container">
        """
        
        # Convertir a lista para asegurar orden consistente
        items_list = list(results_dict.items())
        
        # Agregar contenido de cada imagen (una debajo de la otra)
        for idx, (img_name, result_data) in enumerate(items_list):
            # Asegurar que el nombre no esté vacío
            img_name_str = str(img_name) if img_name is not None else ""
            display_name = img_name_str.strip() if img_name_str.strip() else f"Imagen {idx + 1}"
            
            # Imágenes en base64
            original_with_points_str = f"data:image/png;base64,{result_data.get('original_with_points_b64', '')}" if result_data.get('original_with_points_b64') else ""
            final_all_masks_str = f"data:image/png;base64,{result_data.get('final_all_masks_b64', '')}" if result_data.get('final_all_masks_b64') else ""
            
            html_content += f"""
            <div class="batch-image-section">
                <h3>📷 {display_name}</h3>
                
                <!-- Imagen original con todos los puntos -->
                <div style="margin: 20px 0;">
                    <h4 style="color: #333;">1️⃣ Imagen Original con Puntos Seleccionados</h4>
                    <div style="text-align: center; margin: 10px 0;">
                        <img src="{original_with_points_str}" alt="Original con puntos" style="max-width: 100%; border: 2px solid #ddd; border-radius: 5px;">
                    </div>
                    <p style="text-align: center; color: #333;">Total de puntos: {result_data.get('num_points', 0)}</p>
                </div>
                
                <hr style="margin: 30px 0;">
                
                <!-- Máscaras individuales por punto -->
                <h4 style="color: #333;">2️⃣ Máscaras Individuales (Máscara seleccionada de cada segmentación)</h4>
            """
            
            # Mostrar cada máscara individual
            individual_masks = result_data.get('individual_masks', [])
            if individual_masks:
                html_content += "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;'>"
                for i, mask_info in enumerate(individual_masks):
                    point = mask_info.get('point', [0, 0])
                    point_idx = mask_info.get('point_index', i + 1)
                    mask_img_b64 = mask_info.get('image_b64', '')
                    mask_data = mask_info.get('mask', {})
                    
                    html_content += f"""
                    <div style='border: 2px solid #ddd; border-radius: 8px; padding: 10px; text-align: center;'>
                        <h5 style='color: #333;'>Punto {point_idx} - Coordenadas: ({point[0]}, {point[1]})</h5>
                        <img src="data:image/png;base64,{mask_img_b64}" alt="Máscara {point_idx}" style="max-width: 100%; border: 1px solid #ccc; border-radius: 5px;">
                        <p style='margin-top: 10px; font-size: 0.9em; color: #333;'>
                            Área: {mask_data.get('area', 0)} px² | Score: {mask_data.get('stability_score', 0.0):.3f}
                        </p>
                    </div>
                    """
                html_content += "</div>"
            else:
                html_content += "<p style='color: #333;'>No hay máscaras individuales para mostrar</p>"
            
            html_content += f"""
                <hr style="margin: 30px 0;">
                
                <!-- Imagen final con todas las máscaras superpuestas -->
                <div style="margin: 20px 0;">
                    <h4 style="color: #333;">3️⃣ Imagen Final con Todas las Máscaras Superpuestas</h4>
                    <div style="text-align: center; margin: 10px 0;">
                        <img src="{final_all_masks_str}" alt="Todas las máscaras" style="max-width: 100%; border: 2px solid #ddd; border-radius: 5px;">
                    </div>
                    <p style="text-align: center; color: #333;">Total de máscaras: {result_data.get('num_masks', 0)}</p>
                    <p style="text-align: center; color: #333; font-size: 0.9em;">
                        <strong style="color: #333;">Formato:</strong> PNG de alta calidad (sin pérdida) | 
                        <strong style="color: #333;">Resolución:</strong> Original | 
                        <strong style="color: #333;">Archivo:</strong> Disponible en la sección de descarga abajo
                    </p>
                </div>
                
                <div class="batch-info" style="margin-top: 20px; color: #333 !important;">
                    <p style="color: #333 !important;"><strong style="color: #333 !important;">📊 Estado:</strong> <span style="color: #333 !important;">{result_data['status']}</span></p>
            """
            
            if result_data.get('masks'):
                masks = result_data['masks']
                html_content += f"<p style='color: #333 !important;'><strong style='color: #333 !important;'>🎭 Resumen de Máscaras:</strong></p>"
                html_content += "<table style='width: 100%; margin-top: 10px; border-collapse: collapse; background: white; color: #333;'><thead><tr style='background: #e0e0e0;'><th style='padding: 8px; text-align: left; border: 1px solid #ccc; color: #333;'>Máscara</th><th style='padding: 8px; text-align: left; border: 1px solid #ccc; color: #333;'>Punto</th><th style='padding: 8px; text-align: left; border: 1px solid #ccc; color: #333;'>Área</th><th style='padding: 8px; text-align: left; border: 1px solid #ccc; color: #333;'>Stability Score</th></tr></thead><tbody>"
                for i, mask_data in enumerate(masks):
                    point = result_data.get('points', [])[i] if i < len(result_data.get('points', [])) else [0, 0]
                    html_content += f"<tr style='background: {'#f5f5f5' if i % 2 == 0 else 'white'};'><td style='padding: 8px; border: 1px solid #ccc; color: #333;'>Máscara {i+1}</td><td style='padding: 8px; border: 1px solid #ccc; color: #333;'>({point[0]}, {point[1]})</td><td style='padding: 8px; border: 1px solid #ccc; color: #333;'>{mask_data.get('area', 0)} px²</td><td style='padding: 8px; border: 1px solid #ccc; color: #333;'>{mask_data.get('stability_score', 0.0):.3f}</td></tr>"
                html_content += "</tbody></table>"
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </div>
        """
        
        return gr.update(visible=True, value=html_content)
    
    # Retornar el contenido y las referencias para conectar eventos
    return (
        batch_content,
        {
            'model_dropdown': batch_model_dropdown,
            'load_btn': batch_load_btn,
            'model_info': batch_model_info,
            'image_upload': batch_image_upload,
            'images_info': batch_images_info,
            'points_per_side': batch_points_per_side,
            'pred_iou_thresh': batch_pred_iou_thresh,
            'stability_score_thresh': batch_stability_score_thresh,
            'mask_index': batch_mask_index,
            'process_btn': batch_process_btn,
            'status': batch_status,
            'results_html': batch_results_html,
            'export_files': batch_export_files,
            'export_info': batch_export_info,
            'update_results_display': update_results_display,
            'selected_model_state': selected_model_state,
            'images_state': images_state,
            'image_points_state': image_points_state,
            'processing_results_state': processing_results_state,
            'images_container': batch_images_container,
            'image_components': batch_image_components_list,
            'original_images_state': batch_original_images_state,
            'points_info': batch_points_info,
            'undo_point_btn': batch_undo_point_btn,
            'clear_points_btn': batch_clear_points_btn,
            'on_model_load': on_batch_model_load,
            'on_images_upload': on_images_upload,
            'update_image_with_points': update_image_with_points,
            'create_image_click_handler': create_image_click_handler,
            'undo_last_point_batch': undo_last_point_batch,
            'clear_all_points': clear_all_points,
            'process_batch_images': process_batch_images
        }
    )
