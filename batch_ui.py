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
from segmentation import segment_image, preprocess_image
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
    processing_results_state = gr.State(value={})  # Diccionario con resultados por imagen
    
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
    
    # Funciones de eventos
    def on_batch_model_load(model_name):
        """Carga el modelo seleccionado"""
        status = load_model(model_name)
        if "✅" in status:
            return status, model_name, gr.update(visible=True)
        return status, None, gr.update(visible=False)
    
    def on_images_upload(files):
        """Maneja la carga de múltiples imágenes"""
        if files is None or len(files) == 0:
            return "Esperando imágenes...", [], gr.update(visible=False)
        
        # Cargar imágenes
        images = []
        for file_path in files:
            try:
                img = Image.open(file_path)
                images.append({
                    'image': img,
                    'path': file_path,
                    'name': os.path.basename(file_path)
                })
            except Exception as e:
                print(f"Error cargando imagen {file_path}: {e}")
        
        if len(images) == 0:
            return "⚠️ No se pudieron cargar las imágenes", [], gr.update(visible=False)
        
        info_text = f"✅ {len(images)} imagen(es) cargada(s) correctamente"
        return info_text, images, gr.update(visible=True)
    
    def process_batch_images(model_name, images, points_per_side, pred_iou_thresh, stability_score_thresh):
        """Procesa todas las imágenes en batch"""
        if not images or len(images) == 0:
            return "⚠️ No hay imágenes para procesar", {}, gr.update(visible=False)
        
        if model_name is None:
            return "⚠️ Por favor, carga un modelo primero", {}, gr.update(visible=False)
        
        # Asegurar que el modelo esté cargado
        success, status = ensure_model_loaded(model_name)
        if not success:
            return f"❌ {status}", {}, gr.update(visible=False)
        
        results = {}
        total = len(images)
        
        # Procesar cada imagen
        for idx, img_data in enumerate(images):
            image = img_data['image']
            img_name = img_data['name']
            
            # Actualizar estado
            status_msg = f"🔄 Procesando imagen {idx + 1}/{total}: {img_name}"
            yield status_msg, results, gr.update(visible=False)
            
            # Segmentar imagen
            result = segment_image(
                image, 
                model_name, 
                points_per_side, 
                pred_iou_thresh, 
                stability_score_thresh
            )
            
            if result[0] is not None:
                result_image, status, masks, image_np = result
                mask_info_list = get_mask_info_list(masks) if masks else []
                
                # Guardar imágenes como base64 para serialización
                import base64
                from io import BytesIO
                
                original_img_b64 = ""
                result_img_b64 = ""
                
                try:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    original_img_b64 = base64.b64encode(buffered.getvalue()).decode()
                except:
                    pass
                
                try:
                    if result_image:
                        buffered = BytesIO()
                        result_image.save(buffered, format="PNG")
                        result_img_b64 = base64.b64encode(buffered.getvalue()).decode()
                except:
                    pass
                
                results[img_name] = {
                    'original_image_b64': original_img_b64,
                    'result_image_b64': result_img_b64,
                    'image_path': img_data['path'],
                    'mask_info_list': mask_info_list,
                    'status': status,
                    'num_masks': len(mask_info_list) if mask_info_list else 0
                }
            else:
                # Guardar imagen original como base64 incluso en caso de error
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
                    'mask_info_list': [],
                    'status': result[1] if len(result) > 1 else "Error desconocido",
                    'num_masks': 0
                }
        
        # Los resultados se mostrarán en el contenedor
        # Actualizar estado final con todos los resultados
        final_status = f"✅ Procesamiento completado: {total} imagen(es) procesada(s)"
        # El último yield debe incluir todos los resultados completos
        yield final_status, results, gr.update(visible=True)
    
    def update_results_display(results_dict):
        """Actualiza la visualización de resultados con tabs dinámicos"""
        if not results_dict or len(results_dict) == 0:
            return gr.update(visible=False)
        
        # Crear HTML con tabs usando JavaScript
        html_content = """
        <style>
            .batch-tabs {
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 20px 0;
            }
            .batch-tab-buttons {
                display: flex;
                flex-wrap: wrap;
                background: #f5f5f5;
                border-bottom: 1px solid #ddd;
                padding: 10px;
                gap: 5px;
            }
            .batch-tab-button {
                padding: 10px 20px;
                background: white;
                border: 1px solid #ddd;
                border-radius: 5px 5px 0 0;
                cursor: pointer;
                margin-bottom: -1px;
            }
            .batch-tab-button.active {
                background: #007bff;
                color: white;
                border-color: #007bff;
            }
            .batch-tab-content {
                display: none;
                padding: 20px;
            }
            .batch-tab-content.active {
                display: block;
            }
            .batch-image-container {
                display: flex;
                gap: 20px;
                margin: 20px 0;
            }
            .batch-image-box {
                flex: 1;
                text-align: center;
            }
            .batch-image-box img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .batch-info {
                background: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
        <div class="batch-tabs">
            <div class="batch-tab-buttons" id="tab-buttons">
        """
        
        # Agregar botones de tabs
        for idx, img_name in enumerate(results_dict.keys()):
            active_class = "active" if idx == 0 else ""
            html_content += f'<button class="batch-tab-button {active_class}" onclick="showBatchTab({idx})">{img_name}</button>'
        
        html_content += """
            </div>
        """
        
        # Agregar contenido de cada tab
        for idx, (img_name, result_data) in enumerate(results_dict.items()):
            active_class = "active" if idx == 0 else ""
            
            # Usar imágenes en base64 que ya están guardadas
            original_img_str = f"data:image/png;base64,{result_data.get('original_image_b64', '')}" if result_data.get('original_image_b64') else ""
            result_img_str = f"data:image/png;base64,{result_data.get('result_image_b64', '')}" if result_data.get('result_image_b64') else ""
            
            html_content += f"""
            <div class="batch-tab-content {active_class}" id="tab-content-{idx}">
                <h3>📷 {img_name}</h3>
                <div class="batch-image-container">
                    <div class="batch-image-box">
                        <h4>Imagen Original</h4>
                        <img src="{original_img_str}" alt="Original">
                    </div>
                    <div class="batch-image-box">
                        <h4>Resultado de Segmentación</h4>
                        <img src="{result_img_str}" alt="Resultado">
                    </div>
                </div>
                <div class="batch-info">
                    <p><strong>📊 Estado:</strong> {result_data['status']}</p>
            """
            
            if result_data.get('mask_info_list'):
                html_content += f"<p><strong>🎭 Máscaras detectadas:</strong> {len(result_data['mask_info_list'])}</p>"
                html_content += "<table style='width: 100%; margin-top: 10px;'><tr><th>Máscara</th><th>Área</th><th>Stability Score</th></tr>"
                for i, mask_info in enumerate(result_data['mask_info_list'][:10]):  # Mostrar solo las primeras 10
                    html_content += f"<tr><td>Máscara {i+1}</td><td>{mask_info.get('area', 0)} px²</td><td>{mask_info.get('stability_score', 0.0):.3f}</td></tr>"
                html_content += "</table>"
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </div>
        <script>
            function showBatchTab(index) {
                // Ocultar todos los contenidos
                document.querySelectorAll('.batch-tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                // Remover active de todos los botones
                document.querySelectorAll('.batch-tab-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                // Mostrar el contenido seleccionado
                document.getElementById('tab-content-' + index).classList.add('active');
                // Activar el botón correspondiente
                document.querySelectorAll('.batch-tab-button')[index].classList.add('active');
            }
        </script>
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
            'process_btn': batch_process_btn,
            'status': batch_status,
            'results_html': batch_results_html,
            'update_results_display': update_results_display,
            'selected_model_state': selected_model_state,
            'images_state': images_state,
            'processing_results_state': processing_results_state,
            'on_model_load': on_batch_model_load,
            'on_images_upload': on_images_upload,
            'process_batch_images': process_batch_images
        }
    )
