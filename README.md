# SAM 2.1 - Segmentación de Imágenes

Aplicación de segmentación de imágenes usando SAM 2.1 (Segment Anything Model 2.1) de Meta AI.

## Requisitos

- Python 3.8 o superior
- CUDA (opcional, para GPU)

## Instalación

1. Clonar o descargar el repositorio

2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv
```

3. Activar el entorno virtual:
   - Windows (PowerShell): `venv\Scripts\Activate.ps1`
   - Windows (CMD): `venv\Scripts\activate.bat`
   - Linux/Mac: `source venv/bin/activate`

4. Instalar dependencias:
```bash
pip install -r requirements.txt
```

5. Descargar modelos y configuraciones:
```bash
python download_sam2.py
```

## Uso

1. Ejecutar la aplicación:
```bash
python app.py
```

2. Abrir el navegador en la URL que se muestra (por defecto: http://127.0.0.1:7860)

3. En la interfaz:
   - Seleccionar un modelo y hacer clic en "Cargar Modelo"
   - Subir una imagen
   - Opcional: Seleccionar puntos en la imagen (Foreground/Background) haciendo clic
   - Hacer clic en "SEGMENTAR" para generar las máscaras

## Funcionalidades

- Segmentación automática de imágenes
- Segmentación con puntos (Foreground/Background)
- Visualización de máscaras con colores
- Exportación de máscaras a JSON
- Selección de múltiples modelos (Tiny, Small, Base+, Large)
