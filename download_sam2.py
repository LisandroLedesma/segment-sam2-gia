import os
import urllib.request
import sys
from pathlib import Path

def download_with_progress(url, destination):
    """Descarga con barra de progreso"""
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  Progreso: {percent}%")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook)
        print()  # Nueva línea
        return True
    except KeyboardInterrupt:
        print("\n⚠️  Descarga cancelada por el usuario")
        if os.path.exists(destination):
            os.remove(destination)
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

# Crear directorios
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("configs", exist_ok=True)

print("="*70)
print("📥 DESCARGANDO SAM 2.1 (Versión actualizada)")
print("="*70)

# ============================================================================
# OPCIÓN 1: SAM 2.1 (RECOMENDADO - Septiembre 2024)
# ============================================================================

# URLs de los YAMLs de SAM 2.1 (ubicación correcta)
yaml_configs_2_1 = {
    "sam2.1_hiera_t.yaml": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_s.yaml": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_b+.yaml": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_l.yaml": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
}

# URLs de los checkpoints de SAM 2.1
model_checkpoints_2_1 = {
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

# Información de modelos para tu app (SAM 2.1)
models_info = {
    "tiny": {
        "config": "configs/sam2.1_hiera_t.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt"
    },
    "small": {
        "config": "configs/sam2.1_hiera_s.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_small.pt"
    },
    "base_plus": {
        "config": "configs/sam2.1_hiera_b+.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_base_plus.pt"
    },
    "large": {
        "config": "configs/sam2.1_hiera_l.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_large.pt"
    }
}

# Descargar YAMLs de SAM 2.1
print("\n📝 Descargando configuraciones YAML (SAM 2.1)...")
for filename, url in yaml_configs_2_1.items():
    filepath = Path("configs") / filename
    if filepath.exists():
        print(f"⏭️  {filename} ya existe")
        continue
    
    print(f"⬇️  Descargando {filename}...")
    if download_with_progress(url, filepath):
        # Verificar contenido
        with open(filepath, 'r') as f:
            content = f.read()
            if "404" in content or len(content) < 100:
                print(f"❌ Error: {filename} no válido")
                filepath.unlink()
            else:
                print(f"✅ {filename} descargado ({len(content)} bytes)")

# Descargar Checkpoints de SAM 2.1
print("\n🧠 Descargando modelos SAM 2.1...")
for filename, url in model_checkpoints_2_1.items():
    filepath = Path("checkpoints") / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"⏭️  {filename} ya existe ({size_mb:.1f} MB)")
        continue
    
    print(f"⬇️  Descargando {filename}...")
    if download_with_progress(url, filepath):
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✅ {filename} descargado ({size_mb:.1f} MB)")

print("\n" + "="*70)
print("🎉 DESCARGA COMPLETADA")
print("="*70)

# Verificar archivos
print("\n📋 Verificación:")
print("\n📝 Configuraciones:")
for filename in yaml_configs_2_1.keys():
    filepath = Path("configs") / filename
    if filepath.exists():
        size = filepath.stat().st_size
        print(f"  ✅ {filename} ({size} bytes)")
    else:
        print(f"  ❌ {filename} FALTA")

print("\n🧠 Checkpoints:")
for filename in model_checkpoints_2_1.keys():
    filepath = Path("checkpoints") / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✅ {filename} ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ {filename} FALTA")

print("\n💡 Usar en tu código:")
print("models_info = {")
for name, info in models_info.items():
    print(f'    "{name}": {{')
    print(f'        "config": "{info["config"]}",')
    print(f'        "checkpoint": "{info["checkpoint"]}"')
    print(f'    }},')
print("}")

print(f"\n📁 Archivos en: {os.path.abspath('.')}")