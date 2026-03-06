"""
03_cleanup.py - Script de limpieza de recursos de Vertex AI Vector Search.

Usa este script cuando quieras limpiar índices y endpoints existentes
que ya no necesites, para evitar costos inesperados.

Puedes encontrar los IDs de tus recursos en:
- Índices: https://console.cloud.google.com/vertex-ai/matching-engine/indexes
- Endpoints: https://console.cloud.google.com/vertex-ai/matching-engine/index-endpoints
"""

from google.cloud import aiplatform

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
PROJECT_ID = "tu-proyecto-id"        # ← Cambia esto
LOCATION = "us-east1"                # ← Tu región

# IDs de los recursos a limpiar (obtenlos de la consola de GCP)
INDEX_ID = "[tu-index-id]"           # ← Cambia esto
ENDPOINT_ID = "[tu-endpoint-id]"     # ← Cambia esto
BUCKET_URI = "[gs://tu-bucket]"      # ← Cambia esto (opcional)

# =============================================================================
# EJECUTAR LIMPIEZA
# =============================================================================
aiplatform.init(project=PROJECT_ID, location=LOCATION)

print("Recuperando recursos existentes...")
my_index = aiplatform.MatchingEngineIndex(INDEX_ID)
my_endpoint = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT_ID)

print("Eliminando despliegues del endpoint...")
my_endpoint.undeploy_all()

print("Eliminando endpoint...")
my_endpoint.delete(force=True)

print("Eliminando índice...")
my_index.delete()

if BUCKET_URI and not BUCKET_URI.startswith("["):
    import subprocess
    print(f"Eliminando bucket {BUCKET_URI}...")
    subprocess.run(["gsutil", "rm", "-r", BUCKET_URI], capture_output=True)

print("✅ Limpieza completada.")
