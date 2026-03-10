"""
01_semantic_search.py - Ejemplo completo de búsqueda semántica con Vertex AI.

Este script demuestra el flujo completo:
1. Configurar → 2. Cargar datos → 3. Generar embeddings → 4. Indexar → 5. Consultar → 6. Limpiar

⚠️ Requiere un proyecto de Google Cloud con la API de Vertex AI habilitada.
⚠️ Genera costos mientras el endpoint esté activo. Ejecuta la limpieza al final.
"""

import sys
import os
from datetime import datetime

import pandas as pd

# Asegurar que podemos importar los módulos del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.embeddings import get_dense_embedding, get_dense_embeddings_batch
from scripts.vector_search import (
    save_embeddings_to_gcs,
    create_index,
    create_endpoint,
    deploy_index,
    semantic_query,
    cleanup,
)


# =============================================================================
# CONFIGURACIÓN - Modifica estos valores
# =============================================================================
PROJECT_ID = "tu-proyecto-id"        # ← Cambia esto
LOCATION = "us-east1"                # ← Tu región
UID = datetime.now().strftime("%m%d%H%M")
BUCKET_URI = f"gs://{PROJECT_ID}-semantic-search-{UID}"
DEPLOYED_INDEX_ID = f"semantic_search_deployed_{UID}"

# =============================================================================
# PASO 1: Inicializar clientes
# =============================================================================
print("=" * 60)
print("PASO 1: Inicializando clientes...")
print("=" * 60)

from google import genai
from google.cloud import aiplatform

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# =============================================================================
# PASO 2: Cargar datos
# =============================================================================
print("\nPASO 2: Cargando datos de ejemplo...")
# Ejemplo: usando la BD pública de Stack Overflow desde BigQuery
# Puedes reemplazar esto con: df = pd.read_csv("tus_datos.csv")
from google.cloud import bigquery

bq_client = bigquery.Client(project=PROJECT_ID)
QUERY = """
    SELECT DISTINCT q.id, q.title
    FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
    WHERE q.tags LIKE '%python%'
    LIMIT 100
"""
df = bq_client.query(QUERY).result().to_dataframe()
print(f"  Cargadas {len(df)} filas")

# =============================================================================
# PASO 3: Generar embeddings
# =============================================================================
print("\nPASO 3: Generando embeddings densos...")
embeddings = get_dense_embeddings_batch(client, df["title"].tolist())
df["embedding"] = embeddings
print(f"  Generados {len(embeddings)} embeddings de {len(embeddings[0])} dimensiones")

# =============================================================================
# PASO 4: Subir a GCS y crear índice
# =============================================================================
print("\nPASO 4: Subiendo a Cloud Storage y creando índice...")
save_embeddings_to_gcs(df, BUCKET_URI, mode="dense")
my_index = create_index(
    display_name=f"semantic-search-index-{UID}",
    bucket_uri=BUCKET_URI,
    dimensions=len(embeddings[0]),
)

# =============================================================================
# PASO 5: Crear endpoint y desplegar
# =============================================================================
print("\nPASO 5: Creando endpoint y desplegando índice...")
my_endpoint = create_endpoint(f"semantic-search-endpoint-{UID}")
deploy_index(my_endpoint, my_index, DEPLOYED_INDEX_ID)

# =============================================================================
# PASO 6: Consultar
# =============================================================================
print("\nPASO 6: Ejecutando consulta de prueba...")
test_question = "How to read JSON files in Python?"
query_emb = get_dense_embedding(client, test_question)
results = semantic_query(my_endpoint, DEPLOYED_INDEX_ID, query_emb)

print(f"\n  Pregunta: '{test_question}'")
print(f"  Top {len(results)} resultados similares:")
for i, neighbor in enumerate(results):
    row = df[df["id"].astype(str) == str(neighbor.id)]
    if not row.empty:
        print(f"    {i+1}. [{neighbor.distance:.4f}] {row.title.values[0]}")

# =============================================================================
# PASO 7: Limpiar recursos
# =============================================================================
print("\nPASO 7: Limpiando recursos...")
cleanup(my_endpoint, my_index, BUCKET_URI)
print("\n¡Ejemplo completado!")
