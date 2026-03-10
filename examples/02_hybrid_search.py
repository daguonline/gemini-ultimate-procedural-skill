"""
02_hybrid_search.py - Ejemplo completo de búsqueda híbrida con Vertex AI.

La búsqueda híbrida combina:
- Búsqueda semántica (embeddings densos): encuentra por SIGNIFICADO
- Búsqueda por tokens (embeddings dispersos TF-IDF): encuentra por PALABRAS CLAVE

Usa Reciprocal Rank Fusion (RRF) para fusionar ambos resultados.

⚠️ Requiere un proyecto de Google Cloud con la API de Vertex AI habilitada.
"""

import sys
import os
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.embeddings import (
    get_dense_embedding,
    train_sparse_vectorizer,
    get_sparse_embedding,
)
from scripts.hybrid_search import hybrid_query, print_hybrid_results
from scripts.vector_search import (
    create_endpoint,
    deploy_index,
    cleanup,
)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================
PROJECT_ID = "tu-proyecto-id"        # ← Cambia esto
LOCATION = "us-east1"                # ← Tu región
UID = datetime.now().strftime("%m%d%H%M")
BUCKET_URI = f"gs://{PROJECT_ID}"
DEPLOYED_INDEX_ID = f"hybrid_search_deployed_{UID}"

# =============================================================================
# PASO 1: Inicializar
# =============================================================================
print("PASO 1: Inicializando...")
from google.cloud import aiplatform
from google import genai

aiplatform.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def get_dense_emb(text):
    response = client.models.embed_content(model="gemini-embedding-001", contents=[text])
    return response.embeddings[0].values


# =============================================================================
# PASO 2: Cargar datos de ejemplo
# =============================================================================
print("PASO 2: Cargando datos...")
CSV_URL = "https://storage.googleapis.com/spls/gsp1297/google_merch_shop_items.csv"
df = pd.read_csv(CSV_URL)
print(f"  Cargados {len(df)} productos")

# =============================================================================
# PASO 3: Entrenar vectorizador disperso y generar embeddings
# =============================================================================
print("PASO 3: Generando embeddings densos y dispersos...")
corpus = df.title.tolist()
vectorizer = train_sparse_vectorizer(corpus)

items = []
for i in range(len(df)):
    items.append({
        "id": i,
        "title": df.title[i],
        "embedding": get_dense_emb(df.title[i]),
        "sparse_embedding": get_sparse_embedding(vectorizer, df.title[i]),
    })
print(f"  Generados {len(items)} items con embeddings duales")

# =============================================================================
# PASO 4: Crear índice híbrido y endpoint
# =============================================================================
print("PASO 4: Creando índice híbrido... (esto toma varios minutos)")
import json

with open("items.json", "w") as f:
    for item in items:
        f.write(f"{item}\n")

os.system(f"gsutil cp items.json {BUCKET_URI}")

from google.cloud import aiplatform_v1

api_endpoint = f"{LOCATION}-aiplatform.googleapis.com"
gapic_client = aiplatform_v1.IndexServiceClient(
    client_options={"api_endpoint": api_endpoint}
)

index_request = {
    "display_name": f"hybrid-search-index-{UID}",
    "metadata": {
        "contentsDeltaUri": BUCKET_URI,
        "config": {
            "dimensions": 3072,
            "approximateNeighborsCount": 10,
            "distanceMeasureType": "DOT_PRODUCT_DISTANCE",
            "algorithmConfig": {
                "treeAhConfig": {
                    "leafNodeEmbeddingCount": 1000,
                    "leafNodesToSearchPercent": 10,
                }
            },
        },
    },
}

parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
operation = gapic_client.create_index(parent=parent, index=index_request)
response = operation.result(timeout=1800)
my_hybrid_index = aiplatform.MatchingEngineIndex(index_name=response.name)
print(f"  Índice creado: {my_hybrid_index.resource_name}")

my_endpoint = create_endpoint(f"hybrid-search-endpoint-{UID}")
deploy_index(my_endpoint, my_hybrid_index, DEPLOYED_INDEX_ID)

# =============================================================================
# PASO 5: Consulta híbrida
# =============================================================================
print("\nPASO 5: Ejecutando consulta híbrida...")
query_text = "Kids"
query_dense = get_dense_emb(query_text)
query_sparse = get_sparse_embedding(vectorizer, query_text)

results = hybrid_query(
    my_endpoint, DEPLOYED_INDEX_ID,
    query_dense, query_sparse, alpha=0.5
)

print(f"\n  Consulta: '{query_text}'")
print_hybrid_results(results, df)

# =============================================================================
# PASO 6: Limpiar
# =============================================================================
print("\nPASO 6: Limpiando recursos...")
cleanup(my_endpoint, my_hybrid_index, BUCKET_URI)
print("\n¡Ejemplo completado!")
