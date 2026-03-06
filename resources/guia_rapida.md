# Guía Rápida: Vertex AI Vector Search

Referencia rápida de las funciones principales del skill.

## Configuración

```python
from google import genai
from google.cloud import aiplatform

PROJECT_ID = "tu-proyecto"
LOCATION = "us-east1"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)
```

## Embeddings

```python
from scripts.embeddings import *

# Denso (semántico)
emb = get_dense_embedding(client, "texto")
embs = get_dense_embeddings_batch(client, ["texto1", "texto2"])

# Disperso (tokens)
vectorizer = train_sparse_vectorizer(["texto1", "texto2", ...])
sparse = get_sparse_embedding(vectorizer, "texto")
```

## Índice y Endpoint

```python
from scripts.vector_search import *

save_embeddings_to_gcs(df, BUCKET_URI, mode="dense")  # o "hybrid"
index = create_index("mi-indice", BUCKET_URI, dimensions=3072)
endpoint = create_endpoint("mi-endpoint")
deploy_index(endpoint, index, "mi_deploy_id")
```

## Consultas

```python
# Semántica
results = semantic_query(endpoint, "mi_deploy_id", query_embedding)

# Híbrida
from scripts.hybrid_search import hybrid_query
results = hybrid_query(endpoint, "mi_deploy_id", dense_emb, sparse_emb, alpha=0.5)
```

## Limpieza

```python
cleanup(endpoint, index, BUCKET_URI)
```
