name: Gemini Ultimate Procedural Skill
description: Skill definitiva para implementar búsqueda semántica, multimodal e híbrida con Vertex AI Vector Search y Gemini 3.1.

# Gemini Ultimate Procedural Skill

Este skill definitivo te permitirá implementar flujos de búsqueda vectorial de última generación (semántica, por tokens, híbrida y **multimodal**) usando Gemini 3.1 en Google Cloud Vertex AI.

## Requisitos Previos

- Un proyecto de Google Cloud con la API de Vertex AI habilitada.
- Python 3.10+ con los paquetes listados en `requirements.txt`.
- Credenciales de Google Cloud configuradas (`gcloud auth login`).

## Flujo General

```
1. CONFIGURAR → Establecer PROJECT_ID, LOCATION, inicializar clientes.
2. PREPARAR DATOS → Cargar tus datos (CSV, BigQuery, etc.) en un DataFrame.
3. GENERAR EMBEDDINGS → Convertir texto a vectores (densos y/o dispersos).
4. SUBIR A GCS → Guardar como JSONL y subir a un bucket de Cloud Storage.
5. CREAR ÍNDICE → Construir un índice en Vector Search.
6. CREAR ENDPOINT → Crear un Index Endpoint público.
7. DESPLEGAR → Desplegar el índice en el endpoint.
8. CONSULTAR → Ejecutar búsquedas por similitud.
9. LIMPIAR → Eliminar recursos para evitar costos.
```

## Instrucciones Paso a Paso

### Paso 1: Configurar el entorno

```python
PROJECT_ID = "tu-proyecto-id"
LOCATION = "us-east1"  # o la región que uses

# Para embeddings densos (Gemini)
from google import genai
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# Para Vector Search
from google.cloud import aiplatform
aiplatform.init(project=PROJECT_ID, location=LOCATION)
```

### Paso 2: Generar embeddings densos (semánticos)

Usa el módulo `scripts/embeddings.py`:

```python
from scripts.embeddings import get_dense_embedding, get_dense_embeddings_batch

# Un solo texto
embedding = get_dense_embedding(client, "¿Cómo leer JSON con Python?")

# En lotes (para datasets grandes)
embeddings = get_dense_embeddings_batch(client, lista_de_textos)
```

### Paso 3: Generar embeddings dispersos (para búsqueda por tokens)

```python
from scripts.embeddings import train_sparse_vectorizer, get_sparse_embedding

# Entrenar el vectorizador con tu corpus
vectorizer = train_sparse_vectorizer(lista_de_textos)

# Generar embedding disperso para un texto
sparse_emb = get_sparse_embedding(vectorizer, "Chrome Dino Pin")
```

### Paso 4: Preparar datos y subir a Cloud Storage

```python
from scripts.vector_search import save_embeddings_to_gcs

# Para búsqueda semántica (solo densos)
save_embeddings_to_gcs(df, BUCKET_URI, mode="dense")

# Para búsqueda híbrida (densos + dispersos)
save_embeddings_to_gcs(df, BUCKET_URI, mode="hybrid")
```

### Paso 5: Crear índice y endpoint

```python
from scripts.vector_search import create_index, create_endpoint, deploy_index

# Crear índice
my_index = create_index(
    display_name="mi-indice",
    bucket_uri=BUCKET_URI,
    dimensions=3072  # Recomendado: 3072 para multimodal-embedding-001 o gemini-embedding-001
)

# Crear endpoint
my_endpoint = create_endpoint("mi-endpoint")

# Desplegar
deploy_index(my_endpoint, my_index, "mi_indice_desplegado")
```

### Paso 6: Consultar

```python
from scripts.vector_search import semantic_query
from scripts.hybrid_search import hybrid_query

# Búsqueda semántica
results = semantic_query(my_endpoint, "mi_indice_desplegado", query_embedding)

# Búsqueda híbrida
results = hybrid_query(
    my_endpoint, "mi_indice_desplegado",
    dense_embedding, sparse_embedding, alpha=0.5
)
```

### Paso 7: Limpiar recursos

```python
from scripts.vector_search import cleanup
cleanup(my_endpoint, my_index, BUCKET_URI)
```

## Cuándo usar cada tipo de búsqueda

| Tipo | Caso de uso | Ejemplo |
|------|------------|---------|
| **Semántica** | Buscar por significado | "¿Cómo leer archivos?" encuentra docs sobre parsing JSON |
| **Por Tokens** | Buscar por palabras exactas | "SKU-12345" encuentra ese producto específico |
| **Híbrida** | Combinar ambas ventajas | "Kids sunglasses" encuentra tanto por palabra "Kids" como por significado "Youth" |

## Patrones Avanzados (Habilidad Definitiva)

### 1. Razonamiento Profundo (Thinking Level)
Para búsquedas que requieren analizar lógica compleja (ej: discrepancias legales):
```python
config = GenerateContentConfig(thinking_config={"thinking_level": "high"})
response = client.models.generate_content(model="gemini-3.1-pro", contents=query, config=config)
```

### 2. RAG Multimodal Avanzado
Para documentos con contenido mixto (tablas, diagramas, imágenes):
- **Extracción de Metadatos**: Usa `get_document_metadata` con sus 4 argumentos clave:
  ```python
  text_df, img_df = get_document_metadata(model, pdf_path, save_dir, prompt)
  ```
- **Búsqueda Semántica de Fragmentos**: Recupera fragmentos (chunks) usando similitud de coseno.
- **Construcción de Contexto**: Une los fragmentos recuperados en un solo string de contexto antes de enviarlo a Gemini para mayor fidelidad.

### 3. Salida Estructurada para Automatización
Asegura que tu sistema de búsqueda devuelva JSON puro:
```python
config = GenerateContentConfig(response_mime_type="application/json")
```

### 4. Búsqueda con Context Caching
Si consultas repetidamente el mismo corpus masivo (ej: 1M+ tokens), habilita **Context Caching** para reducir latencia y costo drásticamente.

---

## 🛠️ Solución de Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| `TypeError: get_document_metadata() missing...` | Falta de argumentos posicionales. | Pasar: `(modelo, folder_pdf, folder_save, prompt)`. |
| `TypeError: 'dict' object is not subscriptable` | Intentar `chunks[0]` en un diccionario. | Usar `for key, value in chunks.items()` o `list(chunks.values())[0]`. |
| `FileNotFoundError: Google_Branding/` | Las imágenes no se han descargado o la carpeta no existe. | Ejecutar `!gsutil -m rsync -r gs://spls/gsp520 .` primero. |

---

## Parámetro `rrf_ranking_alpha` (solo búsqueda híbrida)

- `1.0` = Solo resultados semánticos (densos)
- `0.0` = Solo resultados por tokens (dispersos)
- `0.5` = Peso equilibrado entre ambos (recomendado para empezar)

## Archivos del Skill

- `scripts/embeddings.py` — Funciones para generar embeddings densos y dispersos
- `scripts/vector_search.py` — Funciones para gestionar índices, endpoints y consultas
- `scripts/hybrid_search.py` — Funciones para búsqueda híbrida con RRF
- `examples/` — Ejemplos completos listos para ejecutar
- `resources/` — Documentación conceptual y guías rápidas
