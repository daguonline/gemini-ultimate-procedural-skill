# Conceptos: Text Embeddings y Vector Search

## ¿Qué es un Embedding?

Un **embedding** es una representación numérica del **significado** de un texto. Las computadoras no entienden palabras, entienden números. Un modelo de IA convierte texto en una lista de miles de números (un "vector") donde textos con significados similares producen vectores similares.

- "perro" y "cachorro" → vectores **cercanos** en el espacio
- "perro" y "avión" → vectores **lejanos** en el espacio

## Tipos de Embeddings

### Embeddings Densos (Dense)
- Generados por modelos de IA como Gemini o text-embedding-005
- Capturan el **significado semántico** del texto
- Tienen dimensiones fijas (ej: 768 o 3072) y todos los valores son distintos de cero
- Ideal para: "¿Cómo leer archivos?" encuentra docs sobre "parsing JSON"

### Embeddings Dispersos (Sparse)
- Generados por algoritmos como TF-IDF, BM25 o SPLADE
- Capturan la **frecuencia de palabras clave**
- Tienen miles de dimensiones pero la mayoría son ceros
- Ideal para: Buscar un SKU específico, un nombre de producto o un código

## ¿Qué es Vector Search?

**Vertex AI Vector Search** (antes Matching Engine) es una base de datos especializada en buscar vectores similares de forma ultra rápida. Utiliza algoritmos de **ANN (Approximate Nearest Neighbor)** para encontrar resultados en milisegundos, incluso con miles de millones de registros.

### Componentes

| Componente | Analogía | Función |
|------------|----------|---------|
| **Index** | La biblioteca | Almacena todos los embeddings organizados |
| **Endpoint** | La ventanilla | Recibe las consultas y devuelve resultados |
| **Deploy** | Abrir la biblioteca | Conecta el índice al endpoint para servir |

## Búsqueda Híbrida

La **búsqueda híbrida** combina ambos tipos de embeddings en un solo índice. Usa **Reciprocal Rank Fusion (RRF)** para fusionar los rankings de ambas búsquedas.

### ¿Cuándo usar cada tipo?

| Situación | Tipo de búsqueda |
|-----------|-----------------|
| Buscar por concepto o idea | Semántica (densa) |
| Buscar por código, SKU o nombre exacto | Por tokens (dispersa) |
| Buscar productos en e-commerce | Híbrida |
| Implementar RAG para chatbots | Semántica o híbrida |

### Parámetro Alpha (RRF)

El parámetro `rrf_ranking_alpha` controla el peso entre ambos tipos:
- `1.0` = Solo semántica
- `0.0` = Solo por tokens
- `0.5` = Equilibrado (recomendado para empezar)

## Flujo de Datos

```
Datos originales (CSV, BigQuery, etc.)
    ↓
DataFrame de Pandas
    ↓
Embeddings (densos y/o dispersos)
    ↓
Archivo JSONL
    ↓
Cloud Storage Bucket
    ↓
Vector Search Index (la "BD" vectorial)
    ↓
Index Endpoint (el "servidor")
    ↓
Consultas en milisegundos
```

## Aplicaciones en el Mundo Real

- **RAG (Retrieval-Augmented Generation)**: Darle a un chatbot acceso a tus documentos internos
- **Búsqueda de productos**: E-commerce inteligente con búsqueda por concepto
- **Recomendaciones**: "Si te gustó X, te gustará Y"
- **Clasificación de texto**: Agrupar documentos similares automáticamente
- **Detección de duplicados**: Encontrar contenido repetido por significado
