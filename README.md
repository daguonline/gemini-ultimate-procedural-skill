# 🚀 Gemini Ultimate Procedural Skill

Repositorio de nivel experto para implementar flujos de IA con **Gemini** y **Vertex AI Vector Search**: búsqueda semántica, híbrida, multimodal, y RAG sobre documentos complejos.

## ¿Qué incluye?

| Capacidad | Descripción |
|-----------|-------------|
| **Búsqueda Semántica** | Encontrar documentos por significado usando embeddings densos |
| **Búsqueda Híbrida** | Fusión de significado + palabras clave con Reciprocal Rank Fusion |
| **Multimodal RAG** | Extraer y entender tablas, gráficos e imágenes de PDFs complejos |
| **Razonamiento Avanzado** | Uso de `thinking_level` en Gemini 3.1 Pro para lógica profunda |
| **Salida Estructurada** | Respuestas en JSON puro para integración con aplicaciones |

## 📁 Estructura

```
├── SKILL.md              # Guía técnica principal (paso a paso)
├── README.md             # Este archivo
├── requirements.txt      # Dependencias Python
├── scripts/
│   ├── embeddings.py     # Embeddings densos (Gemini) y dispersos (TF-IDF)
│   ├── vector_search.py  # Índices, endpoints, consultas y limpieza
│   └── hybrid_search.py  # Búsqueda híbrida con RRF
├── resources/
│   ├── conceptos.md      # Teoría: embeddings, Vector Search, búsqueda híbrida
│   └── gemini_definitive_guide_es.md  # Guía de modelos Gemini y patrones enterprise
└── examples/
    ├── 01_semantic_search.py   # Ejemplo ejecutable de búsqueda semántica
    ├── 02_hybrid_search.py     # Ejemplo ejecutable de búsqueda híbrida
    ├── 03_cleanup.py           # Limpieza de recursos
    └── *_es.ipynb              # Notebooks traducidos al español (7 labs)
```

## 🚀 Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar credenciales de Google Cloud
gcloud auth login
gcloud config set project TU_PROYECTO_ID
```

Después, sigue la guía paso a paso en **[SKILL.md](SKILL.md)**.

## 📓 Notebooks de Ejemplo

Todos los notebooks están traducidos al español con explicaciones detalladas:

| Notebook | Tema |
|----------|------|
| [intro_multimodal_use_cases_es](examples/intro_multimodal_use_cases_es.ipynb) | Capacidades multimodales de Gemini |
| [multimodal_retail_recommendations_es](examples/multimodal_retail_recommendations_es.ipynb) | Recomendaciones visuales para retail |
| [intro_multimodal_rag_es](examples/intro_multimodal_rag_es.ipynb) | RAG multimodal con PyMuPDF |
| [intro_textemb_vectorsearch_es](examples/intro_textemb_vectorsearch_es.ipynb) | Text embeddings y Vector Search |
| [hybrid_search_es](examples/hybrid_search_es.ipynb) | Búsqueda híbrida semántica + keywords |
| [classify_text_bert_es](examples/classify_text_bert_es.ipynb) | Clasificación de texto con BERT |
| [image_captioning_es](examples/image_captioning_es.ipynb) | Subtitulado automático de imágenes |

## ⚠️ Importante

- **Costos**: Los Index Endpoints de Vector Search se facturan por hora. Siempre ejecuta `03_cleanup.py` al terminar.
- **Cuotas**: Los modelos Gemini en preview pueden tener límites de Rate Limit por tier.
- **Modelos soportados**: Compatible con `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3.1-pro-preview`, y `gemini-embedding-001`.

## 📚 Origen de los Datos (Referencias)

Los datos, documentos y activos usados en los ejemplos de esta Skill provienen de laboratorios oficiales de **Google Cloud Skills Boost** (y de scripts de aprovisionamiento automático como `skill.sh`). 

A continuación se detalla de forma explícita y transparente el origen de cada conjunto de datos:

- **Papers Científicos en PDF** (RAG Multimodal): Extraídos del bucket de Qwiklabs `gs://cloud-training/gsp1210/2507.06261v5.pdf` (y afines).
- **Google Merch Shop Dataset** (Retail y Búsqueda Híbrida): Catálogo de productos oficial de GCSB disponible en `https://storage.googleapis.com/spls/gsp1297/google_merch_shop_items.csv`.
- **Database Stack Overflow** (Búsqueda Semántica): Tabla pública alojada en BigQuery `bigquery-public-data.stackoverflow.posts_questions`.
- **Activos de Imagen y Branding** (Tutoriales visuales): Descargados desde el bucket de GCSB `gs://spls/gsp520`.

## 📄 Licencia

Apache License 2.0
