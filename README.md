# 🚀 Gemini Ultimate Procedural Skill

La **Gemini Ultimate Procedural Skill** es un repositorio de nivel experto diseñado para implementar flujos de IA de vanguardia usando **Gemini 3.1** y **Vertex AI Vector Search**. 

Este proyecto ha evolucionado de una simple herramienta de búsqueda vectorial a una solución completa para **Multimodal RAG**, **Enterprise Search** y **Razonamiento Avanzado**.

## ¿Qué es esto?

Este repositorio es un **skill** definitivo (guía procedimental y herramientas) que cubre:

- **Búsqueda Multimodal**: Indexación y búsqueda en texto, imágenes y tablas simultáneamente.
- **Multimodal RAG**: Generación aumentada por recuperación que "entiende" gráficos y fotos en documentos PDF complejos.
- **Razonamiento Avanzado**: Uso de `thinking_level` de Gemini 3.1 Pro para tareas que requieren lógica profunda.
- **Patrones Enterprise**: Implementación de Context Caching, Batch Prediction y Salida Estructurada (JSON).
- **Búsqueda Híbrida**: Fusión de significado semántico y palabras clave (RRF).

## 📁 Estructura del Repositorio

```
├── SKILL.md                          # Guía técnica principal del Skill
├── README.md                         # Este archivo
├── requirements.txt                  # Dependencias Python
├── resources/
│   ├── gemini_definitive_guide_es.md  # Guía maestra de Gemini 2.5/3 [NUEVO]
│   ├── conceptos.md                   # Teoría de búsqueda vectorial
│   └── guia_rapida.md                 # Referencia de código
### Ejemplos y Guías (Directorio `examples/`)

Contenido maestro traducido y adaptado:
- [**Multimodal RAG**](examples/intro_multimodal_rag_es.ipynb): RAG avanzado con PyMuPDF.
- [**Rich Documents Challenge**](examples/inspect_rich_documents_es.ipynb): Solución al desafío de documentos complejos con tablas y gráficos.
- [**Recomendaciones Retail**](examples/multimodal_retail_recommendations_es.ipynb): Recomendaciones visuales.
- [**Casos de Uso Multimodales**](examples/intro_multimodal_use_cases_es.ipynb): Capacidades multimodales.
- [**Text Embeddings & Vector Search**](examples/intro_textemb_vectorsearch_es.ipynb): Búsqueda semántica.
- [**Búsqueda Híbrida**](examples/hybrid_search_es.ipynb): Combinando semántica y keywords.
- [**Clasificación con BERT**](examples/classify_text_bert_es.ipynb): Análisis de sentimientos.
- [**Subtitulado de Imágenes**](examples/image_captioning_es.ipynb): Generación de texto con atención.

---

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
pip install pymupdf  # Requerido para RAG Multimodal
```

### 2. Configurar credenciales

```bash
gcloud auth login
gcloud config set project TU_PROYECTO_ID
```

### 3. Explorar la Habilidad Definitiva
Se recomienda comenzar por la [Guía Maestra de Gemini](resources/gemini_definitive_guide_es.md) para entender cómo configurar los niveles de razonamiento y el caching.

## 📚 Documentación Principal

- **[SKILL.md](SKILL.md)**: El flujo de trabajo completo del Skill.
- **[resources/gemini_definitive_guide_es.md](resources/gemini_definitive_guide_es.md)**: Todo sobre Gemini 3.1 y patrones Enterprise.

## 🧠 Conceptos Avanzados

| Concepto | Aplicación |
|----------|-------------|
| **Thinking Level** | Controla el razonamiento profundo en Gemini 3.1 |
| **Multimodal RAG** | Extrae y "lee" gráficos de tus PDF automáticamente |
| **Context Caching** | Reduce costos de tokens en corpus masivos |
| **Structured Output** | Asegura que la IA responda en JSON puro para apps |

## ⚠️ Importante

- **Cuotas**: Los modelos Gemini 3 tienen límites de Rate Limit específicos por tier.
- **Limpieza**: Los Index Endpoints de Vertex AI son facturables por hora. Usa `03_cleanup.py` al terminar.

## 📄 Licencia

Apache License 2.0 - Basado en los reportes técnicos de Gemini y tutoriales de Google Cloud.
