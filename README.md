# 🔍 Vertex AI Vector Search Skill

Un conjunto de scripts, ejemplos y documentación reutilizable para implementar **búsqueda semántica**, **búsqueda por tokens** y **búsqueda híbrida** con Google Cloud Vertex AI Vector Search.

## ¿Qué es esto?

Este repositorio es un **skill** (conjunto de instrucciones y herramientas) que permite a cualquier desarrollador o agente de IA implementar rápidamente un sistema de búsqueda inteligente basado en:

- **Embeddings densos** (búsqueda semántica): Encuentran contenido por *significado*, no por palabras exactas.
- **Embeddings dispersos** (búsqueda por tokens): Encuentran contenido por *palabras clave exactas* usando TF-IDF.
- **Búsqueda híbrida**: Combina ambos enfoques para máxima calidad de resultados.

## 📁 Estructura del Repositorio

```
├── SKILL.md                  # Instrucciones paso a paso
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias Python
├── scripts/
│   ├── embeddings.py         # Generar embeddings densos y dispersos
│   ├── vector_search.py      # Crear índices, endpoints, consultar
│   └── hybrid_search.py      # Búsqueda híbrida con RRF
├── examples/
│   ├── 01_semantic_search.py  # Ejemplo: búsqueda semántica end-to-end
│   ├── 02_hybrid_search.py    # Ejemplo: búsqueda híbrida end-to-end
│   └── 03_cleanup.py          # Ejemplo: limpieza de recursos
└── resources/
    ├── conceptos.md           # Explicación teórica detallada
    └── guia_rapida.md         # Referencia rápida de código
```

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar credenciales

```bash
gcloud auth login
gcloud config set project TU_PROYECTO_ID
```

### 3. Ejecutar un ejemplo

```bash
python examples/01_semantic_search.py
```

## 📚 Documentación

- **[SKILL.md](SKILL.md)**: Instrucciones detalladas paso a paso
- **[resources/conceptos.md](resources/conceptos.md)**: Explicación teórica completa
- **[resources/guia_rapida.md](resources/guia_rapida.md)**: Referencia rápida de código

## 🧠 Conceptos Clave

| Concepto | Descripción |
|----------|-------------|
| **Embedding** | Representación numérica del significado de un texto (vector de N dimensiones) |
| **Vector Search** | Base de datos optimizada para buscar vectores similares en milisegundos |
| **Index** | La estructura de datos que almacena los embeddings |
| **Endpoint** | El servidor que recibe las consultas |
| **RRF** | Reciprocal Rank Fusion: algoritmo para fusionar rankings de búsqueda semántica y por tokens |

## ⚠️ Importante

- **Costos**: Los Index Endpoints generan costos mientras estén activos. Siempre ejecuta la limpieza.
- **Regiones**: Asegúrate de usar una región que soporte Vector Search.
- **Cuotas**: La API de embeddings tiene límites de velocidad. Los scripts incluyen control de cuota.

## 📄 Licencia

Apache License 2.0 - Basado en los tutoriales oficiales de Google Cloud.
