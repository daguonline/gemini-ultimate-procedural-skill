"""
embeddings.py - Funciones reutilizables para generar embeddings densos y dispersos.

Embeddings Densos: Capturan el SIGNIFICADO semántico del texto (Gemini / Vertex AI).
Embeddings Dispersos: Capturan la FRECUENCIA de palabras clave (TF-IDF).
"""

import time
from typing import List, Optional

from tqdm import tqdm


# =============================================================================
# EMBEDDINGS DENSOS (Semánticos) - Usando Gemini / Vertex AI
# =============================================================================

EMBEDDING_MODEL = "gemini-embedding-001"


def get_dense_embedding(client, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Genera un embedding denso para un solo texto usando la API de Vertex AI.

    Args:
        client: Cliente de google.genai inicializado con vertexai=True.
        text: El texto a convertir en embedding.
        model: Modelo de embedding a usar (default: gemini-embedding-001).

    Returns:
        Lista de floats representando el embedding (ej: 3072 dimensiones).
    """
    response = client.models.embed_content(model=model, contents=[text])
    return response.embeddings[0].values


def get_dense_embeddings_batch(
    client,
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 5,
    delay: float = 1.0,
) -> List[List[float]]:
    """
    Genera embeddings densos en lotes con control de cuota.

    La API de Vertex AI tiene límites de velocidad. Esta función procesa
    los textos en lotes pequeños con una pausa entre cada lote para evitar
    errores de cuota (429 Too Many Requests).

    Args:
        client: Cliente de google.genai inicializado.
        texts: Lista de textos a convertir.
        model: Modelo de embedding.
        batch_size: Número de textos por lote (máx 5 para la API).
        delay: Segundos de espera entre lotes.

    Returns:
        Lista de embeddings (una lista de floats por cada texto).
    """
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generando embeddings"):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(model=model, contents=batch)
        batch_embeddings = [e.values for e in response.embeddings]
        all_embeddings.extend(batch_embeddings)
        time.sleep(delay)
    return all_embeddings


# =============================================================================
# EMBEDDINGS DISPERSOS (Por Tokens) - Usando TF-IDF
# =============================================================================


def train_sparse_vectorizer(corpus: List[str]):
    """
    Entrena un vectorizador TF-IDF con el corpus proporcionado.

    El vectorizador aprende qué palabras son "importantes" (poco comunes)
    y cuáles son "triviales" (muy comunes como "the", "a", "of").

    Args:
        corpus: Lista de textos con los que entrenar el vectorizador.

    Returns:
        Instancia de TfidfVectorizer entrenada.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)
    return vectorizer


def get_sparse_embedding(vectorizer, text: str) -> dict:
    """
    Genera un embedding disperso para un texto usando el vectorizador TF-IDF.

    Los embeddings dispersos tienen miles de dimensiones pero solo unas pocas
    con valores distintos de cero. El formato de salida es compatible con
    Vertex AI Vector Search.

    Args:
        vectorizer: Instancia de TfidfVectorizer ya entrenada.
        text: El texto a convertir.

    Returns:
        Diccionario con 'values' (los valores TF-IDF) y 'dimensions' (las posiciones).
        Ejemplo: {"values": [0.93, 0.36], "dimensions": [191, 78]}
    """
    tfidf_vector = vectorizer.transform([text])
    values = []
    dims = []
    for i, tfidf_value in enumerate(tfidf_vector.data):
        values.append(float(tfidf_value))
        dims.append(int(tfidf_vector.indices[i]))
    return {"values": values, "dimensions": dims}
