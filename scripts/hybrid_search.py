"""
hybrid_search.py - Funciones para búsqueda híbrida (semántica + tokens) con RRF.

La búsqueda híbrida combina embeddings densos (significado) con embeddings
dispersos (palabras clave), fusionando los resultados con Reciprocal Rank Fusion (RRF).
"""

from typing import List

from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    HybridQuery,
)


def create_hybrid_query(
    dense_embedding: List[float],
    sparse_embedding: dict,
    alpha: float = 0.5,
) -> HybridQuery:
    """
    Crea un objeto HybridQuery para consultas que combinan búsqueda semántica y por tokens.

    El parámetro alpha controla el peso entre ambos tipos de búsqueda:
    - alpha = 1.0  → Solo búsqueda semántica (densa)
    - alpha = 0.0  → Solo búsqueda por tokens (dispersa)
    - alpha = 0.5  → Peso equilibrado entre ambas (recomendado)

    Args:
        dense_embedding: Embedding denso (semántico) de la consulta.
        sparse_embedding: Embedding disperso con keys 'values' y 'dimensions'.
        alpha: Peso para RRF ranking (0.0 a 1.0).

    Returns:
        Objeto HybridQuery listo para usar con find_neighbors.
    """
    return HybridQuery(
        dense_embedding=dense_embedding,
        sparse_embedding_dimensions=sparse_embedding["dimensions"],
        sparse_embedding_values=sparse_embedding["values"],
        rrf_ranking_alpha=alpha,
    )


def hybrid_query(
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
    deployed_index_id: str,
    dense_embedding: List[float],
    sparse_embedding: dict,
    alpha: float = 0.5,
    num_neighbors: int = 10,
):
    """
    Ejecuta una consulta híbrida que combina búsqueda semántica y por tokens.

    Args:
        endpoint: El endpoint con el índice híbrido desplegado.
        deployed_index_id: ID del índice desplegado.
        dense_embedding: Embedding denso de la consulta.
        sparse_embedding: Embedding disperso de la consulta.
        alpha: Peso RRF (0.0 = solo tokens, 1.0 = solo semántica, 0.5 = equilibrado).
        num_neighbors: Número de resultados a devolver.

    Returns:
        Lista de vecinos más cercanos con distancias densas y dispersas.
    """
    query = create_hybrid_query(dense_embedding, sparse_embedding, alpha)

    response = endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query],
        num_neighbors=num_neighbors,
    )
    return response[0] if response else []


def print_hybrid_results(results, df, title_col: str = "title", id_col: str = None):
    """
    Imprime los resultados de una consulta híbrida de forma legible.

    Args:
        results: Lista de vecinos devueltos por hybrid_query.
        df: DataFrame original con los datos.
        title_col: Nombre de la columna con el título/texto para mostrar.
        id_col: (Opcional) Nombre de la columna ID, para búsquedas más robustas.
    """
    print(f"{'Resultado':<45} {'Dist. Densa':>12} {'Dist. Dispersa':>15}")
    print("-" * 75)
    for neighbor in results:
        if id_col and id_col in df.columns:
            row_df = df[df[id_col].astype(str) == str(neighbor.id)]
            title = row_df[title_col].iloc[0] if not row_df.empty else f"ID: {neighbor.id}"
        else:
            try:
                title = df[title_col].iloc[int(neighbor.id)]
            except (ValueError, IndexError):
                title = f"ID: {neighbor.id}"
                
        dense_dist = neighbor.distance if neighbor.distance else 0.0
        sparse_dist = neighbor.sparse_distance if neighbor.sparse_distance else 0.0
        print(f"{title:<45} {dense_dist:>12.4f} {sparse_dist:>15.4f}")
