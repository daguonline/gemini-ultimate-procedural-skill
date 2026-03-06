"""
vector_search.py - Funciones para gestionar índices, endpoints y consultas en Vector Search.

Este módulo encapsula las operaciones principales de Vertex AI Vector Search:
crear índices, endpoints, desplegar índices, ejecutar consultas y limpiar recursos.
"""

import json
from typing import List, Optional

from google.cloud import aiplatform


# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================


def save_embeddings_to_jsonl(
    df,
    output_path: str = "items.json",
    id_col: str = "id",
    embedding_col: str = "embedding",
    sparse_col: Optional[str] = None,
):
    """
    Guarda un DataFrame con embeddings como archivo JSONL compatible con Vector Search.

    Args:
        df: DataFrame de pandas con las columnas de id y embeddings.
        output_path: Ruta del archivo JSONL de salida.
        id_col: Nombre de la columna con el identificador único.
        embedding_col: Nombre de la columna con el embedding denso.
        sparse_col: Nombre de la columna con el embedding disperso (opcional, para híbrido).
    """
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            item = {"id": str(row[id_col]), "embedding": row[embedding_col]}
            if sparse_col and sparse_col in df.columns:
                item["sparse_embedding"] = row[sparse_col]
            f.write(json.dumps(item) + "\n")
    print(f"Guardado {len(df)} registros en {output_path}")


def upload_to_gcs(local_path: str, bucket_uri: str):
    """
    Sube un archivo local a un bucket de Cloud Storage usando gsutil.

    Args:
        local_path: Ruta al archivo local.
        bucket_uri: URI del bucket (ej: gs://mi-proyecto-bucket).
    """
    import subprocess

    result = subprocess.run(
        ["gsutil", "cp", local_path, bucket_uri],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Archivo subido a {bucket_uri}")
    else:
        raise RuntimeError(f"Error subiendo archivo: {result.stderr}")


def save_embeddings_to_gcs(
    df,
    bucket_uri: str,
    mode: str = "dense",
    id_col: str = "id",
    embedding_col: str = "embedding",
    sparse_col: str = "sparse_embedding",
):
    """
    Guarda embeddings como JSONL y los sube a Cloud Storage en un solo paso.

    Args:
        df: DataFrame con los datos.
        bucket_uri: URI del bucket de GCS.
        mode: "dense" (solo semántico) o "hybrid" (denso + disperso).
        id_col: Nombre de la columna ID.
        embedding_col: Nombre de la columna con embedding denso.
        sparse_col: Nombre de la columna con embedding disperso.
    """
    sparse = sparse_col if mode == "hybrid" else None
    save_embeddings_to_jsonl(df, "items.json", id_col, embedding_col, sparse)
    upload_to_gcs("items.json", bucket_uri)


# =============================================================================
# GESTIÓN DE ÍNDICES Y ENDPOINTS
# =============================================================================


def create_index(
    display_name: str,
    bucket_uri: str,
    dimensions: int = 768,
    approximate_neighbors_count: int = 20,
    distance_measure_type: str = "DOT_PRODUCT_DISTANCE",
) -> aiplatform.MatchingEngineIndex:
    """
    Crea un índice de Vector Search con los embeddings almacenados en GCS.

    Args:
        display_name: Nombre visible del índice.
        bucket_uri: URI del bucket con los embeddings JSONL.
        dimensions: Dimensiones del embedding (768 para text-embedding-005, 3072 para gemini-embedding-001).
        approximate_neighbors_count: Cuántos vecinos cercanos recuperar.
        distance_measure_type: Métrica de distancia (DOT_PRODUCT_DISTANCE, COSINE_DISTANCE, etc.).

    Returns:
        Instancia de MatchingEngineIndex creada.
    """
    print(f"Creando índice '{display_name}'... (esto puede tomar varios minutos)")
    my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        contents_delta_uri=bucket_uri,
        dimensions=dimensions,
        approximate_neighbors_count=approximate_neighbors_count,
        distance_measure_type=distance_measure_type,
    )
    print(f"Índice creado: {my_index.resource_name}")
    return my_index


def create_endpoint(display_name: str) -> aiplatform.MatchingEngineIndexEndpoint:
    """
    Crea un Index Endpoint público para recibir consultas.

    Args:
        display_name: Nombre visible del endpoint.

    Returns:
        Instancia de MatchingEngineIndexEndpoint creada.
    """
    print(f"Creando endpoint '{display_name}'...")
    my_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=display_name,
        public_endpoint_enabled=True,
    )
    print(f"Endpoint creado: {my_endpoint.resource_name}")
    return my_endpoint


def deploy_index(
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
    index: aiplatform.MatchingEngineIndex,
    deployed_index_id: str,
    machine_type: str = "e2-standard-16",
    min_replicas: int = 1,
    max_replicas: int = 1,
):
    """
    Despliega un índice en un endpoint para que pueda recibir consultas.

    Args:
        endpoint: El endpoint donde desplegar.
        index: El índice a desplegar.
        deployed_index_id: ID único para el despliegue.
        machine_type: Tipo de máquina (e2-standard-16 por defecto).
        min_replicas: Réplicas mínimas.
        max_replicas: Réplicas máximas.
    """
    print(f"Desplegando índice... (primera vez puede tardar ~25 minutos)")
    endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_index_id,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
    )
    print("Despliegue completado.")


# =============================================================================
# CONSULTAS
# =============================================================================


def semantic_query(
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
    deployed_index_id: str,
    query_embedding: List[float],
    num_neighbors: int = 10,
):
    """
    Ejecuta una consulta semántica en el índice desplegado.

    Args:
        endpoint: El endpoint con el índice desplegado.
        deployed_index_id: ID del índice desplegado.
        query_embedding: El embedding de la consulta.
        num_neighbors: Número de resultados a devolver.

    Returns:
        Lista de vecinos más cercanos con sus distancias.
    """
    response = endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_embedding],
        num_neighbors=num_neighbors,
    )
    return response[0]


# =============================================================================
# LIMPIEZA
# =============================================================================


def cleanup(
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
    index: aiplatform.MatchingEngineIndex,
    bucket_uri: Optional[str] = None,
):
    """
    Elimina el endpoint, el índice y opcionalmente el bucket de GCS.

    ⚠️ IMPORTANTE: Siempre ejecutar después de terminar de usar los recursos
    para evitar costos inesperados.

    Args:
        endpoint: El endpoint a eliminar.
        index: El índice a eliminar.
        bucket_uri: URI del bucket a eliminar (opcional).
    """
    import subprocess

    print("Limpiando recursos...")

    # Eliminar despliegues y endpoint
    endpoint.undeploy_all()
    endpoint.delete(force=True)
    print("  ✓ Endpoint eliminado")

    # Eliminar índice
    index.delete()
    print("  ✓ Índice eliminado")

    # Eliminar bucket
    if bucket_uri:
        subprocess.run(["gsutil", "rm", "-r", bucket_uri], capture_output=True)
        print(f"  ✓ Bucket {bucket_uri} eliminado")

    print("Limpieza completada.")
