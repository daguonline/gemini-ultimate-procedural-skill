# Guía de la Gemini Ultimate Procedural Skill

Esta guía consolida los aprendizajes de los reportes técnicos oficiales y las mejores prácticas de implementación para crear aplicaciones de IA de vanguardia.

## 1. Evolución de Modelos: Gemini 2.5 vs Gemini 3

### Gemini 2.5 (Pro y Flash)
- **Ventana de Contexto**: Hasta 1 millón de tokens nativos.
- **Multimodalidad**: Procesamiento fluido de hasta 3 horas de video, audio de larga duración y documentos PDF complejos.
- **Modelo Flash**: Optimizado para tareas de alta frecuencia, priorizando la latencia sin sacrificar capacidades multimodales.

### Gemini 3 y 3.1
- **Gemini 3 Pro**: El modelo más inteligente de Google, con razonamiento avanzado y soporte nativo para agentes.
- **Gemini 3.1 Pro**: Doble rendimiento en razonamiento (`ARC-AGI-2`) comparado con la serie 3.0.
- **Pensamiento Dinámico**: Introducción de niveles de pensamiento controlables para equilibrar calidad y costo.

---

## 2. Parámetros Críticos para la Habilidad Definitiva

### Configuración de Pensamiento (`thinking_config`)
Permite controlar qué tanto "reflexiona" el modelo antes de responder.
- `minimal`: Baja latencia, ideal para chats rápidos.
- `low`: Equilibrio para seguimiento de instrucciones simples.
- `medium`: Estándar para la mayoría de las tareas.
- `high`: Razonamiento profundo (default para Gemini 3.1 en tareas complejas).

```python
# Ejemplo de uso en el SDK
config = GenerateContentConfig(
    thinking_config={"thinking_level": "high"}
)
```

### Resolución de Medios (`media_resolution`)
Crucial para búsquedas multimodales en videos y audios largos.
- Usa resoluciones bajas para escaneo rápido de eventos en videos de 1 hora o más.

---

## 3. Patrones Enterprise y RAG Multimodal

### Context Caching
Para optimizar costos en ventanas de contexto gigantes (1M+ tokens).
- **Caso de uso**: Bases de código completas o bibliotecas de documentos PDF que se consultan repetidamente.

### RAG Multimodal con PyMuPDF
No te limites a extraer texto. Usa un enfoque híbrido:
1. **Extracción**: Usa `PyMuPDF` para separar texto de imágenes/tablas.
2. **Embeddings**: Genera **Multimodal Embeddings** para que el sistema "entienda" tanto el gráfico como el texto que lo describe.
3. **Etiquetado**: En el prompt, etiqueta las opciones (ej: "Imagen 1: Gráfico de ventas") para reducir alucinaciones visuales.

### Salida Estructurada (JSON)
Para integrar Gemini en flujos automatizados de retail o búsqueda:
```python
config = GenerateContentConfig(
    response_mime_type="application/json"
)
```

---

## 4. Mejores Prácticas de Búsqueda
- **Ranking RRF**: Combina búsqueda semántica (densos) y por tokens (dispersos) con un alpha de `0.5`.
- **Reasoning**: Para búsquedas que requieren lógica (ej: encontrar discrepancias en contratos), usa **Gemini 3.1 Pro** con `thinking_level="high"`.
