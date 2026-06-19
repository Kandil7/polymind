# Qdrant Patterns

## What is Qdrant?

Qdrant is a purpose-built **vector database** for similarity search. It supports:
- Dense vector search (embeddings)
- Sparse vector search (BM25-like)
- Hybrid search (dense + sparse)
- Payload filtering (metadata)
- Sharding and replication

## PolyMind Usage

### Collection Setup
```python
from qdrant_client.models import VectorParams, Distance

client.create_collection(
    collection_name="polymind",
    vectors_config=VectorParams(
        size=1024,  # bge-m3 dimension
        distance=Distance.COSINE,
    ),
)
```

### Hybrid Search
```python
# Dense vector search
results = client.search(
    collection_name="polymind",
    query_vector=query_embedding,
    query_filter=Filter(must=[
        FieldCondition(key="modality", match=MatchValue(value="text"))
    ]),
    limit=5,
)
```

### Payload Filtering
```python
# Filter by metadata
filter = Filter(must=[
    FieldCondition(key="source", match=MatchValue(value="doc.pdf")),
    FieldCondition(key="page", range={"gte": 1, "lte": 10}),
])
```

## Best Practices

1. **Version collections** — Use `polymind_v1`, `polymind_v2`
2. **Index metadata** — Filter on indexed fields for speed
3. **Use payload** — Store source, modality, page in payload
4. **Batch upserts** — Insert multiple points at once
5. **Monitor metrics** — Track collection size and query latency

## Common Patterns

### Upsert with Metadata
```python
client.upsert(
    collection_name="polymind",
    points=[PointStruct(
        id=str(chunk.id),
        vector=embedding,
        payload={
            "text": chunk.text,
            "source": chunk.metadata.source,
            "modality": chunk.metadata.modality,
        },
    )]
)
```

### Search with Filter
```python
results = client.search(
    collection_name="polymind",
    query_vector=query_vec,
    query_filter=Filter(must=[
        FieldCondition(key="modality", match=MatchValue(value="audio"))
    ]),
    limit=5,
)
```
