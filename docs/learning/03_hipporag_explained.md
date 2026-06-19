# HippoRAG Explained

## What is HippoRAG?

HippoRAG is a retrieval method inspired by **hippocampal indexing theory** in neuroscience. It builds a Knowledge Graph from document passages and uses **Personalized PageRank** for multi-hop traversal.

## How It Works

### 1. Triple Extraction
Extract subject-predicate-object triples from passages:
```
"PolyMind uses Qdrant for vector storage"
→ (PolyMind, uses, Qdrant)
→ (Qdrant, is, vector database)
```

### 2. Knowledge Graph Construction
Build a NetworkX DiGraph from triples:
- Nodes = entities (subjects, objects)
- Edges = relationships (predicates)
- Each node tracks which passages it appears in

### 3. Synonymy Linking
Connect similar entities using embedding similarity:
```
cosine_sim("Qdrant", "vector DB") = 0.87 → link
cosine_sim("PolyMind", "assistant") = 0.45 → no link
```

### 4. Query Processing
```
Query: "What database does PolyMind use?"
→ Extract entities: ["PolyMind", "database"]
→ Match to graph nodes: ["PolyMind", "Qdrant"]
→ Personalized PageRank from these nodes
→ Rank passages by PPR score
→ Return top-k passages
```

## Why It's Better

| Method | Multi-hop Accuracy | Speed |
|--------|-------------------|-------|
| Standard Vector RAG | 79% | Fast |
| IRCoT | 89% | Slow (10-30x) |
| **HippoRAG v2** | **86%** | **Fast** |

## PolyMind Implementation

```python
class HippoRAGRetriever(IRetriever):
    def index(self, chunks):
        # 1. Extract triples from each chunk
        # 2. Build Knowledge Graph
        # 3. Add synonymy edges

    def retrieve(self, query, top_k=5):
        # 1. Extract query entities
        # 2. Match to graph nodes
        # 3. Run Personalized PageRank
        # 4. Aggregate scores at passage level
        # 5. Return top-k passages
```

## When to Use HippoRAG

- **Multi-hop questions** — "Who founded the company that made X?"
- **Relationship-heavy queries** — "What's the connection between A and B?"
- **Complex reasoning** — Questions requiring 2+ facts from different sources

## When NOT to Use

- **Simple factual queries** — "What is Python?" (use standard RAG)
- **No relationships** — Documents with isolated facts
- **Very large corpora** — Graph construction becomes expensive
