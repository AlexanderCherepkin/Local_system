"""Проверка пофлаговой ACL-фильтрации в ChromaDB."""

import chromadb

c = chromadb.EphemeralClient()
col = c.create_collection("flags", metadata={"hnsw:space": "cosine"})
col.upsert(
    ids=["1", "2"],
    documents=["hello", "world"],
    embeddings=[[1.0, 0.0], [0.0, 1.0]],
    metadatas=[
        {"acl_hr": True, "acl_legal": True},
        {"acl_engineering": True},
    ],
)

tests = [
    ("single role direct", {"acl_hr": {"$eq": True}}),
    ("or two roles", {"$or": [{"acl_hr": {"$eq": True}}, {"acl_engineering": {"$eq": True}}]}),
    ("deny-all (missing key)", {"acl_ghost": {"$eq": True}}),
]
for name, where in tests:
    r = col.query(query_embeddings=[[1.0, 0.0]], where=where, include=["documents"])
    print(f"{name}: ids={r['ids']}")

