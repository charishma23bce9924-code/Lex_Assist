import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHUNKS_FILE = Path("data/document_chunks.json")
INDEX_FILE = Path("data/statutes.faiss")
METADATA_FILE = Path("data/statutes_metadata.json")
OUTPUT_FILE = Path("data/document_statute_results.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
BATCH_SIZE = 16


# --------------------------------------------------
# Header
# --------------------------------------------------

print("=" * 70)
print("LEXASSIST DOCUMENT → STATUTE SEMANTIC SEARCH")
print("=" * 70)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# --------------------------------------------------
# Load document chunks
# --------------------------------------------------

print("\nLoading document chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"Documents available: {len(documents)}")


# --------------------------------------------------
# Load statute FAISS index
# --------------------------------------------------

print("\nLoading statute FAISS index...")

index = faiss.read_index(str(INDEX_FILE))

print(f"Statute vectors: {index.ntotal}")


# --------------------------------------------------
# Load statute metadata
# --------------------------------------------------

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Statute metadata records: {len(metadata)}")


# --------------------------------------------------
# Validate index and metadata
# --------------------------------------------------

if index.ntotal != len(metadata):
    raise ValueError(
        f"FAISS index contains {index.ntotal} vectors "
        f"but metadata contains {len(metadata)} records."
    )


# --------------------------------------------------
# Process all documents
# --------------------------------------------------

all_documents_results = []

total_chunks = sum(
    document["chunk_count"]
    for document in documents
)

print("\n" + "=" * 70)
print("DOCUMENT → STATUTE SEARCH")
print("=" * 70)

print(f"\nTotal documents: {len(documents)}")
print(f"Total chunks: {total_chunks}")
print(f"Top-K: {TOP_K}")
print(f"Batch size: {BATCH_SIZE}")


for document_number, document in enumerate(documents, start=1):

    file_name = document["file_name"]
    chunks = document["chunks"]

    print("\n" + "-" * 70)
    print(
        f"Document {document_number}/{len(documents)}"
    )
    print(file_name)
    print(
        f"Pages: {document['page_count']} | "
        f"Chunks: {document['chunk_count']}"
    )
    print("-" * 70)

    # --------------------------------------------------
    # Extract chunk texts
    # --------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------
    # Generate embeddings in batches
    # --------------------------------------------------

    print("\nGenerating document chunk embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------
    # Search statute FAISS index
    # --------------------------------------------------

    print("Searching statute index...")

    scores, indices = index.search(
        embeddings,
        TOP_K
    )

    # --------------------------------------------------
    # Build chunk results
    # --------------------------------------------------

    document_results = []

    for chunk_position, chunk in enumerate(chunks):

        chunk_results = []

        for rank in range(TOP_K):

            idx = int(indices[chunk_position][rank])
            score = float(scores[chunk_position][rank])

            statute = metadata[idx]

            chunk_results.append({
                "rank": rank + 1,
                "statute_id": statute["id"],
                "provision_name": statute["provision_name"],
                "similarity": score
            })

        document_results.append({
            "chunk_id": chunk["chunk_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "results": chunk_results
        })

    # --------------------------------------------------
    # Add document result
    # --------------------------------------------------

    all_documents_results.append({
        "file_name": file_name,
        "page_count": document["page_count"],
        "chunk_count": document["chunk_count"],
        "top_k": TOP_K,
        "results": document_results
    })

    print(
        f"Completed: {len(document_results)} chunks"
    )


# --------------------------------------------------
# Save results
# --------------------------------------------------

output = {
    "model": MODEL_NAME,
    "embedding_dimension": int(embeddings.shape[1]),
    "statute_vectors": index.ntotal,
    "top_k": TOP_K,
    "batch_size": BATCH_SIZE,
    "document_count": len(all_documents_results),
    "total_chunks": total_chunks,
    "documents": all_documents_results
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


# --------------------------------------------------
# Final
# --------------------------------------------------

print("\n" + "=" * 70)
print("DOCUMENT → STATUTE SEARCH COMPLETE")
print("=" * 70)

print(f"Documents processed: {len(documents)}")
print(f"Total chunks processed: {total_chunks}")
print(f"Statutes searched: {index.ntotal}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Top-K per chunk: {TOP_K}")
print(f"Output: {OUTPUT_FILE}")