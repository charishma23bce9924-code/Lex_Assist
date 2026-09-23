import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHUNKS_FILE = Path("data/document_chunks.json")
INDEX_FILE = Path("data/precedents.faiss")
METADATA_FILE = Path("data/precedents_metadata.json")
OUTPUT_FILE = Path("data/document_precedent_results.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



TOP_K = 5


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("=" * 70)
print("LEXASSIST DOCUMENT → PRECEDENT SEMANTIC SEARCH")
print("=" * 70)

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# --------------------------------------------------
# Load document chunks
# --------------------------------------------------

print("\nLoading document chunks...")

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:
    documents = json.load(f)

print(f"Documents available: {len(documents)}")


# --------------------------------------------------
# Process all documents
# --------------------------------------------------

print("\nDocuments selected for precedent search:")

for document in documents:

    print(
        f"- {document['file_name']} "
        f"| pages={document['page_count']} "
        f"| chunks={document['chunk_count']}"
    )

# --------------------------------------------------
# Load precedent FAISS index
# --------------------------------------------------

print("\nLoading precedent FAISS index...")

index = faiss.read_index(
    str(INDEX_FILE)
)

print(f"Precedent vectors: {index.ntotal}")


# --------------------------------------------------
# Load precedent metadata
# --------------------------------------------------

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

print(
    f"Precedent metadata records: {len(metadata)}"
)


# --------------------------------------------------
# Search each document chunk
# --------------------------------------------------

all_results = []

print("\n" + "=" * 70)
print("DOCUMENT → PRECEDENT SEARCH")
print("=" * 70)


for document in documents:

    print(
        f"\nProcessing document: {document['file_name']}"
    )

    document_results = []

    for chunk in document["chunks"]:

        chunk_id = chunk["chunk_id"]
        text = chunk["text"]

        print(
            f"\nChunk {chunk_id}"
            f" | pages {chunk['page_start']}-{chunk['page_end']}"
        )

        # ------------------------------------------
        # Create chunk embedding
        # ------------------------------------------

        embedding = model.encode(
            [text],
            normalize_embeddings=True
        )

        embedding = np.asarray(
            embedding,
            dtype="float32"
        )

        # ------------------------------------------
        # Search precedent index
        # ------------------------------------------

        scores, indices = index.search(
            embedding,
            TOP_K
        )

        chunk_results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            precedent = metadata[idx]

            result = {
                "rank": rank,
                "precedent_id": precedent["id"],
                "case_title": precedent["case_title"],
                "date": precedent["date"],
                "jurisdiction": precedent["jurisdiction"],
                "similarity": float(score)
            }

            chunk_results.append(result)

            print(
                f"  {rank}. "
                f"{precedent['case_title']} "
                f"| similarity={score:.4f}"
            )

        document_results.append({
            "chunk_id": chunk_id,
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "results": chunk_results
        })

    all_results.append({
        "file_name": document["file_name"],
        "page_count": document["page_count"],
        "chunk_count": document["chunk_count"],
        "results": document_results
    })
# --------------------------------------------------
# Save results
# --------------------------------------------------

output = {
    "document_count": len(all_results),
    "top_k": TOP_K,
    "total_chunks_processed": sum(
        document["chunk_count"]
        for document in all_results
    ),
    "documents": all_results
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
# Final verification
# --------------------------------------------------

print("\n" + "=" * 70)
print("DOCUMENT → PRECEDENT SEARCH COMPLETE")
print("=" * 70)

print(
    f"Document: {document['file_name']}"
)

print(
    f"Chunks processed: {len(all_results)}"
)

print(
    f"Precedents searched: {index.ntotal}"
)

print(
    f"Output: {OUTPUT_FILE}"
)