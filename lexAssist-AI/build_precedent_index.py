import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_FILE = Path("data/precedents_clean_repaired.json")
INDEX_FILE = Path("data/precedents.faiss")
METADATA_FILE = Path("data/precedents_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Load precedents
# --------------------------------------------------

print("=" * 70)
print("LEXASSIST PRECEDENT INDEX BUILDER")
print("=" * 70)

print("\nLoading precedents...")

with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as f:
    precedents = json.load(f)

print(f"Loaded {len(precedents)} precedents")


# --------------------------------------------------
# Prepare searchable text
# --------------------------------------------------

texts = []

for precedent in precedents:

    case_title = precedent.get("case_title", "")
    date = precedent.get("date", "")
    jurisdiction = precedent.get("jurisdiction", "")
    text = precedent.get("text", "")

    searchable_text = (
        f"Case: {case_title}\n"
        f"Date: {date}\n"
        f"Jurisdiction: {jurisdiction}\n"
        f"{text}"
    )

    texts.append(searchable_text)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded.")


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

print("\nGenerating precedent embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print(
    f"Embedding shape: {embeddings.shape}"
)


# --------------------------------------------------
# Build FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print("\nFAISS index created.")

print(
    f"Vectors in index: {index.ntotal}"
)

print(
    f"Embedding dimension: {dimension}"
)


# --------------------------------------------------
# Save index
# --------------------------------------------------

faiss.write_index(
    index,
    str(INDEX_FILE)
)

print(
    f"\nSaved FAISS index -> {INDEX_FILE}"
)


# --------------------------------------------------
# Save metadata
# --------------------------------------------------

metadata = []

for precedent in precedents:

    metadata.append({
        "id": precedent.get("id"),
        "case_title": precedent.get("case_title"),
        "date": precedent.get("date"),
        "jurisdiction": precedent.get("jurisdiction")
    })


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"Saved metadata -> {METADATA_FILE}"
)


# --------------------------------------------------
# Final verification
# --------------------------------------------------

print("\n" + "=" * 70)
print("PRECEDENT INDEX COMPLETE")
print("=" * 70)

print(
    f"Precedents: {len(precedents)}"
)

print(
    f"Embedding dimension: {dimension}"
)

print(
    f"FAISS vectors: {index.ntotal}"
)

print(
    f"Metadata records: {len(metadata)}"
)