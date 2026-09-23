import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_FILE = Path("data/statutes_clean.json")
INDEX_FILE = Path("data/statutes.faiss")
METADATA_FILE = Path("data/statutes_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Load statutes
# --------------------------------------------------

print("Loading statutes...")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    statutes = json.load(f)

print(f"Loaded {len(statutes)} statutes")


# --------------------------------------------------
# Prepare text
# --------------------------------------------------

texts = []

for statute in statutes:
    text = statute["text"]

    # Include provision name because it carries
    # useful legal context.
    searchable_text = (
        statute["provision_name"]
        + "\n"
        + text
    )

    texts.append(searchable_text)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("\nLoading embedding model:")
print(MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

print("\nGenerating embeddings...")

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

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# Build FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("\nFAISS index created.")
print("Vectors in index:", index.ntotal)


# --------------------------------------------------
# Save FAISS index
# --------------------------------------------------

faiss.write_index(
    index,
    str(INDEX_FILE)
)

print(f"Saved FAISS index -> {INDEX_FILE}")


# --------------------------------------------------
# Save metadata
# --------------------------------------------------

metadata = []

for statute in statutes:
    metadata.append({
        "id": statute["id"],
        "provision_name": statute["provision_name"]
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

print(f"Saved metadata -> {METADATA_FILE}")


# --------------------------------------------------
# Final verification
# --------------------------------------------------

print("\n" + "=" * 60)
print("STATUTE INDEX COMPLETE")
print("=" * 60)

print("Statutes:", len(statutes))
print("Embedding dimension:", dimension)
print("FAISS vectors:", index.ntotal)