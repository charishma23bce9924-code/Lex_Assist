import json
import faiss
from sentence_transformers import SentenceTransformer


# Load model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
index = faiss.read_index(
    "data/statutes.faiss"
)

# Load metadata
with open(
    "data/statutes_metadata.json",
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)


# --------------------------------------------------
# Test question
# --------------------------------------------------

query = """
What provision allows a Claims Tribunal to award
interest on compensation in a motor vehicle accident claim?
"""


# Create query embedding
query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

# Search top 5
scores, indices = index.search(
    query_embedding,
    5
)


# Display results
print("\n" + "=" * 70)
print("LEXASSIST STATUTE SEARCH")
print("=" * 70)

print("\nQuery:")
print(query)

print("\nTop 5 results:")

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    result = metadata[idx]

    print(f"\n{rank}. {result['provision_name']}")
    print(f"   ID: {result['id']}")
    print(f"   Similarity: {score:.4f}")