import json
import faiss
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_FILE = "data/statutes.faiss"
METADATA_FILE = "data/statutes_metadata.json"

# IMPORTANT:
# This is the role-aware query file created in
# build_role_aware_queries.py
QUERY_FILE = "data/queries_test_role_aware.json"

TOP_K = 10


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# Load FAISS statute index
# --------------------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)


# --------------------------------------------------
# Load statute metadata
# --------------------------------------------------

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)


# --------------------------------------------------
# Load role-aware test queries
# --------------------------------------------------

with open(
    QUERY_FILE,
    "r",
    encoding="utf-8"
) as f:
    queries = json.load(f)

print(
    f"Role-aware test queries loaded: {len(queries)}"
)


# --------------------------------------------------
# Evaluation variables
# --------------------------------------------------

recall_at_1 = 0
recall_at_5 = 0
recall_at_10 = 0

mrr_total = 0

evaluated_queries = 0
skipped_queries = 0


# --------------------------------------------------
# Evaluate every query
# --------------------------------------------------

print("\nRunning role-aware retrieval evaluation...\n")

for i, query in enumerate(queries):

    # Ground-truth relevant statutes
    relevant_ids = query.get(
        "relevant_statute_ids",
        []
    )

    # Skip if no ground-truth statutes exist
    if not relevant_ids:
        skipped_queries += 1
        continue

    relevant_ids = {
        str(x)
        for x in relevant_ids
    }

    # Role-aware query representation
    query_text = query["text"]

    # Generate embedding
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True
    )

    # Search top 10 statutes
    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    # Convert FAISS positions to statute IDs
    retrieved_ids = [
        str(metadata[idx]["id"])
        for idx in indices[0]
    ]

    evaluated_queries += 1

    # --------------------------------------------------
    # Recall@1
    # --------------------------------------------------

    if any(
        statute_id in relevant_ids
        for statute_id in retrieved_ids[:1]
    ):
        recall_at_1 += 1

    # --------------------------------------------------
    # Recall@5
    # --------------------------------------------------

    if any(
        statute_id in relevant_ids
        for statute_id in retrieved_ids[:5]
    ):
        recall_at_5 += 1

    # --------------------------------------------------
    # Recall@10
    # --------------------------------------------------

    if any(
        statute_id in relevant_ids
        for statute_id in retrieved_ids[:10]
    ):
        recall_at_10 += 1

    # --------------------------------------------------
    # Reciprocal Rank
    # --------------------------------------------------

    reciprocal_rank = 0

    for rank, statute_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if statute_id in relevant_ids:
            reciprocal_rank = 1 / rank
            break

    mrr_total += reciprocal_rank

    # --------------------------------------------------
    # Progress
    # --------------------------------------------------

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i + 1}/{len(queries)} queries"
        )


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

if evaluated_queries > 0:

    recall_at_1_score = (
        recall_at_1 / evaluated_queries
    )

    recall_at_5_score = (
        recall_at_5 / evaluated_queries
    )

    recall_at_10_score = (
        recall_at_10 / evaluated_queries
    )

    mrr_score = (
        mrr_total / evaluated_queries
    )

else:

    recall_at_1_score = 0
    recall_at_5_score = 0
    recall_at_10_score = 0
    mrr_score = 0


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n" + "=" * 60)
print("ROLE-AWARE STATUTE RETRIEVAL EVALUATION")
print("=" * 60)

print(
    f"Total queries:       {len(queries)}"
)

print(
    f"Evaluated queries:   {evaluated_queries}"
)

print(
    f"Skipped queries:     {skipped_queries}"
)

print("\nMetrics:")

print(
    f"Recall@1:  {recall_at_1_score:.4f}"
)

print(
    f"Recall@5:  {recall_at_5_score:.4f}"
)

print(
    f"Recall@10: {recall_at_10_score:.4f}"
)

print(
    f"MRR:       {mrr_score:.4f}"
)

print("\nPercentages:")

print(
    f"Recall@1:  {recall_at_1_score * 100:.2f}%"
)

print(
    f"Recall@5:  {recall_at_5_score * 100:.2f}%"
)

print(
    f"Recall@10: {recall_at_10_score * 100:.2f}%"
)

print(
    f"MRR:       {mrr_score:.4f}"
)

print("=" * 60)