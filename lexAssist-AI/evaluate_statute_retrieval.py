import json
import faiss
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_FILE = "data/statutes.faiss"
METADATA_FILE = "data/statutes_metadata.json"
QUERY_FILE = "data/queries_test_clean.json"

TOP_K = 10


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# Load FAISS index
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
# Load test queries
# --------------------------------------------------

with open(
    QUERY_FILE,
    "r",
    encoding="utf-8"
) as f:
    queries = json.load(f)

print(f"Test queries loaded: {len(queries)}")


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

print("\nRunning retrieval evaluation...\n")

for i, query in enumerate(queries):

    relevant_ids = query.get(
        "relevant_statute_ids",
        []
    )

    # Skip queries without ground-truth statutes
    if not relevant_ids:
        skipped_queries += 1
        continue

    relevant_ids = set(
        str(x) for x in relevant_ids
    )

    query_text = query["text"]

    # Generate query embedding
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True
    )

    # Search top 10
    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    retrieved_ids = [
        str(metadata[idx]["id"])
        for idx in indices[0]
    ]

    evaluated_queries += 1

    # ----------------------------------------------
    # Recall@1
    # ----------------------------------------------

    if any(
        doc_id in relevant_ids
        for doc_id in retrieved_ids[:1]
    ):
        recall_at_1 += 1

    # ----------------------------------------------
    # Recall@5
    # ----------------------------------------------

    if any(
        doc_id in relevant_ids
        for doc_id in retrieved_ids[:5]
    ):
        recall_at_5 += 1

    # ----------------------------------------------
    # Recall@10
    # ----------------------------------------------

    if any(
        doc_id in relevant_ids
        for doc_id in retrieved_ids[:10]
    ):
        recall_at_10 += 1

    # ----------------------------------------------
    # Reciprocal Rank
    # ----------------------------------------------

    reciprocal_rank = 0

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if doc_id in relevant_ids:
            reciprocal_rank = 1 / rank
            break

    mrr_total += reciprocal_rank

    # Progress
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
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("STATUTE RETRIEVAL EVALUATION")
print("=" * 60)

print(f"Total queries:       {len(queries)}")
print(f"Evaluated queries:   {evaluated_queries}")
print(f"Skipped queries:     {skipped_queries}")

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