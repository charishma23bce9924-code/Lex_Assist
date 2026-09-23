import json
import faiss
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_FILE = "data/statutes.faiss"
METADATA_FILE = "data/statutes_metadata.json"
CASE_RESULTS_FILE = "data/case_results.json"


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# --------------------------------------------------
# Load statute FAISS index
# --------------------------------------------------

print("Loading statute FAISS index...")

index = faiss.read_index(INDEX_FILE)

print(f"Statute FAISS vectors: {index.ntotal}")


# --------------------------------------------------
# Load statute metadata
# --------------------------------------------------

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Statute metadata records: {len(metadata)}")


# --------------------------------------------------
# Load combined case results
# --------------------------------------------------

print("Loading case results...")

with open(CASE_RESULTS_FILE, "r", encoding="utf-8") as f:
    case_data = json.load(f)

cases = case_data["documents"]

print(f"Cases available: {len(cases)}")


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    query = ""

    if request.method == "POST":

        query = request.form.get("query", "").strip()

        if query:

            # Create query embedding
            query_embedding = model.encode(
                [query],
                normalize_embeddings=True
            )

            # Search top 5 statutes
            scores, indices = index.search(
                query_embedding,
                5
            )

            # Prepare statute results
            for rank, (score, idx) in enumerate(
                zip(scores[0], indices[0]),
                start=1
            ):

                result = metadata[idx]

                results.append({
                    "rank": rank,
                    "id": result["id"],
                    "provision_name": result["provision_name"],
                    "similarity": float(score)
                })

    return render_template(
        "index.html",
        query=query,
        results=results,
        cases=cases
    )


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)