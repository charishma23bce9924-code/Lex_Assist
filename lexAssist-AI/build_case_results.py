import json
from pathlib import Path


STATUTE_FILE = Path("data/document_statute_results.json")
PRECEDENT_FILE = Path("data/document_precedent_results.json")
OUTPUT_FILE = Path("data/case_results.json")


print("=" * 70)
print("LEXASSIST CASE RESULTS BUILDER")
print("=" * 70)


with open(STATUTE_FILE, "r", encoding="utf-8") as f:
    statute_data = json.load(f)

with open(PRECEDENT_FILE, "r", encoding="utf-8") as f:
    precedent_data = json.load(f)


precedent_by_file = {
    document["file_name"]: document
    for document in precedent_data["documents"]
}


combined_documents = []


for statute_document in statute_data["documents"]:

    file_name = statute_document["file_name"]

    if file_name not in precedent_by_file:
        raise ValueError(
            f"Precedent results missing for: {file_name}"
        )

    precedent_document = precedent_by_file[file_name]


    # ----------------------------------------------
    # Collect all statute results
    # ----------------------------------------------

    statute_results = []

    for chunk in statute_document["results"]:
        for result in chunk["results"]:

            statute_results.append({
                **result,
                "chunk_id": chunk["chunk_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"]
            })


    # ----------------------------------------------
    # Collect all precedent results
    # ----------------------------------------------

    precedent_results = []

    for chunk in precedent_document["results"]:
        for result in chunk["results"]:

            precedent_results.append({
                **result,
                "chunk_id": chunk["chunk_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"]
            })


    # ----------------------------------------------
    # Top 5 unique statutes for document
    # ----------------------------------------------

    best_statutes = {}

    for result in statute_results:

        key = result["statute_id"]

        if (
            key not in best_statutes
            or result["similarity"] > best_statutes[key]["similarity"]
        ):
            best_statutes[key] = result


    top_statutes = sorted(
        best_statutes.values(),
        key=lambda x: x["similarity"],
        reverse=True
    )[:5]


    # ----------------------------------------------
    # Top 5 unique precedents for document
    # ----------------------------------------------

    best_precedents = {}

    for result in precedent_results:

        key = result["precedent_id"]

        if (
            key not in best_precedents
            or result["similarity"] > best_precedents[key]["similarity"]
        ):
            best_precedents[key] = result


    top_precedents = sorted(
        best_precedents.values(),
        key=lambda x: x["similarity"],
        reverse=True
    )[:5]


    # ----------------------------------------------
    # Combined document
    # ----------------------------------------------

    combined_documents.append({
        "file_name": file_name,
        "page_count": statute_document["page_count"],
        "chunk_count": statute_document["chunk_count"],

        "top_statutes": top_statutes,

        "top_precedents": top_precedents,

        "statutes": statute_document["results"],

        "precedents": precedent_document["results"]
    })


# ----------------------------------------------
# Save
# ----------------------------------------------

output = {
    "document_count": len(combined_documents),
    "documents": combined_documents
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nDocuments:", len(combined_documents))

for document in combined_documents:

    print(
        f"\n{document['file_name']}"
    )

    print(
        f"  Top statutes: {len(document['top_statutes'])}"
    )

    print(
        f"  Top precedents: {len(document['top_precedents'])}"
    )

    print(
        f"  Full-analysis chunks: {document['chunk_count']}"
    )


print("\n" + "=" * 70)
print("CASE RESULTS BUILD COMPLETE")
print("=" * 70)