import json

with open(
    "data/document_statute_results.json",
    encoding="utf-8"
) as f:
    data = json.load(f)

for document in data["documents"]:

    result = document["results"][0]

    print("\n" + "=" * 90)
    print(document["file_name"])
    print("=" * 90)

    print(
        "Chunk:",
        result["chunk_id"],
        "Pages:",
        result["page_start"],
        "-",
        result["page_end"]
    )

    for statute in result["results"]:
        print(
            f"  {statute['rank']}. "
            f"{statute['provision_name']} | "
            f"similarity={statute['similarity']:.4f}"
        )