import json

with open(
    "data/queries_test_clean.json",
    "r",
    encoding="utf-8"
) as f:
    queries = json.load(f)

print("=" * 70)
print("RHETORICAL ROLE INSPECTION")
print("=" * 70)

for i in range(min(3, len(queries))):

    query = queries[i]

    print(f"\nQUERY {i + 1}")
    print("ID:", query["id"])

    print("\nRhetorical roles:")
    print(query["rhetorical_roles"])

    print("\nText length:")
    print(len(query["text"]))

    print("\nRelevant statute IDs:")
    print(query["relevant_statute_ids"])

    print("-" * 70)