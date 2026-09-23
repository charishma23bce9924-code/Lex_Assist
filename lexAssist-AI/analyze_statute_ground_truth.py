import json
from collections import Counter


with open(
    "data/queries_test_clean.json",
    "r",
    encoding="utf-8"
) as f:
    queries = json.load(f)


with open(
    "data/statutes_clean.json",
    "r",
    encoding="utf-8"
) as f:
    statutes = json.load(f)


statute_ids = {
    str(statute["id"])
    for statute in statutes
}


counts = Counter()
missing_ids = 0
total_relevant = 0


for query in queries:

    relevant = query.get(
        "relevant_statute_ids",
        []
    )

    relevant = [
        str(x)
        for x in relevant
    ]

    counts[len(relevant)] += 1
    total_relevant += len(relevant)

    for statute_id in relevant:
        if statute_id not in statute_ids:
            missing_ids += 1


print("=" * 60)
print("STATUTE GROUND-TRUTH ANALYSIS")
print("=" * 60)

print("Test queries:", len(queries))
print("Total relevant statute links:", total_relevant)
print("Average relevant statutes/query:",
      total_relevant / len(queries))

print("\nDistribution:")

for count, number in sorted(counts.items()):
    print(
        f"{number} queries have {count} relevant statute(s)"
    )

print("\nRelevant IDs missing from 936-statute corpus:",
      missing_ids)

print("=" * 60)