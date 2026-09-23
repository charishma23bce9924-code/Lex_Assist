import json
from collections import Counter


QUERY_FILE = "data/queries_test_clean.json"
PRECEDENT_FILE = "data/precedents_clean.json"


# --------------------------------------------------
# Load test queries
# --------------------------------------------------

with open(
    QUERY_FILE,
    "r",
    encoding="utf-8"
) as f:
    queries = json.load(f)


# --------------------------------------------------
# Load precedent corpus
# --------------------------------------------------

with open(
    PRECEDENT_FILE,
    "r",
    encoding="utf-8"
) as f:
    precedents = json.load(f)


# --------------------------------------------------
# Build corpus ID set
# --------------------------------------------------

precedent_ids = {
    str(precedent["id"])
    for precedent in precedents
}


# --------------------------------------------------
# Analyze ground truth
# --------------------------------------------------

counts = Counter()

total_relevant = 0
missing_ids = 0

queries_with_relevant = 0
queries_without_relevant = 0


for query in queries:

    relevant = query.get(
        "relevant_precedent_ids",
        []
    )

    relevant = [
        str(x)
        for x in relevant
    ]

    counts[len(relevant)] += 1

    total_relevant += len(relevant)

    if relevant:
        queries_with_relevant += 1
    else:
        queries_without_relevant += 1

    for precedent_id in relevant:

        if precedent_id not in precedent_ids:
            missing_ids += 1


# --------------------------------------------------
# Results
# --------------------------------------------------

print("=" * 60)
print("PRECEDENT GROUND-TRUTH ANALYSIS")
print("=" * 60)

print(
    "Test queries:",
    len(queries)
)

print(
    "Queries with relevant precedents:",
    queries_with_relevant
)

print(
    "Queries without relevant precedents:",
    queries_without_relevant
)

print(
    "Total relevant precedent links:",
    total_relevant
)

if len(queries) > 0:
    print(
        "Average relevant precedents/query:",
        total_relevant / len(queries)
    )

print("\nDistribution:")

for count, number in sorted(counts.items()):

    print(
        f"{number} queries have "
        f"{count} relevant precedent(s)"
    )

print(
    "\nRelevant IDs missing from "
    "3,183-precedent corpus:",
    missing_ids
)

print("=" * 60)