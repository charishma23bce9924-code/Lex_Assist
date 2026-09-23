import json


INPUT_FILE = "data/queries_test_clean.json"
OUTPUT_FILE = "data/queries_test_role_aware.json"


# Roles that are potentially useful for statute retrieval
SELECTED_ROLES = {
    "Issue",
    "Argument by Petitioner",
    "Argument by Respondent",
    "Statute Analysis",
    "Court Reasoning",
}


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    queries = json.load(f)


role_aware_queries = []


for query in queries:

    roles = query["rhetorical_roles"]
    original_text = query["original_text"]

    # original_text is a list of text segments
    selected_segments = []

    for role, segment in zip(roles, original_text):

        if role in SELECTED_ROLES:
            selected_segments.append(
                f"[{role}]\n{segment}"
            )

    # Combine selected legal sections
    role_aware_text = "\n\n".join(
        selected_segments
    )

    item = dict(query)

    # Preserve the original normalized text
    item["full_text"] = query["text"]

    # New retrieval representation
    item["text"] = role_aware_text

    role_aware_queries.append(item)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        role_aware_queries,
        f,
        ensure_ascii=False,
        indent=2
    )


print("=" * 60)
print("ROLE-AWARE QUERY CREATION")
print("=" * 60)

print("Original queries:", len(queries))
print("Role-aware queries:", len(role_aware_queries))

print("\nSelected roles:")

for role in SELECTED_ROLES:
    print("-", role)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nFirst query:")
print(role_aware_queries[0]["text"][:2000])