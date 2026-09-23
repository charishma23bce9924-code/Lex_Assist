import json


INPUT_FILE = "data/precedents_clean.json"


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    precedents = json.load(f)


print("=" * 70)
print("PRECEDENT STRUCTURE INSPECTION")
print("=" * 70)

print("Total precedents:", len(precedents))


for i in range(min(3, len(precedents))):

    precedent = precedents[i]

    roles = precedent["rhetorical_roles"]
    original_text = precedent["original_text"]

    print("\n" + "-" * 70)
    print(f"PRECEDENT {i + 1}")
    print("-" * 70)

    print("ID:", precedent["id"])
    print("Case title:", precedent["case_title"])
    print("Jurisdiction:", precedent["jurisdiction"])
    print("Date:", precedent["date"])

    print("\nNumber of rhetorical roles:")
    print(len(roles))

    print("Number of original text segments:")
    print(len(original_text))

    if len(roles) != len(original_text):
        print("WARNING: ROLE/TEXT LENGTH MISMATCH")
    else:
        print("ROLE/TEXT ALIGNMENT: OK")

    print("\nRhetorical roles:")
    print(roles)

    print("\nRelevant statute IDs:")
    print(precedent["relevant_statute_ids"])

    print("\nRelevant precedent IDs:")
    print(precedent["relevant_precedent_ids"])

    print("\nFirst 3 aligned segments:")

    for role, segment in zip(
        roles[:3],
        original_text[:3]
    ):
        print(f"\n[{role}]")
        print(segment[:1000])