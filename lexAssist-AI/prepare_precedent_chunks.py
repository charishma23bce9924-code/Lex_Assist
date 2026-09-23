import json
from pathlib import Path


INPUT_FILE = Path("data/precedents_clean.json")
OUTPUT_FILE = Path("data/precedent_segments.json")


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    precedents = json.load(f)


segments = []

for precedent in precedents:

    roles = precedent["rhetorical_roles"]
    text_segments = precedent["original_text"]

    if len(roles) != len(text_segments):
        raise ValueError(
            f"Role/text mismatch for precedent "
            f"{precedent['id']}"
        )

    for segment_id, (role, text) in enumerate(
        zip(roles, text_segments)
    ):

        text = str(text).strip()

        if not text:
            continue

        segments.append({
            "segment_id": segment_id,
            "precedent_id": precedent["id"],
            "case_title": precedent["case_title"],
            "date": precedent["date"],
            "jurisdiction": precedent["jurisdiction"],
            "rhetorical_role": role,
            "text": text,

            # Preserve ground-truth metadata
            "relevant_statute_ids":
                precedent["relevant_statute_ids"],

            "relevant_precedent_ids":
                precedent["relevant_precedent_ids"]
        })


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        segments,
        f,
        ensure_ascii=False,
        indent=2
    )


print("=" * 60)
print("PRECEDENT SEGMENT CREATION")
print("=" * 60)

print(
    "Original precedents:",
    len(precedents)
)

print(
    "Generated segments:",
    len(segments)
)

print(
    "Saved:",
    OUTPUT_FILE
)

print("=" * 60)