from datasets import load_dataset
import json
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Helper: convert list-based text to one string
# --------------------------------------------------

def normalize_text(text):
    if isinstance(text, list):
        return "\n".join(str(item).strip() for item in text if item)
    return str(text).strip()


# --------------------------------------------------
# Save dataset while preserving original metadata
# --------------------------------------------------

def save_json(records, filename):
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} records -> {output_path}")


# --------------------------------------------------
# 1. STATUTES
# --------------------------------------------------

print("\nLoading statutes...")

statutes_ds = load_dataset(
    "Exploration-Lab/IL-PCSR",
    "statutes"
)

statutes = statutes_ds["statute_candidates"]

statute_records = []

for record in statutes:
    item = dict(record)

    # Preserve original fields
    item["original_text"] = record["text"]

    # Create normalized searchable text
    item["text"] = normalize_text(record["text"])

    statute_records.append(item)

save_json(
    statute_records,
    "statutes_clean.json"
)


# --------------------------------------------------
# 2. PRECEDENTS
# --------------------------------------------------

print("\nLoading precedents...")

precedents_ds = load_dataset(
    "Exploration-Lab/IL-PCSR",
    "precedents"
)

precedents = precedents_ds["precedent_candidates"]

precedent_records = []

for record in precedents:
    item = dict(record)

    # Preserve original text
    item["original_text"] = record["text"]

    # Create normalized searchable text
    item["text"] = normalize_text(record["text"])

    precedent_records.append(item)

save_json(
    precedent_records,
    "precedents_clean.json"
)


# --------------------------------------------------
# 3. QUERIES
# --------------------------------------------------

print("\nLoading queries...")

queries_ds = load_dataset(
    "Exploration-Lab/IL-PCSR",
    "queries"
)

query_files = {
    "train_queries": "queries_train_clean.json",
    "dev_queries": "queries_dev_clean.json",
    "test_queries": "queries_test_clean.json"
}

for split_name, filename in query_files.items():

    queries = queries_ds[split_name]

    query_records = []

    for record in queries:
        item = dict(record)

        # Preserve original text
        item["original_text"] = record["text"]

        # Create normalized searchable text
        item["text"] = normalize_text(record["text"])

        query_records.append(item)

    save_json(
        query_records,
        filename
    )


# --------------------------------------------------
# 4. Verification
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Statutes:   {len(statute_records)}")
print(f"Precedents: {len(precedent_records)}")

for split_name in query_files:
    print(f"{split_name}: {len(queries_ds[split_name])}")

print("\nOutput directory:")
print(OUTPUT_DIR.resolve())