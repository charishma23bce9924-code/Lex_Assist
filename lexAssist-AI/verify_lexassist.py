import json
from pathlib import Path


DATA_DIR = Path("data")


def load_json(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


files = {
    "statutes_clean.json": 936,
    "precedents_clean.json": 3183,
    "queries_train_clean.json": 5017,
    "queries_dev_clean.json": 627,
    "queries_test_clean.json": 627,
}


print("=" * 60)
print("LEXASSIST DATA VERIFICATION")
print("=" * 60)

for filename, expected_count in files.items():

    records = load_json(filename)

    print(f"\n{filename}")
    print(f"Records: {len(records)}")
    print(f"Expected: {expected_count}")

    if len(records) == expected_count:
        print("STATUS: OK")
    else:
        print("STATUS: ERROR")

    # Check first record
    if records:
        print("Fields:")
        print(list(records[0].keys()))

        print("Text type:", type(records[0]["text"]).__name__)
        print("Text length:", len(records[0]["text"]))


print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)