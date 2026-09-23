from datasets import load_dataset

configs = ["statutes", "precedents", "queries"]

for config in configs:
    print("\n" + "=" * 70)
    print(f"CONFIG: {config}")
    print("=" * 70)

    ds = load_dataset("Exploration-Lab/IL-PCSR", config)

    print("\nSplits:")
    print(ds)

    for split_name, data in ds.items():
        print(f"\n--- Split: {split_name} ---")
        print("Number of records:", len(data))
        print("Columns:", data.column_names)
        print("\nFirst record:")
        print(data[0])