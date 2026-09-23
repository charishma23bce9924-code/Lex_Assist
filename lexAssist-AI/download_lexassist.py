from datasets import load_dataset

print("Downloading IL-PCSR queries...")

ds = load_dataset(
    "Exploration-Lab/IL-PCSR",
    "queries"
)

print("\nDataset downloaded successfully!")
print(ds)

for split_name, split_data in ds.items():
    print(f"\nSplit: {split_name}")
    print("Columns:", split_data.column_names)
    print("Number of records:", len(split_data))

    print("\nFirst record:")
    print(split_data[0])