import json
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("data/document_chunks.json")
OUTPUT_FILE = Path("data/document_summaries.json")

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

TARGET_DOCUMENT = (
    "Mohd_Ameeruddin_Anr_vs_United_India_Insurance_Co_Ltd_Anr_on_11_November_2010.PDF"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("=" * 70)
print("LEXASSIST DOCUMENT SUMMARIZER")
print("=" * 70)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading summarization model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded.")


# --------------------------------------------------
# Load chunks
# --------------------------------------------------

print("\nLoading document chunks...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    documents = json.load(f)

print(
    f"Documents available: {len(documents)}"
)


# --------------------------------------------------
# Find target document
# --------------------------------------------------

document = None

for item in documents:

    if item["file_name"] == TARGET_DOCUMENT:
        document = item
        break


if document is None:

    raise ValueError(
        f"Target document not found:\n{TARGET_DOCUMENT}"
    )


print("\nTarget document:")
print(document["file_name"])

print(
    f"Chunks: {document['chunk_count']}"
)


# --------------------------------------------------
# Summarize one chunk
# --------------------------------------------------

def summarize_chunk(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=30,
        num_beams=4,
        no_repeat_ngram_size=3
    )

    summary = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return summary.strip()


# --------------------------------------------------
# Process chunks
# --------------------------------------------------

chunk_summaries = []

print("\n" + "=" * 70)
print("CHUNK SUMMARIZATION")
print("=" * 70)

for chunk in document["chunks"]:

    print(
        f"\nSummarizing chunk {chunk['chunk_id']}"
        f" | pages {chunk['page_start']}-{chunk['page_end']}"
    )

    summary = summarize_chunk(
        chunk["text"]
    )

    print("Summary:")
    print(summary)

    chunk_summaries.append({
        "chunk_id": chunk["chunk_id"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "summary": summary
    })


# --------------------------------------------------
# Combine summaries
# --------------------------------------------------

combined_summary_text = "\n\n".join(
    item["summary"]
    for item in chunk_summaries
)


# --------------------------------------------------
# Save result
# --------------------------------------------------

result = {
    "file_name": document["file_name"],
    "page_count": document["page_count"],
    "chunk_count": document["chunk_count"],
    "chunk_summaries": chunk_summaries,
    "combined_summary": combined_summary_text
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )


# --------------------------------------------------
# Final output
# --------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARIZATION COMPLETE")
print("=" * 70)

print(
    f"Document: {document['file_name']}"
)

print(
    f"Chunks summarized: {len(chunk_summaries)}"
)

print(
    f"Output: {OUTPUT_FILE}"
)