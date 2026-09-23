import json
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("data/document_summaries.json")
OUTPUT_FILE = Path("data/structured_case_summaries.json")

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("=" * 70)
print("LEXASSIST STRUCTURED CASE SUMMARY")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded.")


# --------------------------------------------------
# Load existing chunk summaries
# --------------------------------------------------

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    document = json.load(f)


# --------------------------------------------------
# Generate a summary for a section
# --------------------------------------------------

def generate_summary(text, max_new_tokens=100):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=20,
        num_beams=4,
        no_repeat_ngram_size=3
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()


# --------------------------------------------------
# Combine chunk summaries
# --------------------------------------------------

chunk_summaries = document["chunk_summaries"]

combined_text = "\n\n".join(
    (
        f"Pages {item['page_start']}-{item['page_end']}: "
        f"{item['summary']}"
    )
    for item in chunk_summaries
)


# --------------------------------------------------
# Extract basic metadata directly from source
# --------------------------------------------------

case_name = document["file_name"]

# We deliberately don't infer these fields.
# They will be populated only if clearly present
# in the generated summary/source material.

structured = {
    "file_name": case_name,
    "page_count": document["page_count"],

    "case_overview": generate_summary(
        combined_text,
        max_new_tokens=120
    ),

    "facts": generate_summary(
        combined_text,
        max_new_tokens=120
    ),

    "legal_issue": generate_summary(
        combined_text,
        max_new_tokens=100
    ),

    "court_reasoning": generate_summary(
        combined_text,
        max_new_tokens=120
    ),

    "decision": generate_summary(
        combined_text,
        max_new_tokens=100
    ),

    "chunk_sources": [
        {
            "chunk_id": item["chunk_id"],
            "page_start": item["page_start"],
            "page_end": item["page_end"]
        }
        for item in chunk_summaries
    ]
}


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        structured,
        f,
        ensure_ascii=False,
        indent=2
    )


# --------------------------------------------------
# Display
# --------------------------------------------------

print("\n" + "=" * 70)
print("STRUCTURED SUMMARY")
print("=" * 70)

for key, value in structured.items():

    if key != "chunk_sources":

        print(f"\n{key.upper()}:")
        print(value)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"Output: {OUTPUT_FILE}"
)