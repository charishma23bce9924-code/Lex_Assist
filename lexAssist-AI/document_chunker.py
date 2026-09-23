import json
import re
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("data/documents_processed.json")
OUTPUT_FILE = Path("data/document_chunks.json")

TARGET_WORDS = 300
OVERLAP_WORDS = 50
MIN_WORDS = 100


# --------------------------------------------------
# Text utilities
# --------------------------------------------------

def normalize_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_source_boilerplate(text, document_title):
    """
    Remove only source/PDF boilerplate that we have
    already observed in the actual documents.

    Legal content is intentionally preserved.
    """

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        # Indian Kanoon footer
        if line.startswith(
            "Indian Kanoon - http://indiankanoon.org/doc/"
        ):
            continue

        # Repeated source title
        if document_title and line == document_title:
            continue

        lines.append(line)

    return "\n".join(lines)


def split_into_paragraphs(text):

    text = normalize_text(text)

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    return [
        p.strip()
        for p in paragraphs
        if p.strip()
    ]


def split_long_paragraph(paragraph):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        paragraph
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def word_count(text):
    return len(text.split())


# --------------------------------------------------
# Build document units
# --------------------------------------------------

def build_units(document):

    units = []

    # The title is stored in the document metadata/file name,
    # so the repeated source title doesn't need to be embedded
    # in every retrieval chunk.
    document_title = ""

    if document["pages"]:
        first_page = document["pages"][0]["text"]

        first_lines = [
            line.strip()
            for line in first_page.split("\n")
            if line.strip()
        ]

        if first_lines:
            document_title = first_lines[0]

    for page in document["pages"]:

        page_number = page["page_number"]

        cleaned_page = remove_source_boilerplate(
            page["text"],
            document_title
        )

        paragraphs = split_into_paragraphs(
            cleaned_page
        )

        for paragraph in paragraphs:

            if word_count(paragraph) <= TARGET_WORDS:

                units.append({
                    "text": paragraph,
                    "page_number": page_number
                })

            else:

                sentences = split_long_paragraph(
                    paragraph
                )

                for sentence in sentences:

                    units.append({
                        "text": sentence,
                        "page_number": page_number
                    })

    return units


# --------------------------------------------------
# Create chunks with guaranteed overlap
# --------------------------------------------------

def create_chunks(units):
    """
    Create chunks around TARGET_WORDS with a true
    word-level OVERLAP_WORDS between consecutive chunks.

    Page information is preserved for every word.
    """

    # Flatten units into word-level records
    words = []

    for unit in units:
        unit_words = unit["text"].split()

        for word in unit_words:
            words.append({
                "word": word,
                "page_number": unit["page_number"]
            })

    chunks = []

    start = 0
    total_words = len(words)

    while start < total_words:

        end = min(
            start + TARGET_WORDS,
            total_words
        )

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunk_text = " ".join(
            item["word"]
            for item in chunk_words
        )

        chunks.append({
            "page_start": chunk_words[0]["page_number"],
            "page_end": chunk_words[-1]["page_number"],
            "text": chunk_text
        })

        # --------------------------------------------------
        # Move forward while retaining exact overlap
        # --------------------------------------------------

        if end >= total_words:
            break

        start = end - OVERLAP_WORDS

    return chunks

# --------------------------------------------------
# Merge tiny final chunks
# --------------------------------------------------

def merge_small_chunks(chunks):

    if len(chunks) <= 1:
        return chunks

    result = []

    for chunk in chunks:

        words = word_count(
            chunk["text"]
        )

        if (
            result
            and words < MIN_WORDS
        ):

            previous = result[-1]

            previous["text"] = (
                previous["text"]
                + "\n\n"
                + chunk["text"]
            )

            previous["page_end"] = (
                chunk["page_end"]
            )

        else:

            result.append(chunk)

    return result


# --------------------------------------------------
# Process document
# --------------------------------------------------

def process_document(document):

    units = build_units(
        document
    )

    chunks = create_chunks(
        units
    )

    chunks = merge_small_chunks(
        chunks
    )

    final_chunks = []

    for chunk_id, chunk in enumerate(
        chunks,
        start=1
    ):

        text = chunk["text"]

        final_chunks.append({
            "chunk_id": chunk_id,
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "word_count": word_count(text),
            "text": text
        })

    return {
        "document_id": document["document_id"],
        "file_name": document["file_name"],
        "page_count": document["page_count"],
        "chunk_count": len(final_chunks),
        "chunks": final_chunks
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("LEXASSIST LEGAL DOCUMENT CHUNKER")
    print("=" * 70)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        documents = json.load(f)

    print(
        f"Documents loaded: {len(documents)}"
    )

    processed_documents = []

    total_chunks = 0

    for document in documents:

        result = process_document(
            document
        )

        processed_documents.append(
            result
        )

        total_chunks += result["chunk_count"]

        print("\n" + "-" * 70)
        print(document["file_name"])
        print(
            f"Pages: {result['page_count']}"
        )
        print(
            f"Chunks: {result['chunk_count']}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            processed_documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)

    print(
        f"Documents: {len(processed_documents)}"
    )

    print(
        f"Total chunks: {total_chunks}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()