import json
import re
from pathlib import Path
from collections import Counter


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("data/documents_clean.json")
OUTPUT_FILE = Path("data/documents_processed.json")


# --------------------------------------------------
# Basic text cleaning
# --------------------------------------------------

def clean_page_text(text):
    """
    Conservative cleaning for extracted legal documents.

    Removes only clearly identified extraction/layout noise.
    Legal content such as sections, citations, case names,
    dates, and reasoning is preserved.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")
    text = text.replace("\u00a0", " ")

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        # Remove Indian Kanoon source footer
        if line.startswith(
            "Indian Kanoon - http://indiankanoon.org/doc/"
        ):
            continue

        lines.append(line)

    # Normalize spaces inside lines
    lines = [
        re.sub(r"[ \t]+", " ", line)
        for line in lines
    ]

    text = "\n".join(lines)

    # Preserve paragraph separation without excessive blanks
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# Detect suspicious pages
# --------------------------------------------------

def page_diagnostics(pages):
    diagnostics = []

    for page in pages:

        text = page["text"].strip()

        diagnostics.append({
            "page_number": page["page_number"],
            "character_count": len(text),
            "word_count": len(text.split()),
            "is_empty": len(text) == 0,
            "is_suspiciously_short": len(text) < 100
        })

    return diagnostics


# --------------------------------------------------
# Find repeated short lines
# --------------------------------------------------

def find_repeated_lines(pages):
    """
    Finds lines that occur repeatedly across pages.

    We only REPORT these here.
    We do not automatically delete them yet.
    """

    counter = Counter()

    for page in pages:

        lines = page["text"].split("\n")

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # Ignore very long lines
            if len(line) > 150:
                continue

            counter[line] += 1

    repeated = []

    for line, count in counter.items():

        if count >= 3:

            repeated.append({
                "text": line,
                "count": count
            })

    repeated.sort(
        key=lambda x: x["count"],
        reverse=True
    )

    return repeated


# --------------------------------------------------
# Process one document
# --------------------------------------------------

def process_document(document):

    processed_pages = []

    for page in document["pages"]:

        cleaned_text = clean_page_text(
            page["text"]
        )

        processed_pages.append({
            "page_number": page["page_number"],
            "text": cleaned_text,
            "character_count": len(cleaned_text),
            "word_count": len(cleaned_text.split())
        })

    diagnostics = page_diagnostics(
        processed_pages
    )

    repeated_lines = find_repeated_lines(
        processed_pages
    )

    return {
        "document_id": document["document_id"],
        "file_name": document["file_name"],
        "page_count": document["page_count"],
        "pages": processed_pages,
        "diagnostics": diagnostics,
        "repeated_lines": repeated_lines
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("LEXASSIST DOCUMENT PREPROCESSOR")
    print("=" * 70)

    # Load extracted documents
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        documents = json.load(f)

    print(f"Documents loaded: {len(documents)}")

    processed_documents = []

    for document in documents:

        print("\n" + "-" * 70)
        print(document["file_name"])

        processed = process_document(
            document
        )

        processed_documents.append(
            processed
        )

        suspicious_pages = [
            d
            for d in processed["diagnostics"]
            if d["is_empty"]
            or d["is_suspiciously_short"]
        ]

        print(
            f"Pages: {processed['page_count']}"
        )

        print(
            f"Suspicious/short pages: "
            f"{len(suspicious_pages)}"
        )

        print(
            f"Repeated candidate lines: "
            f"{len(processed['repeated_lines'])}"
        )

        if suspicious_pages:

            print("Suspicious pages:")

            for page in suspicious_pages[:10]:

                print(
                    f"  Page {page['page_number']}: "
                    f"{page['character_count']} characters"
                )

        if processed["repeated_lines"]:

            print("Top repeated lines:")

            for item in processed["repeated_lines"][:10]:

                print(
                    f"  [{item['count']}x] "
                    f"{item['text']}"
                )

    # Save processed documents
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
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()