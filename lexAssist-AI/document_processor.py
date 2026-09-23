import json
import re
from pathlib import Path
import pymupdf




# --------------------------------------------------
# Configuration
# --------------------------------------------------

DOCUMENTS_DIR = Path("documents/raw")
OUTPUT_FILE = Path("data/documents_clean.json")


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_text(text):
    """
    Basic document-level text cleaning.

    We intentionally keep this conservative.
    Legal text should not be aggressively rewritten.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Fix excessive spaces while preserving newlines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# Process one PDF
# --------------------------------------------------

def process_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path.name}")

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        raw_text = page.get_text("text")

        cleaned = clean_text(raw_text)

        pages.append({
            "page_number": page_number,
            "text": cleaned,
            "character_count": len(cleaned)
        })

    document.close()

    return {
        "document_id": pdf_path.stem,
        "file_name": pdf_path.name,
        "page_count": len(pages),
        "pages": pages
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    # Also support .PDF
    pdf_files += sorted(
        DOCUMENTS_DIR.glob("*.PDF")
    )

    # Remove duplicates
    pdf_files = list(dict.fromkeys(pdf_files))

    if not pdf_files:
        print("No PDF files found in documents/")
        return

    print("=" * 70)
    print("LEXASSIST DOCUMENT PROCESSOR")
    print("=" * 70)

    print(f"PDF files found: {len(pdf_files)}")

    documents = []

    for pdf_path in pdf_files:

        result = process_pdf(pdf_path)

        documents.append(result)

        print(
            f"Pages extracted: {result['page_count']}"
        )

        print(
            f"Characters extracted: "
            f"{sum(p['character_count'] for p in result['pages'])}"
        )

    # Save
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("DOCUMENT PROCESSING COMPLETE")
    print("=" * 70)

    print(f"Documents: {len(documents)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()