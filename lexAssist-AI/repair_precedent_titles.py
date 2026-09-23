import json
import re
from pathlib import Path


INPUT_FILE = Path("data/precedents_clean.json")
OUTPUT_FILE = Path("data/precedents_clean_repaired.json")


def clean_party(text):
    text = re.sub(r"\s+", " ", text).strip()

    # Remove header labels accidentally captured in party names
    text = re.sub(
        r"\s+(?:APPELLANT|APPELLANTS)\s*$",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+(?:RESPONDENT|RESPONDENTS)\s*$",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove a leading Vs./Versus from respondent
    text = re.sub(
        r"^\s*(?:Vs?\.?|Versus)\s+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove a trailing Vs./Versus from petitioner
    text = re.sub(
        r"\s+(?:Vs?\.?|Versus)\s*$",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip(" ,;:-")


def normalize_title(petitioner, respondent):
    title = f"{petitioner} v. {respondent}"

    # Normalize accidental duplicated separators such as:
    # "A v. Vs. B" -> "A v. B"
    # "A v. Versus B" -> "A v. B"
    title = re.sub(
        r"\bv\.\s*(?:Vs?\.?|Versus)\s*",
        "v. ",
        title,
        flags=re.IGNORECASE
    )

    # Remove header labels immediately before "v."
    # Example:
    # "A RESPONDENT v. B" -> "A v. B"
    # "A APPELLANT v. B" -> "A v. B"
    title = re.sub(
        r"\s+(APPELLANT|APPELLANTS|RESPONDENT|RESPONDENTS)\s+(?=v\.)",
        "",
        title,
        flags=re.IGNORECASE
    )

    # Keep existing cleanup for labels appearing elsewhere
    title = re.sub(
        r"\s+(APPELLANT|APPELLANTS|RESPONDENT|RESPONDENTS)\b",
        "",
        title,
        flags=re.IGNORECASE
    )

    # Final whitespace normalization
    title = re.sub(r"\s+", " ", title).strip()

    return title


def extract_title(text):
    # Only inspect the beginning of the judgment.
    header = text[:3000]

    # ---------------------------------------------------------
    # Pattern 1: PETITIONER / RESPONDENT
    # ---------------------------------------------------------
    match = re.search(
        r"PETITIONER\s*:\s*(.*?)"
        r"\s+RESPONDENT\s*:\s*(.*?)"
        r"(?=\s+(?:DATE\s+OF\s+JUDGMENT|DATE\s+OF\s+DECISION|"
        r"BENCH\s*:|JUDGMENT\s*:|HEADNOTE\s*:|ACT\s*:|$))",
        header,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        petitioner = clean_party(match.group(1))
        respondent = clean_party(match.group(2))

        if petitioner and respondent:
            return normalize_title(petitioner, respondent)

    # ---------------------------------------------------------
    # Pattern 2: APPELLANT / RESPONDENT
    # ---------------------------------------------------------
    match = re.search(
        r"APPELLANT\s*:\s*(.*?)"
        r"\s+RESPONDENT\s*:\s*(.*?)"
        r"(?=\s+(?:DATE\s+OF\s+JUDGMENT|DATE\s+OF\s+DECISION|"
        r"BENCH\s*:|JUDGMENT\s*:|HEADNOTE\s*:|ACT\s*:|$))",
        header,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        appellant = clean_party(match.group(1))
        respondent = clean_party(match.group(2))

        if appellant and respondent:
            return normalize_title(appellant, respondent)

    return None


def main():
    print("=" * 70)
    print("LEXASSIST PRECEDENT TITLE REPAIR")
    print("=" * 70)

    print("\nLoading original precedents...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total precedents: {len(data)}")

    repaired = 0
    still_missing = 0

    for record in data:

        # Existing titles are NEVER changed.
        if record.get("case_title"):
            continue

        title = extract_title(
            record.get("text", "")
        )

        if title:
            record["case_title"] = title
            repaired += 1
        else:
            still_missing += 1

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("REPAIR COMPLETE")
    print("=" * 70)

    print(f"Total precedents:       {len(data)}")
    print(f"Titles repaired:        {repaired}")
    print(f"Still missing titles:   {still_missing}")
    print(f"Titles available:       {len(data) - still_missing}")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()