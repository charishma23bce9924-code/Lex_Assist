import json
from collections import Counter


INPUT_FILE = "data/precedent_segments.json"


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    segments = json.load(f)


print("=" * 70)
print("PRECEDENT SEGMENT ANALYSIS")
print("=" * 70)

print("Total segments:", len(segments))

# --------------------------------------------------
# Role distribution
# --------------------------------------------------

role_counts = Counter(
    segment["rhetorical_role"]
    for segment in segments
)

print("\nRhetorical role distribution:")

for role, count in role_counts.most_common():
    print(f"{role}: {count}")


# --------------------------------------------------
# Text length statistics
# --------------------------------------------------

lengths = [
    len(segment["text"])
    for segment in segments
    if segment["text"]
]

print("\nText length statistics:")

print("Minimum:", min(lengths))
print("Maximum:", max(lengths))
print(
    "Average:",
    sum(lengths) / len(lengths)
)


# --------------------------------------------------
# Very short segments
# --------------------------------------------------

short_segments = [
    segment
    for segment in segments
    if len(segment["text"]) < 50
]

print("\nSegments shorter than 50 characters:")
print(len(short_segments))


# --------------------------------------------------
# Empty segments
# --------------------------------------------------

empty_segments = [
    segment
    for segment in segments
    if not segment["text"].strip()
]

print("Empty segments:", len(empty_segments))


# --------------------------------------------------
# Sample records
# --------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE SEGMENTS")
print("=" * 70)

for i, segment in enumerate(segments[:5]):

    print(f"\nSegment {i + 1}")

    print("Precedent ID:",
          segment["precedent_id"])

    print("Segment ID:",
          segment["segment_id"])

    print("Role:",
          segment["rhetorical_role"])

    print("Text:")
    print(segment["text"][:500])

print("\n" + "=" * 70)