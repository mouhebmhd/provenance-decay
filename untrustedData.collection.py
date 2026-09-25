import json
import uuid
from collections import Counter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

input_file = "./data/healthRelease.json"
output_file = "./data/HealthNewsReview_processed.json"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

with open(input_file, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Number of records: {len(records)}")


# --------------------------------------------------
# 3. Process records
# --------------------------------------------------

processed = []

for r in records:

    # ----------------------------------------------
    # Build textual representation
    # ----------------------------------------------

    summary = r.get("summary", {})

    text_parts = [
        r.get("original_title", ""),
        r.get("description", ""),
        summary.get("Our Review Summary", "")
    ]

    text = " ".join(
        str(p).strip()
        for p in text_parts
        if p
    ).strip()

    if not text:
        continue


    # ----------------------------------------------
    # Rating
    # ----------------------------------------------

    rating = r.get("rating")

    if rating is None:
        continue


    # ----------------------------------------------
    # Binary credibility label
    #
    # 4-5 = credible
    # 1-2 = low credibility
    # 3   = ambiguous -> excluded
    # ----------------------------------------------

    if rating >= 4:
        credibility_label = 1

    elif rating <= 2:
        credibility_label = 0

    else:
        continue


    # ----------------------------------------------
    # Criteria
    # ----------------------------------------------

    criteria = r.get("criteria", [])

    n_satisfactory = sum(
        1
        for criterion in criteria
        if criterion.get("answer") == "Satisfactory"
    )

    n_not_satisfactory = sum(
        1
        for criterion in criteria
        if criterion.get("answer") == "Not Satisfactory"
    )


    # ----------------------------------------------
    # Reviewers
    # ----------------------------------------------

    reviewers = r.get("reviewers", [])

    author = ", ".join(
        str(x).strip()
        for x in reviewers
        if x
    )


    # ----------------------------------------------
    # Create standardized record
    # ----------------------------------------------

    processed_record = {

        "record_id": str(uuid.uuid4()),

        "text": text,

        "source_id": r.get(
            "news_source",
            "unknown"
        ),

        "author": author,

        "platform": "HealthNewsReview",

        "url": r.get(
            "source_link",
            ""
        ),

        "claim_id": r.get(
            "news_id",
            str(uuid.uuid4())
        ),

        "credibility_label": credibility_label,

        "provenance_signal": r.get(
            "news_source",
            "unknown"
        ),

        "rating": rating,

        "n_satisfactory": n_satisfactory,

        "n_not_satisfactory": n_not_satisfactory,

        "tags": r.get(
            "tags",
            []
        )
    }

    processed.append(processed_record)


# --------------------------------------------------
# 4. Save
# --------------------------------------------------

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        processed,
        f,
        ensure_ascii=False,
        indent=2
    )


# --------------------------------------------------
# 5. Statistics
# --------------------------------------------------

labels = [
    r["credibility_label"]
    for r in processed
]

label_counts = Counter(labels)

ratings = Counter(
    r["rating"]
    for r in records
    if r.get("rating") is not None
)

print("\n" + "=" * 60)
print("HEALTHNEWSREVIEW PROCESSING")
print("=" * 60)

print(f"Original records       : {len(records)}")
print(f"Processed records      : {len(processed)}")

print("\nOriginal rating distribution:")
for rating in sorted(ratings):
    print(
        f"Rating {rating}: "
        f"{ratings[rating]}"
    )

print("\nBinary classes:")
print(
    f"Credible (1)           : "
    f"{label_counts.get(1, 0)}"
)

print(
    f"Low credibility (0)    : "
    f"{label_counts.get(0, 0)}"
)

print(
    f"Total excluded (rating 3): "
    f"{len(records) - len(processed)}"
)

print(f"\nSaved to: {output_file}")