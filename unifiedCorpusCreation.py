import json
import re
import random
from collections import Counter
from statistics import median


# ============================================================
# CONFIGURATION
# ============================================================

PLOS_FILE = "./data/PLOSArticles_with_abstracts.json"
HNR_FILE = "./data/HealthNewsReview_processed.json"

OUTPUT_FILE = "./data/unified_corpus.json"

# Minimum text length
MIN_WORDS = 40

# Reproducibility
RANDOM_SEED = 42

# ------------------------------------------------------------
# Class balancing
#
# None = automatically balance to the smallest class
#
# Example:
# TARGET_PER_CLASS = 5000
#
# If None:
# trusted = 3000
# misleading = 1800
# -> 1800 records sampled from each class
# ------------------------------------------------------------

TARGET_PER_CLASS = None


# ============================================================
# UTILITIES
# ============================================================

def normalize_text(text):
    """
    Normalize whitespace and remove HTML-like tags.
    """

    if text is None:
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_count(text):
    """
    Count words.
    """

    return len(
        re.findall(
            r"\b\w+\b",
            text,
            flags=re.UNICODE
        )
    )


def make_absolute_plos_url(url):
    """
    Convert PLOS relative URLs to absolute URLs.
    """

    if not url:
        return ""

    url = str(url)

    if url.startswith("http://") or url.startswith("https://"):
        return url

    if url.startswith("\\/"):
        url = url.replace("\\/", "/")

    if url.startswith("/"):
        return "https://journals.plos.org" + url

    return "https://journals.plos.org/" + url


# ============================================================
# 1. PROCESS PLOS
# ============================================================

def process_plos(path):

    print("\n" + "=" * 70)
    print("PROCESSING PLOS")
    print("=" * 70)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        records = json.load(f)

    print(f"Input PLOS records: {len(records)}")

    processed = []

    dropped_short = 0
    dropped_empty = 0

    for index, r in enumerate(records):

        title = normalize_text(
            r.get("articleTitle", "")
        )

        abstract = normalize_text(
            r.get("abstract", "")
        )

        # ----------------------------------------------------
        # Build scientific text
        # ----------------------------------------------------

        text_parts = []

        if title:
            text_parts.append(title)

        if abstract:
            text_parts.append(abstract)

        text = " ".join(text_parts).strip()

        if not text:
            dropped_empty += 1
            continue

        # ----------------------------------------------------
        # Minimum length
        # ----------------------------------------------------

        n_words = word_count(text)

        if n_words < MIN_WORDS:
            dropped_short += 1
            continue

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        article_url = make_absolute_plos_url(
            r.get("articleLink", "")
        )

        scraped_url = r.get(
            "scraped_url",
            article_url
        )

        # ----------------------------------------------------
        # Provenance identifier
        #
        # Prefer DOI contained in the PLOS URL.
        # Otherwise use article URL.
        # ----------------------------------------------------

        provenance_signal = article_url

        doi_match = re.search(
            r"id=(10\.\d{4,9}/[^\s]+)",
            article_url
        )

        if doi_match:
            provenance_signal = doi_match.group(1)

        # ----------------------------------------------------
        # Canonical record
        # ----------------------------------------------------

        record = {

            "record_id":
                f"plos_{index:06d}",

            "text":
                text,

            "source_id":
                provenance_signal,

            "author":
                normalize_text(
                    r.get("authors", "")
                ),

            "platform":
                "PLOS",

            "url":
                scraped_url or article_url,

            "claim_id":
                provenance_signal,

            "credibility_label":
                1,

            "provenance_signal":
                provenance_signal,

            "origin":
                "PLOS",

            "rating":
                None,

            "tags":
                [],

            "meta": {
                "article_title": title,
                "article_link": article_url,
                "word_count": n_words,
                "scraping_status":
                    r.get(
                        "scraping_status",
                        None
                    )
            }
        }

        processed.append(record)

    print(f"Kept PLOS records : {len(processed)}")
    print(f"Dropped empty      : {dropped_empty}")
    print(f"Dropped < {MIN_WORDS} words : {dropped_short}")

    return processed


# ============================================================
# 2. PROCESS HEALTHNEWSREVIEW
# ============================================================

def process_healthnewsreview(path):

    print("\n" + "=" * 70)
    print("PROCESSING HEALTHNEWSREVIEW")
    print("=" * 70)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        records = json.load(f)

    print(
        f"Input HealthNewsReview records: "
        f"{len(records)}"
    )

    processed = []

    dropped_short = 0
    dropped_empty = 0
    dropped_rating_3 = 0

    for index, r in enumerate(records):

        # ----------------------------------------------------
        # Existing processed text
        # ----------------------------------------------------

        text = normalize_text(
            r.get("text", "")
        )

        if not text:
            dropped_empty += 1
            continue

        # ----------------------------------------------------
        # Minimum length
        # ----------------------------------------------------

        n_words = word_count(text)

        if n_words < MIN_WORDS:
            dropped_short += 1
            continue

        # ----------------------------------------------------
        # Rating
        # ----------------------------------------------------

        rating = r.get("rating")

        if rating is None:
            continue

        # ----------------------------------------------------
        # Binary credibility
        #
        # 1-2 -> 0
        # 3   -> removed
        # 4-5 -> 1
        # ----------------------------------------------------

        if rating <= 2:

            credibility_label = 0

        elif rating >= 4:

            credibility_label = 1

        else:

            dropped_rating_3 += 1
            continue

        # ----------------------------------------------------
        # Provenance
        # ----------------------------------------------------

        source_id = normalize_text(
            r.get(
                "source_id",
                "unknown"
            )
        )

        provenance_signal = normalize_text(
            r.get(
                "provenance_signal",
                source_id
            )
        )

        claim_id = normalize_text(
            r.get(
                "claim_id",
                f"hnr_{index:06d}"
            )
        )

        # ----------------------------------------------------
        # Canonical record
        # ----------------------------------------------------

        record = {

            "record_id":
                r.get(
                    "record_id",
                    f"hnr_{index:06d}"
                ),

            "text":
                text,

            "source_id":
                source_id,

            "author":
                normalize_text(
                    r.get(
                        "author",
                        ""
                    )
                ),

            "platform":
                "HealthNewsReview",

            "url":
                r.get(
                    "url",
                    ""
                ),

            "claim_id":
                claim_id,

            "credibility_label":
                credibility_label,

            "provenance_signal":
                provenance_signal,

            "origin":
                "HealthNewsReview",

            "rating":
                rating,

            "tags":
                r.get(
                    "tags",
                    []
                ),

            "meta": {

                "n_satisfactory":
                    r.get(
                        "n_satisfactory",
                        None
                    ),

                "n_not_satisfactory":
                    r.get(
                        "n_not_satisfactory",
                        None
                    ),

                "word_count":
                    n_words
            }
        }

        processed.append(record)

    print(
        f"Kept HealthNewsReview records : "
        f"{len(processed)}"
    )

    print(
        f"Dropped empty                  : "
        f"{dropped_empty}"
    )

    print(
        f"Dropped < {MIN_WORDS} words          : "
        f"{dropped_short}"
    )

    print(
        f"Dropped rating = 3             : "
        f"{dropped_rating_3}"
    )

    return processed


# ============================================================
# 3. BALANCE CLASSES
# ============================================================

def balance_classes(records):

    print("\n" + "=" * 70)
    print("BALANCING CLASSES")
    print("=" * 70)

    trusted = [
        r for r in records
        if r["credibility_label"] == 1
    ]

    misleading = [
        r for r in records
        if r["credibility_label"] == 0
    ]

    print(
        f"Trusted before balancing    : "
        f"{len(trusted)}"
    )

    print(
        f"Misleading before balancing : "
        f"{len(misleading)}"
    )

    # --------------------------------------------------------
    # Determine target
    # --------------------------------------------------------

    if TARGET_PER_CLASS is None:

        target = min(
            len(trusted),
            len(misleading)
        )

    else:

        target = min(
            TARGET_PER_CLASS,
            len(trusted),
            len(misleading)
        )

    print(
        f"Target per class            : "
        f"{target}"
    )

    random.seed(RANDOM_SEED)

    trusted_sample = random.sample(
        trusted,
        target
    )

    misleading_sample = random.sample(
        misleading,
        target
    )

    balanced = (
        trusted_sample +
        misleading_sample
    )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(balanced)

    print(
        f"Final balanced corpus       : "
        f"{len(balanced)}"
    )

    return balanced


# ============================================================
# 4. QUALITY CHECKS
# ============================================================

def quality_checks(records):

    print("\n" + "=" * 70)
    print("QUALITY CHECKS")
    print("=" * 70)

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    labels = [
        r["credibility_label"]
        for r in records
    ]

    label_counts = Counter(labels)

    print("\nClass distribution:")

    print(
        f"  Trusted (1)     : "
        f"{label_counts.get(1, 0)}"
    )

    print(
        f"  Misleading (0)  : "
        f"{label_counts.get(0, 0)}"
    )

    # --------------------------------------------------------
    # Origin distribution
    # --------------------------------------------------------

    origins = Counter(
        r["origin"]
        for r in records
    )

    print("\nOrigin distribution:")

    for origin, count in origins.items():

        print(
            f"  {origin}: {count}"
        )

    # --------------------------------------------------------
    # Platform distribution
    # --------------------------------------------------------

    platforms = Counter(
        r["platform"]
        for r in records
    )

    print("\nPlatform distribution:")

    for platform, count in platforms.items():

        print(
            f"  {platform}: {count}"
        )

    # --------------------------------------------------------
    # Text lengths
    # --------------------------------------------------------

    lengths = [
        word_count(r["text"])
        for r in records
    ]

    if lengths:

        print("\nText length:")

        print(
            f"  Minimum : {min(lengths)} words"
        )

        print(
            f"  Median  : {median(lengths):.1f} words"
        )

        print(
            f"  Maximum : {max(lengths)} words"
        )

        print(
            f"  Mean    : "
            f"{sum(lengths) / len(lengths):.1f} words"
        )

    # --------------------------------------------------------
    # Duplicate texts
    # --------------------------------------------------------

    texts = [
        r["text"].strip().lower()
        for r in records
    ]

    unique_texts = set(texts)

    duplicate_count = (
        len(texts) -
        len(unique_texts)
    )

    print("\nDuplicates:")

    print(
        f"  Duplicate records: "
        f"{duplicate_count}"
    )

    # --------------------------------------------------------
    # Provenance diversity
    # --------------------------------------------------------

    provenance = set(
        r["provenance_signal"]
        for r in records
        if r.get("provenance_signal")
    )

    print("\nProvenance:")

    print(
        f"  Unique provenance signals: "
        f"{len(provenance)}"
    )

    # --------------------------------------------------------
    # Missing labels
    # --------------------------------------------------------

    missing_labels = sum(
        1
        for r in records
        if r.get("credibility_label")
        not in [0, 1]
    )

    print("\nLabel integrity:")

    print(
        f"  Invalid labels: "
        f"{missing_labels}"
    )

    # --------------------------------------------------------
    # Missing provenance
    # --------------------------------------------------------

    missing_provenance = sum(
        1
        for r in records
        if not r.get("provenance_signal")
    )

    print(
        f"  Missing provenance: "
        f"{missing_provenance}"
    )

    # --------------------------------------------------------
    # Missing text
    # --------------------------------------------------------

    missing_text = sum(
        1
        for r in records
        if not r.get("text")
    )

    print(
        f"  Missing text: "
        f"{missing_text}"
    )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    print("\nChecks:")

    if duplicate_count == 0:
        print("  ✓ No duplicate texts")
    else:
        print(
            f"  ⚠ {duplicate_count} "
            f"duplicate texts"
        )

    if missing_labels == 0:
        print("  ✓ Labels valid")
    else:
        print("  ✗ Invalid labels detected")

    if missing_provenance == 0:
        print("  ✓ Provenance complete")
    else:
        print(
            "  ✗ Missing provenance detected"
        )

    if missing_text == 0:
        print("  ✓ Text complete")
    else:
        print(
            "  ✗ Missing text detected"
        )

    if lengths and median(lengths) >= 100:
        print(
            "  ✓ Median text length >= 100 words"
        )
    else:
        print(
            "  ⚠ Median text length < 100 words"
        )

    if len(provenance) >= 50:
        print(
            "  ✓ Provenance diversity >= 50"
        )
    else:
        print(
            f"  ⚠ Provenance diversity = "
            f"{len(provenance)} (< 50)"
        )


# ============================================================
# 5. SAVE
# ============================================================

def save_json(records, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nCorpus saved to:\n{path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("UNIFIED HEALTH CORPUS GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Process each source
    # --------------------------------------------------------

    plos_records = process_plos(
        PLOS_FILE
    )

    hnr_records = process_healthnewsreview(
        HNR_FILE
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = (
        plos_records +
        hnr_records
    )

    print("\n" + "=" * 70)
    print("COMBINED CORPUS")
    print("=" * 70)

    print(
        f"PLOS records             : "
        f"{len(plos_records)}"
    )

    print(
        f"HealthNewsReview records : "
        f"{len(hnr_records)}"
    )

    print(
        f"Combined records         : "
        f"{len(combined)}"
    )

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    balanced = balance_classes(
        combined
    )

    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    quality_checks(
        balanced
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_json(
        balanced,
        OUTPUT_FILE
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()