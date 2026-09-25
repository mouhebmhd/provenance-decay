import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import threading
import time

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

input_file = "./data/PLOSArticles.json"
output_file = "./PLOSArticles_with_abstracts.json"

MAX_ARTICLES = 1500      # Set to None to process all articles
MAX_WORKERS = 10         # Number of simultaneous requests
TIMEOUT = 10


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

dataSource = pd.read_json(input_file)

print(f"Number of articles: {len(dataSource)}")
print(dataSource.columns.tolist())

if MAX_ARTICLES is not None:
    dataSource = dataSource.iloc[:MAX_ARTICLES].copy()

print(f"Articles to scrape: {len(dataSource)}")


# --------------------------------------------------
# 3. Thread-local session
# --------------------------------------------------

thread_local = threading.local()


def get_session():
    """
    Create one requests.Session per thread.
    Sessions are reused for multiple requests.
    """
    if not hasattr(thread_local, "session"):
        session = requests.Session()

        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/153.0 Safari/537.36"
            )
        })

        thread_local.session = session

    return thread_local.session


# --------------------------------------------------
# 4. Function to scrape one abstract
# --------------------------------------------------

def get_abstract(url):
    try:
        # Handle both relative and absolute URLs
        full_url = urljoin(
            "https://journals.plos.org/",
            str(url)
        )

        session = get_session()

        response = session.get(
            full_url,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # ------------------------------------------
        # Main PLOS abstract selector
        # ------------------------------------------

        abstract_div = soup.select_one(
            "div.abstract-content"
        )

        if abstract_div:

            paragraphs = abstract_div.find_all("p")

            abstract = " ".join(
                p.get_text(" ", strip=True)
                for p in paragraphs
            )

            if abstract.strip():
                return {
                    "abstract": abstract.strip(),
                    "status": "success",
                    "url": full_url
                }

        # ------------------------------------------
        # Fallback: citation_abstract metadata
        # ------------------------------------------

        meta_abstract = soup.find(
            "meta",
            attrs={"name": "citation_abstract"}
        )

        if meta_abstract and meta_abstract.get("content"):
            return {
                "abstract": meta_abstract["content"].strip(),
                "status": "success_meta",
                "url": full_url
            }

        return {
            "abstract": None,
            "status": "abstract_not_found",
            "url": full_url
        }

    except requests.exceptions.RequestException as e:

        return {
            "abstract": None,
            "status": f"request_error: {type(e).__name__}",
            "url": str(url)
        }

    except Exception as e:

        return {
            "abstract": None,
            "status": f"error: {type(e).__name__}",
            "url": str(url)
        }


# --------------------------------------------------
# 5. Multithreaded scraping
# --------------------------------------------------

results = [None] * len(dataSource)

print("\nStarting multithreaded scraping...")
print(f"Workers: {MAX_WORKERS}\n")

start_time = time.time()

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    future_to_index = {
        executor.submit(
            get_abstract,
            row["articleLink"]
        ): index

        for index, row in dataSource.iterrows()
    }

    completed = 0

    for future in as_completed(future_to_index):

        index = future_to_index[future]

        try:
            result = future.result()

        except Exception as e:
            result = {
                "abstract": None,
                "status": f"worker_error: {type(e).__name__}",
                "url": str(dataSource.loc[index, "articleLink"])
            }

        results[index] = result

        completed += 1

        # Progress
        if result["abstract"] is not None:
            status = "OK"
        else:
            status = result["status"]

        print(
            f"[{completed}/{len(dataSource)}] "
            f"Index={index} | {status}"
        )


# --------------------------------------------------
# 6. Add results to dataset
# --------------------------------------------------

dataSource["abstract"] = [
    r["abstract"] if r else None
    for r in results
]

dataSource["scraping_status"] = [
    r["status"] if r else "unknown"
    for r in results
]

dataSource["scraped_url"] = [
    r["url"] if r else None
    for r in results
]


# --------------------------------------------------
# 7. Save dataset
# --------------------------------------------------

dataSource.to_json(
    output_file,
    orient="records",
    force_ascii=False,
    indent=2
)


# --------------------------------------------------
# 8. Statistics
# --------------------------------------------------

elapsed = time.time() - start_time

successful = dataSource["abstract"].notna().sum()
failed = len(dataSource) - successful

print("\n" + "=" * 60)
print("SCRAPING COMPLETED")
print("=" * 60)

print(f"Total articles : {len(dataSource)}")
print(f"Successful     : {successful}")
print(f"Failed         : {failed}")
print(f"Time           : {elapsed:.2f} seconds")
print(f"Saved to       : {output_file}")