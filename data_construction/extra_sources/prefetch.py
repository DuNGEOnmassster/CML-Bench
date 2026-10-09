"""Download every candidate offer into the fetch cache, one thread per host (the per-host delay and robots
rules of fetch.Fetcher still apply), so build_extra.py can then run from the cache.

  python data_construction/extra_sources/prefetch.py --matches data_construction/work/extra_survey/catalog_matches.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from collections import Counter, defaultdict
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from extra_sources import catalogs  # noqa: E402
from extra_sources.build_extra import SOURCE_PRIORITY, candidate_films  # noqa: E402
from extra_sources.fetch import Fetcher, PolicyError  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", default="data_construction/work/extra_survey/catalog_matches.jsonl")
    ap.add_argument("--sources", default=",".join(SOURCE_PRIORITY))
    ap.add_argument("--cache_dir", default="data_construction/work/sources/extra/raw")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()
    films = candidate_films(args.matches, set(args.sources.split(",")))
    fetcher = Fetcher(args.cache_dir, delay=args.delay)
    by_host = defaultdict(list)
    for offers in films.values():
        for o in offers:
            url = o["page_url"] if o["source"] == "imsdb" else o["url"]
            by_host[urlsplit(url).netloc.lower()].append(o)
    outcome = Counter()
    lock = threading.Lock()

    def work(host, offers):
        for o in offers:
            try:
                if o["source"] == "imsdb":
                    page, _ = fetcher.get(o["page_url"])
                    det = catalogs.imsdb_detail(page.decode("latin-1"))
                    fetcher.get(det["script_url"] or o["url"])
                else:
                    fetcher.get(o["url"])
                res = "ok"
            except PolicyError:
                res = "policy"
            except RuntimeError:
                res = "failed"
            with lock:
                outcome[res] += 1
                done = sum(outcome.values())
            if done % 50 == 0:
                print(f"{done} offers fetched {dict(outcome)}", flush=True)

    threads = [threading.Thread(target=work, args=(h, offs), daemon=True) for h, offs in by_host.items()]
    print(f"{sum(len(v) for v in by_host.values())} offers on {len(by_host)} hosts", flush=True)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(dict(outcome))


if __name__ == "__main__":
    main()
