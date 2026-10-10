"""Catalog parsers for screenplay sites. Each returns entries:

  {"source", "title", "year", "year_kind" ("release"|"draft"|None), "url", "format", "imdb_id" (or None),
   "page_url", "kind" ("film"|"tv")}

Only index pages are parsed here (no script text). Download policy lives in fetch.py.
"""
from __future__ import annotations

import html
import re
from urllib.parse import quote, unquote, urljoin

SCRIPT_EXTS = ("pdf", "txt", "html", "htm", "doc", "rtf", "docx")
# Hosts covered by their own catalog parser, so SimplyScripts links to them are not double counted.
DIRECT_HOSTS = ("imsdb.com", "dailyscript.com", "awesomefilm.com", "scriptslug.com")


def _fmt(url: str) -> str:
    ext = unquote(url).rsplit(".", 1)[-1].lower().split("?")[0]
    return {"htm": "html"}.get(ext, ext) if ext in SCRIPT_EXTS else "html"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def imsdb(index_html: str) -> list[dict]:
    """https://imsdb.com/all-scripts.html. The year is the draft date ("1997-11 Draft"); the script text page
    is /scripts/<Title-With-Dashes>.html (the detail page links to it and gives the release date)."""
    out = []
    for path, title, draft in re.findall(r'<a href="(/Movie Scripts/[^"]+)" title="[^"]+">([^<]+)</a>\s*\(([^)]*)\)', index_html):
        title = html.unescape(title).strip()
        m = re.search(r"(\d{4})", draft)
        slug = re.sub(r" Script\.html$", "", path.split("/Movie Scripts/")[1]).replace(" ", "-")
        out.append({
            "source": "imsdb", "title": title, "year": int(m.group(1)) if m else None, "year_kind": "draft" if m else None,
            "url": f"https://imsdb.com/scripts/{quote(slug)}.html", "format": "html", "imdb_id": None,
            "page_url": "https://imsdb.com" + quote(path), "kind": "film",
        })
    return out


def imsdb_detail(detail_html: str) -> dict:
    """Release date and script link from an IMSDb detail page."""
    text = _clean(detail_html[detail_html.find("script-details"):][:6000])
    rel = re.search(r"Movie Release Date\s*:\s*(?:\w+\s+)?(\d{4})", text)
    link = re.search(r'<a href="(/scripts/[^"]+)"[^>]*>Read (?:&quot;|")(.*?)(?:&quot;|") Script</a>', detail_html)
    return {"release_year": int(rel.group(1)) if rel else None,
            "script_url": "https://imsdb.com" + link.group(1) if link else None,
            "script_title": html.unescape(link.group(2)) if link else None}


def dailyscript(index_html: str, base: str = "https://www.dailyscript.com/") -> list[dict]:
    """https://www.dailyscript.com/movie.html and movie_n-z.html: title, writers, release year, IMDb link."""
    out = []
    for li in re.split(r"<p>|<li>|<br>", index_html, flags=re.I):
        m = re.search(r'<a href="(scripts/[^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a |$)', li, re.S | re.I)
        if not m:
            continue
        url, title, rest = m.group(1), _clean(m.group(2)), _clean(m.group(3))
        if not title or _fmt(url) in ("jpg",):
            continue
        year = re.search(r"\b(19\d\d|20\d\d)\b", rest)
        imdb = re.search(r"imdb\.com/(?:Title\?|title/tt)(\d{7,8})", li)
        out.append({
            "source": "dailyscript", "title": title, "year": int(year.group(1)) if year else None,
            "year_kind": "release" if year else None, "url": urljoin(base, url), "format": _fmt(url),
            "imdb_id": f"tt{imdb.group(1)}" if imdb else None, "page_url": base + "movie.html", "kind": "film",
        })
    return out


def awesomefilm(index_html: str, base: str = "https://www.awesomefilm.com/") -> list[dict]:
    out = []
    for url, title in re.findall(r'<a href="(script/[^"]+)"[^>]*>([^<]+)</a>', index_html):
        title = re.sub(r"\s+", " ", re.sub(r"\s*\(.*?\)\s*$", "", html.unescape(title))).strip()
        if re.fullmatch(r"part \d+", title, re.I):
            continue
        out.append({"source": "awesomefilm", "title": title, "year": None, "year_kind": None, "url": urljoin(base, url),
                    "format": _fmt(url), "imdb_id": None, "page_url": base, "kind": "film"})
    return out


def simplyscripts(index_html: str) -> list[dict]:
    """https://www.simplyscripts.com/movie-screenplays.html is an aggregator: each script link is followed by a
    link to its host site. Hosts with their own parser are skipped."""
    out = []
    for url, title in re.findall(r'<a href="(https?://[^"]+)"[^>]*class="style10"[^>]*>([^<]+)</a>', index_html):
        host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        if url.rstrip("/").count("/") < 3 or any(h in url.split("/web/")[-1] for h in DIRECT_HOSTS):
            continue
        if unquote(url).rsplit(".", 1)[-1].lower().split("?")[0] not in SCRIPT_EXTS:
            continue  # landing pages (studio award sites, library indexes), not script files
        title = html.unescape(title).strip()
        out.append({"source": f"simplyscripts:{host}", "title": title, "year": None, "year_kind": None, "url": url,
                    "format": _fmt(url), "imdb_id": None, "page_url": "https://www.simplyscripts.com/movie-screenplays.html",
                    "kind": "film"})
    return out


def scriptslug_sitemap(xml: str) -> list[dict]:
    """Catalog count only: Script Slug forbids use of its PDFs for AI training (robots.txt / ai.txt)."""
    out = []
    for slug in re.findall(r"<loc>https://www\.scriptslug\.com/script/([^<]+)</loc>", xml):
        m = re.match(r"(.+?)-(\d{4})$", slug)
        name, year = (m.group(1), int(m.group(2))) if m else (slug, None)
        tv = re.search(r"-(\d{3,4}|pilot)(-|$)", name)
        out.append({"source": "scriptslug", "title": name.replace("-", " "), "year": year, "year_kind": "release",
                    "url": f"https://www.scriptslug.com/script/{slug}", "format": "pdf", "imdb_id": None,
                    "page_url": "https://www.scriptslug.com/sitemap-scripts.xml", "kind": "tv" if tv else "film"})
    return out
