"""Provisional GT-relation rule for extra-source films (evaluator R4 / contract C15b), until the shared
GT-relation table lands on the main pipeline.

- Remakes of a GT film (same story) are excluded upstream: survey.py drops every title whose normalized
  form equals a GT title, whatever the year (this also drops a few unrelated same-title films).
- Sequels, prequels and same-universe films get `gt_related = {"type", "gt_movie", "evidence"}` and are
  left out of the `eval_safe` subset. Evidence is a curated franchise keyword in the title, or a
  GT base title that prefixes the film's title, optionally confirmed by shared character names.
"""
from __future__ import annotations

import re

from .imdb_index import norm_title

# GT movie_name -> title keywords of the same franchise/universe (normalized like norm_title, no articles).
GT_FRANCHISES = {
    "Toy Story 4_2019": ["toystory"],
    "The Batman_2022": ["batman", "darkknight", "joker", "catwoman", "gotham"],
    "Batman: Mask of the Phantasm_1993": ["batman", "darkknight"],
    "Star Wars: Episode I - The Phantom Menace_1999": ["starwars", "empirestrikesback", "returnofthejedi", "revengeofthejedi",
                                                       "rogueone", "solo", "forceawakens", "lastjedi", "riseofskywalker",
                                                       "attackoftheclones", "revengeofthesith"],
    "The Lord of the Rings: The Return of the King_2003": ["lordoftherings", "fellowshipofthering", "twotowers", "hobbit"],
    "The Wolverine_2013": ["xmen", "x2", "wolverine", "logan", "deadpool", "newmutants"],
    "Fast & Furious_2009": ["fastandthefurious", "fastandfurious", "2fast2furious", "fastfive", "furious7", "fateofthefurious",
                            "hobbsandshaw", "tokyodrift", "fastx"],
    "Black Panther: Wakanda Forever_2022": ["blackpanther", "wakanda"],
    "Blade Runner_1982": ["bladerunner"],
    "Scream_2022": ["scream"],
    "Evil Dead_2013": ["evildead", "armyofdarkness"],
    "Day of the Dead_1985": ["nightofthelivingdead", "dawnofthedead", "landofthedead", "diaryofthedead", "survivalofthedead"],
    "Horrible Bosses_2011": ["horriblebosses"],
    "Bad Moms_2016": ["badmoms", "badmomschristmas"],
    "Home Alone_1990": ["homealone"],
    "Diary of a Wimpy Kid_2010": ["diaryofawimpykid", "wimpykid"],
    "Maleficent_2014": ["maleficent", "sleepingbeauty"],
    "Snow White and the Huntsman_2012": ["huntsman"],
    "King Kong_1933": ["kingkong", "sonofkong", "kongskullisland"],
    "The Italian Job_2003": ["italianjob"],
    "All the King's Men_2006": ["allthekingsmen"],
    "Monty Python and the Holy Grail_1975": ["montypython", "lifeofbrian", "meaningoflife"],
    "12 Monkeys_1995": ["12monkeys", "twelvemonkeys", "lajetee"],
    "Constantine_2005": ["constantine", "hellblazer"],
    "8MM_1999": ["8mm"],
    "The Best Exotic Marigold Hotel_2011": ["bestexoticmarigoldhotel", "secondbestexoticmarigoldhotel"],
    "The Accountant_2016": ["accountant2"],
    "Backdraft_1991": ["backdraft"],
    "Mirrors_2008": ["mirrors2", "intothemirror"],
    "Young Frankenstein_1974": [],
}
_SEQUEL_TAIL_RE = re.compile(r"(\s*[:\-].*|\s+(part|chapter|episode|vol\.?|volume)\s+\w+.*|\s+(\d+|[ivx]+))$", re.I)


def base_title(title: str) -> str:
    return norm_title(_SEQUEL_TAIL_RE.sub("", title.strip()))


def relate(title: str, gt_names: list[str]) -> dict | None:
    """gt_related record for a film title, or None. `gt_names` are GT movie_name values (Title_Year)."""
    key = norm_title(title)
    for gt in gt_names:
        for kw in GT_FRANCHISES.get(gt, []):
            if kw and kw in key:
                return {"type": "sequel_or_series", "gt_movie": gt, "evidence": f"franchise_keyword:{kw}"}
    for gt in gt_names:
        b = base_title(gt.rpartition("_")[0])
        if len(b) >= 6 and key.startswith(b) and key != b:
            return {"type": "sequel_or_series", "gt_movie": gt, "evidence": f"gt_base_title_prefix:{b}"}
    return None
