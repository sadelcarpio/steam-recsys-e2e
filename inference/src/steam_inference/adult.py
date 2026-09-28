"""Adult-content filter: games excluded from everything the pipeline publishes (recommendations,
the popularity fallback, the online catalog and the frontend's search index).

Steam has no complete flag in what we scrape: `required_age` is rarely set and mostly marks
violence, and the "Sexual Content" / "Nudity" genres are set by few developers. Explicit games
usually say so in their name, so a game is adult when it has one of those genres or its name
has one of the unambiguous words below. Deliberately conservative: a false positive hides one
game, a false negative shows explicit art in a demo.

Steam user tags (spec 8) add the players' votes: `NSFW` / `Hentai` anywhere in a game's tags, or
`Sexual Content` / `Nudity` among its top tags (further down, those also show up on mainstream
games such as RPGs with a romance scene).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from steam_inference.features import Games

ADULT_GENRES = frozenset({"Sexual Content", "Nudity"})
ADULT_NAME = re.compile(
    r"\b(?:hentai\w*|nsfw|porn\w*|sex|sexy\w*|sexual|sexbot\w*|lewd\w*|erotic\w*|eroge"
    r"|ecchi|oppai|nude|nudes|nudity|naked|milfs?|futa\w*|ahegao|boob\w*|busty|horny"
    r"|stripteas\w*|strippers?|strip poker|r18|18\+)(?!\w)",
    re.IGNORECASE,
)


ADULT_TAGS = frozenset({"NSFW", "Hentai"})
ADULT_TOP_TAGS = frozenset({"Sexual Content", "Nudity"})
TOP_TAGS = 5


def is_adult(name: str, genres: Iterable[str] = ()) -> bool:
    return bool(ADULT_NAME.search(name)) or not ADULT_GENRES.isdisjoint(genres)


def adult_mask(games: Games) -> np.ndarray:
    """bool per catalog row."""
    genres = games.catalog.items.genres
    adult_ids = [i for i, name in enumerate(games.genre_names) if name in ADULT_GENRES]
    owner = np.repeat(np.arange(len(genres)), np.diff(genres.offsets))
    mask = np.zeros(len(games), dtype=bool)
    mask[owner[np.isin(genres.values, adult_ids)]] = True
    tags = games.catalog.items.tags
    lengths = np.diff(tags.offsets)
    owner = np.repeat(np.arange(len(tags)), lengths)
    rank = np.arange(len(tags.values)) - np.repeat(tags.offsets[:-1], lengths)
    anywhere = [i for i, name in enumerate(games.tag_names) if name in ADULT_TAGS]
    top = [i for i, name in enumerate(games.tag_names) if name in ADULT_TOP_TAGS]
    hit = np.isin(tags.values, anywhere) | (np.isin(tags.values, top) & (rank < TOP_TAGS))
    mask[owner[hit]] = True
    mask |= np.fromiter((bool(ADULT_NAME.search(str(n))) for n in games.name), bool, len(games))
    return mask
