"""Adult-content filter: games excluded from everything the pipeline publishes (recommendations,
the popularity fallback, the online catalog and the frontend's search index).

The exact signal is Steam's content descriptor 3, "Adult Only Sexual Content" (spec 19): the
developer's own answer to Steam's mature content survey, what Steam hides from its store by
default. It separates mature mainstream games (Cyberpunk 2077, GTA V: 1 some nudity or sexual
content, 2 violence, 5 general mature content) from adult-only ones (HuniePop, Mirror: 1, 3, 4,
5). The tags scraper keeps the descriptors (`game_tags` mart); a game scraped before that has
none, so the heuristics below stay as a union with it.

`required_age` is rarely set and mostly marks violence, and the "Sexual Content" / "Nudity"
genres are set by few developers. Explicit games usually say so in their name, so a game is also
adult when it has one of those genres or its name has one of the unambiguous words below.
Deliberately conservative: a false positive hides one game, a false negative shows explicit
art in a demo.

Steam user tags (spec 8) add the players' votes, weighed by how much of them the adult tags
(`ADULT_TAGS`) take: a game is adult when they hold at least `ADULT_TAG_SHARE` of its tag votes,
or at least `ADULT_TOP_TAG_SHARE` with an adult tag being its most voted one. Their mere
presence or rank is not enough (2026-11-18, real tags): mature mainstream games carry them high
(Cyberpunk 2077 "Nudity" 3rd, GTA V "Sexual Content" 2nd, at 7-16% of the votes) and players
troll popular games with them (Marvel Rivals and Helltaker "Hentai", 17th-18th), while
adult-only games give them 20-66% (HuniePop, NEKOPARA, Mirror). The vote share also catches
games whose descriptors understate them (NEKOPARA Vol. 1 declares 1 and 5, no 3).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from steam_inference.features import Games

# Steam content descriptor "Adult Only Sexual Content" (steam_ingestion.steam_api.StoreItem)
ADULT_DESCRIPTOR = 3
ADULT_GENRES = frozenset({"Sexual Content", "Nudity"})
ADULT_NAME = re.compile(
    r"\b(?:hentai\w*|nsfw|porn\w*|sex|sexy\w*|sexual|sexbot\w*|lewd\w*|erotic\w*|eroge"
    r"|ecchi|oppai|nude|nudes|nudity|naked|milfs?|futa\w*|ahegao|boob\w*|busty|horny"
    r"|stripteas\w*|strippers?|strip poker|r18|18\+)(?!\w)",
    re.IGNORECASE,
)


ADULT_TAGS = frozenset({"NSFW", "Hentai", "Sexual Content", "Nudity"})
ADULT_TAG_SHARE = 0.17
ADULT_TOP_TAG_SHARE = 0.10


def is_adult(name: str, genres: Iterable[str] = ()) -> bool:
    return bool(ADULT_NAME.search(name)) or not ADULT_GENRES.isdisjoint(genres)


def adult_mask(games: Games) -> np.ndarray:
    """bool per catalog row."""
    genres = games.catalog.items.genres
    adult_ids = [i for i, name in enumerate(games.genre_names) if name in ADULT_GENRES]
    owner = np.repeat(np.arange(len(genres)), np.diff(genres.offsets))
    mask = np.zeros(len(games), dtype=bool)
    mask[owner[np.isin(genres.values, adult_ids)]] = True
    mask |= adult_by_tag_votes(games)
    mask |= games.has_descriptor(ADULT_DESCRIPTOR)
    mask |= np.fromiter((bool(ADULT_NAME.search(str(n))) for n in games.name), bool, len(games))
    return mask


def adult_by_tag_votes(games: Games) -> np.ndarray:
    """bool per catalog row: the adult tags' share of the game's tag votes (weights normalized per
    game; uniform when a game has none)."""
    tags = games.catalog.items.tags
    rows = len(tags)
    lengths = np.diff(tags.offsets)
    if len(tags.values) == 0:
        return np.zeros(rows, dtype=bool)
    owner = np.repeat(np.arange(rows), lengths)
    weights = (
        tags.weights.astype(np.float64)
        if tags.weights is not None
        else 1.0 / np.repeat(np.maximum(lengths, 1), lengths)
    )
    adult_ids = [i for i, name in enumerate(games.tag_names) if name in ADULT_TAGS]
    is_adult_tag = np.isin(tags.values, adult_ids)
    total = np.bincount(owner, weights=weights, minlength=rows)
    share = np.bincount(owner, weights=weights * is_adult_tag, minlength=rows)
    share = np.divide(share, total, out=np.zeros(rows), where=total > 0)
    top = np.zeros(rows)
    np.maximum.at(top, owner, weights)
    top_adult = np.zeros(rows)
    np.maximum.at(top_adult, owner, np.where(is_adult_tag, weights, 0.0))
    adult_on_top = (top_adult > 0) & (top_adult >= top)
    return (share >= ADULT_TAG_SHARE) | ((share >= ADULT_TOP_TAG_SHARE) & adult_on_top)
