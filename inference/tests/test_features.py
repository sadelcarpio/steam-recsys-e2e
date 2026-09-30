import numpy as np
import pyarrow as pa
import pytest
from conftest import FIRST_GAME, N_GAMES, REVIEWS, FakeSource, add_tags, review_time

from steam_inference import features
from steam_inference.features import load_inference_data, truncate


def test_users_are_their_latest_features(source):
    data = load_inference_data(source)
    users = data.users
    assert users.user_id.tolist() == [101, 102, 103, 104]  # 105 has no user_features row
    u101 = users.user_id.tolist().index(101)
    assert users.history[u101].tolist() == [7, 6, 5, 3, 2]  # most recent positives first
    u103 = users.user_id.tolist().index(103)
    assert users.history[u103].tolist() == [2, 0, 0, 0, 0]


def test_reviews_are_grouped_per_user_newest_first(source):
    data = load_inference_data(source)
    users, reviews = data.users, data.reviews
    for i, user in enumerate(users.user_id):
        start, count = users.review_start[i], users.review_count[i]
        expected = list(reversed(REVIEWS[int(user)]))
        assert reviews.game_idx[start : start + count].tolist() == [g for g, _ in expected]
    reviewed = reviews.of(users)
    assert reviewed.row(0) == [g for g, _ in reversed(REVIEWS[101])]


def test_catalog_has_every_current_game_with_latest_features(source):
    games = load_inference_data(source).games
    assert len(games) == N_GAMES  # including the game newer than the model
    assert np.allclose(games.catalog.items.reviews_ratio, 0.9)  # latest row, not 1970
    row = int(games.catalog.rows(np.array([FIRST_GAME + 4]))[0])
    assert games.game_id[row] == 1000 + FIRST_GAME + 4
    assert games.name[row] == f"Game {FIRST_GAME + 4}"


def test_describe_uses_lookup_names(source):
    games = load_inference_data(source).games
    row = int(games.catalog.rows(np.array([6]))[0])  # genre 2 + 6 % 4 = 4, developer 2 + 0
    text = games.describe(row)
    assert text.startswith("Game 6 | Genre 2 | by Studio 0 | free to play")
    assert text.endswith("90% positive reviews")


def test_max_users_keeps_the_most_active(source):
    users = load_inference_data(source, max_users=2).users
    # 104 has 18 reviews, 101 has 7; kept in user id order
    assert users.user_id.tolist() == [101, 104]
    assert users.review_count.tolist() == [7, 18]


def test_streamed_reduction_matches_single_pass(monkeypatch, marts):
    once = load_inference_data(FakeSource(marts, batch_rows=10_000)).users
    monkeypatch.setattr(features, "REDUCE_ROWS", 1)  # reduce after every batch
    streamed = load_inference_data(FakeSource(marts, batch_rows=2)).users
    assert np.array_equal(once.user_id, streamed.user_id)
    assert np.array_equal(once.history, streamed.history)


def test_null_history_is_padded(marts):
    table = marts["user_features"]
    column = table["games_reviewed_positive"].to_pylist()
    column[-1] = None
    marts["user_features"] = table.set_column(
        2, "games_reviewed_positive", pa.array(column, pa.list_(pa.int64()))
    )
    users = load_inference_data(FakeSource(marts)).users
    assert users.history.shape[1] == 5


def test_popular_counts_are_recent_positive_reviews_per_game():
    day = 86_400 * 1_000_000
    game_idx = np.array([2, 2, 3, 3, 4, 5])
    micros = np.array([100, 99, 100, 100, 100, 50]) * day
    positive = np.array([True, True, True, False, True, True])
    counts = features.popular_counts(game_idx, micros, positive, window_days=10)
    assert counts.tolist() == [0, 0, 2, 1, 1, 0]  # game 3's negative, game 5 too old


def test_popular_counts_from_interactions(source):
    counts = load_inference_data(source).popular_counts
    assert counts[FIRST_GAME] == 3  # users 101, 103, 104
    assert counts[4] == 1  # 101's review is negative
    assert len(counts) == FIRST_GAME + N_GAMES - 2  # the last 2 games: nobody reviewed them


def test_describe_appends_the_short_description(source):
    games = load_inference_data(source, description_chars=200).games
    odd, even = (int(games.catalog.rows(np.array([g]))[0]) for g in (7, 6))
    # scraped as " Shoot &amp; <b>loot</b> 7. ": cleaned before it reaches the prompt
    assert games.describe(odd).endswith("positive reviews | Shoot & loot 7.")
    assert games.describe(even).endswith("positive reviews")  # no description scraped


def test_descriptions_are_off_by_default_and_optional(source, marts):
    assert not load_inference_data(source).games.description.any()
    without = {k: v for k, v in marts.items() if k != "game_details"}
    games = load_inference_data(FakeSource(without), description_chars=200).games
    assert not games.description.any()  # a missing mart is not an error


def test_truncate_cuts_at_a_word_boundary():
    text = "A roguelike deck builder where you climb a spire of monsters and relics"
    assert truncate(text, 200) == text
    short = truncate(text, 30)
    assert short == "A roguelike deck builder…" and len(short) <= 30
    assert truncate("x" * 50, 10) == "x" * 9 + "…"  # no space to cut at


def test_games_get_their_tags_when_the_marts_exist(marts):
    games = load_inference_data(FakeSource(add_tags(marts))).games
    items = games.catalog.items
    first = int(np.flatnonzero(items.game_idx == FIRST_GAME)[0])
    last = int(np.flatnonzero(items.game_idx == FIRST_GAME + N_GAMES - 1)[0])
    assert items.tags.row(first) == [FIRST_GAME + 2, FIRST_GAME + 3]
    assert items.tags.weights[items.tags.offsets[first] : items.tags.offsets[first + 1]].tolist() \
        == pytest.approx([2 / 3, 1 / 3])  # fmt: skip
    assert items.tags.row(last) == []
    assert "tags: Tag 2, Tag 3" in games.describe(first)
    assert "tags:" not in games.describe(last)


def test_tags_are_optional_unless_the_model_needs_them(source):
    games = load_inference_data(source).games
    assert len(games.catalog.items.tags.values) == 0 and len(games.tag_names) == 0
    assert "tags:" not in games.describe(0)
    with pytest.raises(RuntimeError, match="tags"):
        load_inference_data(source, require_tags=True)


def test_max_users_loads_only_the_kept_users_reviews(source):
    full = load_inference_data(source)
    kept = load_inference_data(source, max_users=2)
    assert len(kept.reviews.game_idx) == 7 + 18  # 101 and 104 only
    for i, user in enumerate(kept.users.user_id):
        start, count = kept.users.review_start[i], kept.users.review_count[i]
        expected = [g for g, _ in reversed(REVIEWS[int(user)])]
        assert kept.reviews.game_idx[start : start + count].tolist() == expected
    # popularity still counts every user's reviews
    assert np.array_equal(kept.popular_counts, full.popular_counts)


def test_activity_counts_every_user(source):
    activity = features.review_activity(source, 7)
    assert activity.user_id.tolist() == [101, 102, 103, 104, 105]
    assert activity.reviews.tolist() == [len(REVIEWS[u]) for u in range(101, 106)]
    assert activity.positives.tolist() == [sum(p for _, p in REVIEWS[u]) for u in range(101, 106)]
    assert activity.most_active().tolist() == [101, 102, 103, 104]  # 105: negatives only
    assert activity.most_active(1).tolist() == [104]


def test_streamed_activity_matches_single_pass(monkeypatch, marts):
    once = features.review_activity(FakeSource(marts, batch_rows=10_000), 7)
    monkeypatch.setattr(features, "REDUCE_ROWS", 1)  # reduce after every batch
    streamed = features.review_activity(FakeSource(marts, batch_rows=2), 7)
    assert np.array_equal(once.user_id, streamed.user_id)
    assert np.array_equal(once.reviews, streamed.reviews)
    assert np.array_equal(once.positives, streamed.positives)
    assert once.newest == streamed.newest


def test_as_of_reads_ignore_later_rows(source):
    # 101's reviews are hours 0..6 of its day: keep the first 3 (2+, 3+, 4-)
    cutoff = features._micros(pa.array([review_time(101, 2)]))[0]
    activity = features.review_activity(source, 7, until=cutoff)
    u101 = activity.user_id.tolist().index(101)
    assert (activity.reviews[u101], activity.positives[u101]) == (3, 2)
    users = features.latest_users(
        source.batches("user_features", ["user_id", "timestamp", "games_reviewed_positive"], 7),
        keep=np.array([101]),
        until=cutoff,
    )
    assert users.user_id.tolist() == [101]
    assert users.history[0].tolist() == [3, 2, 0, 0, 0]


def test_member():
    ids = np.array([3, 5, 9])
    assert features.member(np.array([1, 3, 4, 9, 10]), ids).tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert not features.member(np.array([1]), np.array([], dtype=np.int64)).any()
