-- The ledger keeps one review per (user, game), so the reviewed game can only be in its own
-- history if the ASOF join let the review see itself.
select review_id, game_idx, games_reviewed_positive
from {{ ref('interactions') }}
where contains(games_reviewed_positive, game_idx)
