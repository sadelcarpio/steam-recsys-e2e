-- user_idx is exactly 1..count(*) (with `unique`: no gap, no duplicate), so serving can map
-- user_idx to a byte offset of the published index.
select count(*) as users, min(user_idx) as first_idx, max(user_idx) as last_idx
from {{ ref('user_index') }}
having count(*) > 0 and (min(user_idx) != 1 or max(user_idx) != count(*))
