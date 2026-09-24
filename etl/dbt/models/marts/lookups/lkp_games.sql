-- Dense game index for embedding tables (Steam appids are sparse, up to ~3.5M).
{{ dense_id_lookup("select game_id as value from " ~ ref('int_games__deduplicated'), 'game_idx', 'game_id') }}
