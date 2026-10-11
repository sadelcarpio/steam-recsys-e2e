"""Pseudonymous user ids (spec 13): the only user key stored anywhere.

`user_id = HMAC-SHA256(key, str(steamid))[:8] >> 1`, a 63-bit non-negative integer: a bigint in
every schema, stable across runs, and not reversible without the key (a plain hash of the ~2^32
real SteamID64s could be enumerated). The key lives in Secrets Manager
(`data-ingestion/user-id-hmac-key`) and must never be rotated: a new key splits every user's
history. Lambda-safe (stdlib + boto3 only).
"""

from __future__ import annotations

import hashlib
import hmac

import boto3

from steam_ingestion.config import IngestionSettings

USER_ID_BITS = 63


def hash_user_id(steamid: int, key: bytes) -> int:
    digest = hmac.new(key, str(steamid).encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") >> (64 - USER_ID_BITS)


def resolve_user_id_key(settings: IngestionSettings) -> bytes:
    """The HMAC key: the local / test `USER_ID_HMAC_KEY`, else the secret. Never logged."""
    if settings.user_id_hmac_key is not None:
        key = settings.user_id_hmac_key.get_secret_value()
    else:
        key = boto3.client("secretsmanager").get_secret_value(
            SecretId=settings.user_id_key_secret_id
        )["SecretString"]
    if len(key) < 32:
        raise ValueError("the user id HMAC key must have at least 32 characters")
    return key.encode()
