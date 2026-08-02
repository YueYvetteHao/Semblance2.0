"""
Per-user daily quota — the SEAM for "N free queries/day/user".

DISABLED by default (QUOTA_ENABLED=false) so the MVP needs no auth. To turn on:
  - Put managed auth (e.g. Supabase) in front; the SPA sends Authorization: Bearer <jwt>.
  - Verify the JWT here (Supabase JWKS) -> user_id.
  - Atomically count per (user_id, UTC day) in a free DB (Supabase Postgres / Cloudflare D1):
      INSERT ... ON CONFLICT (user_id, day) DO UPDATE SET count = count + 1 RETURNING count
    -> reject with 429 when count > FREE_DAILY_LIMIT.
The same counter underpins a paid tier: a payment webhook just raises the per-user limit.
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException

QUOTA_ENABLED = os.environ.get("QUOTA_ENABLED", "false").lower() == "true"
FREE_DAILY_LIMIT = int(os.environ.get("FREE_DAILY_LIMIT", "3"))


async def quota_dependency(authorization: str | None = Header(default=None)):
    """No-op while QUOTA_ENABLED is false. Returns the user id (or None) for the handler."""
    if not QUOTA_ENABLED:
        return None
    # --- wire this when enabling -------------------------------------------------
    # user_id = verify_jwt(authorization)     # Supabase/Clerk JWKS
    # count = consume(user_id)                # atomic increment in your DB
    # if count > FREE_DAILY_LIMIT:
    #     raise HTTPException(429, f"Daily free limit ({FREE_DAILY_LIMIT}) reached. Try again tomorrow.")
    # return user_id
    raise HTTPException(501, "QUOTA_ENABLED=true but the auth/DB backend is not wired yet (see quota.py).")
