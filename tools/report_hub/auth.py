# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted-proxy identity and per-user mutation tokens."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from typing import Mapping

from .config import Settings
from .domain import PermissionDeniedError, ReportHubError, require_text


ROLE_RANK = {"viewer": 0, "qa": 1, "admin": 2}


class AuthenticationError(ReportHubError):
    status = 401
    code = "authentication_required"


@dataclass(frozen=True)
class Identity:
    user: str
    role: str


class AuthManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def authenticate(self, headers: Mapping[str, str]) -> Identity:
        if self.settings.auth_mode == "development":
            return Identity(self.settings.dev_user, self.settings.dev_role)
        user = headers.get(self.settings.auth_user_header, "").strip()
        role = headers.get(self.settings.auth_role_header, "").strip().lower()
        if not user:
            raise AuthenticationError("authenticated proxy identity is missing")
        require_text(user, "user", maximum=160)
        if role not in ROLE_RANK:
            raise AuthenticationError("authenticated proxy role is missing or unsupported")
        return Identity(user, role)

    def require_role(self, identity: Identity, minimum: str) -> None:
        if ROLE_RANK.get(identity.role, -1) < ROLE_RANK[minimum]:
            raise PermissionDeniedError(f"{minimum} role is required")

    def csrf_token(self, identity: Identity) -> str:
        payload = f"report-hub/v1\0{identity.user}\0{identity.role}".encode("utf-8")
        return hmac.new(self.settings.secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    def verify_mutation(self, identity: Identity, headers: Mapping[str, str]) -> None:
        supplied = headers.get("X-Report-Hub-CSRF", "")
        if not supplied or not hmac.compare_digest(supplied, self.csrf_token(identity)):
            raise PermissionDeniedError("valid Report Hub mutation token is required")
