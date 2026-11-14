"""Asynchronous Tiered Approval Engine with single-use cryptographic tokens."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from shamu.protocols import PermissionDeniedError
from shamu.security.tool_registry import PermissionTier


class TokenSecurityError(PermissionDeniedError):
    """Raised when an approval token is invalid, expired, tampered, or replayed."""
    pass


def canonicalize_arguments(args: Dict[str, Any]) -> str:
    """Return deterministic canonical JSON serialization of arguments."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def compute_argument_hash(args: Dict[str, Any]) -> str:
    """Compute SHA-256 digest of canonicalized arguments."""
    canonical = canonicalize_arguments(args)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ApprovalRequest:
    """Represents a request for interactive approval."""
    request_id: str
    tool_name: str
    arguments: Dict[str, Any]
    canonical_argument_hash: str
    tier: PermissionTier
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    requires_double_confirmation: bool = False

    def __post_init__(self):
        if not self.canonical_argument_hash:
            self.canonical_argument_hash = compute_argument_hash(self.arguments)


@dataclass
class ApprovalResponse:
    """Response returned by an approval handler (UI modal or CLI)."""
    request_id: str
    approved: bool
    double_confirmed: bool = False
    reason: Optional[str] = None


@dataclass
class ApprovalToken:
    """Single-use cryptographic token granting execution authorization."""
    token_id: str
    request_id: str
    tool_name: str
    canonical_argument_hash: str
    expires_at: float
    signature: str


ApprovalHandler = Callable[[ApprovalRequest], Awaitable[ApprovalResponse]]


class ApprovalManager:
    """Manages asynchronous interactive human approvals and issues single-use signed tokens."""

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        default_timeout: float = 30.0,
    ) -> None:
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.default_timeout = default_timeout
        self.handler: Optional[ApprovalHandler] = None
        self._redeemed_tokens: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _evict_expired_tokens(self) -> None:
        """Evicts redeemed tokens whose expiration timestamp has passed to prevent memory growth."""
        now = time.time()
        expired_ids = [tid for tid, exp in self._redeemed_tokens.items() if exp < now]
        for tid in expired_ids:
            del self._redeemed_tokens[tid]

    def register_handler(self, handler: ApprovalHandler) -> None:
        """Register the interactive approval handler callback."""
        self.handler = handler

    def _sign_token(
        self,
        token_id: str,
        request_id: str,
        tool_name: str,
        arg_hash: str,
        expires_at: float,
    ) -> str:
        payload = f"{token_id}:{request_id}:{tool_name}:{arg_hash}:{expires_at}".encode("utf-8")
        return hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()

    async def request_approval(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tier: PermissionTier,
        reason: str = "",
        timeout_sec: Optional[float] = None,
        requires_double_confirmation: bool = False,
    ) -> ApprovalToken:
        """Request human approval for a high-tier action. Fails closed on any rejection or timeout."""
        if self.handler is None:
            raise PermissionDeniedError("No approval handler registered.")

        timeout = timeout_sec if timeout_sec is not None else self.default_timeout
        now = time.time()
        expires_at = now + timeout
        arg_hash = compute_argument_hash(arguments)
        req_id = f"appr-{uuid.uuid4()}"

        request = ApprovalRequest(
            request_id=req_id,
            tool_name=tool_name,
            arguments=arguments,
            canonical_argument_hash=arg_hash,
            tier=tier,
            reason=reason,
            created_at=now,
            expires_at=expires_at,
            requires_double_confirmation=requires_double_confirmation,
        )

        try:
            response: ApprovalResponse = await asyncio.wait_for(
                self.handler(request),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise PermissionDeniedError(
                f"Approval timed out after {timeout:.1f}s (request={req_id})."
            ) from exc
        except Exception as exc:
            raise PermissionDeniedError(
                f"Approval handler failed: {exc} (request={req_id})."
            ) from exc

        if not response.approved:
            reason_str = response.reason or "Rejected by user"
            raise PermissionDeniedError(f"Approval denied: {reason_str}")

        if requires_double_confirmation and not response.double_confirmed:
            raise PermissionDeniedError(
                "High-impact action requires double confirmation, which was not granted."
            )

        token_id = str(uuid.uuid4())
        signature = self._sign_token(
            token_id=token_id,
            request_id=req_id,
            tool_name=tool_name,
            arg_hash=arg_hash,
            expires_at=expires_at,
        )

        return ApprovalToken(
            token_id=token_id,
            request_id=req_id,
            tool_name=tool_name,
            canonical_argument_hash=arg_hash,
            expires_at=expires_at,
            signature=signature,
        )

    def verify_and_redeem_token(
        self,
        token: ApprovalToken,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> bool:
        """Verify token authenticity and redeem it atomically. Single-use only."""
        # 1. Expiration check
        if time.time() > token.expires_at:
            raise TokenSecurityError(f"Token {token.token_id} has expired.")

        # 2. Tool name match
        if token.tool_name != tool_name:
            raise TokenSecurityError(
                f"Token tool mismatch: token was issued for '{token.tool_name}', got '{tool_name}'."
            )

        # 3. Argument hash check
        current_hash = compute_argument_hash(arguments)
        if not hmac.compare_digest(token.canonical_argument_hash, current_hash):
            raise TokenSecurityError(
                f"Token argument tampering detected for tool '{tool_name}'."
            )

        # 4. Signature verification
        expected_sig = self._sign_token(
            token_id=token.token_id,
            request_id=token.request_id,
            tool_name=token.tool_name,
            arg_hash=token.canonical_argument_hash,
            expires_at=token.expires_at,
        )
        if not hmac.compare_digest(token.signature, expected_sig):
            raise TokenSecurityError("Invalid token signature.")

        # 5. Atomic single-use check and expired token eviction
        with self._lock:
            self._evict_expired_tokens()
            if token.token_id in self._redeemed_tokens:
                raise TokenSecurityError(
                    f"Token {token.token_id} has already been redeemed (replay attack prevented)."
                )
            self._redeemed_tokens[token.token_id] = token.expires_at

        return True

