import asyncio
import time
import pytest
from shamu.protocols import PermissionDeniedError
from shamu.security.tool_registry import PermissionTier
from shamu.security.approval_manager import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalToken,
    TokenSecurityError,
)


@pytest.mark.asyncio
async def test_approval_manager_no_handler_fails_closed():
    manager = ApprovalManager()
    with pytest.raises(PermissionDeniedError, match="No approval handler registered"):
        await manager.request_approval(
            tool_name="write_file",
            arguments={"path": "notes.txt", "content": "hello"},
            tier=PermissionTier.TIER_2_MUTATION,
            reason="Writing notes",
        )


@pytest.mark.asyncio
async def test_interactive_approval_success():
    manager = ApprovalManager()

    async def mock_handler(req: ApprovalRequest) -> ApprovalResponse:
        assert req.tool_name == "write_file"
        assert req.arguments == {"path": "notes.txt"}
        return ApprovalResponse(request_id=req.request_id, approved=True)

    manager.register_handler(mock_handler)

    token = await manager.request_approval(
        tool_name="write_file",
        arguments={"path": "notes.txt"},
        tier=PermissionTier.TIER_2_MUTATION,
        reason="Writing notes",
        timeout_sec=5.0,
    )

    assert isinstance(token, ApprovalToken)
    assert token.tool_name == "write_file"
    assert token.canonical_argument_hash is not None
    assert token.signature is not None

    # Redeem token successfully once
    redeemed = manager.verify_and_redeem_token(
        token=token,
        tool_name="write_file",
        arguments={"path": "notes.txt"},
    )
    assert redeemed is True


@pytest.mark.asyncio
async def test_interactive_approval_denied():
    manager = ApprovalManager()

    async def mock_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            request_id=req.request_id,
            approved=False,
            reason="User denied operation",
        )

    manager.register_handler(mock_handler)

    with pytest.raises(PermissionDeniedError, match="User denied operation"):
        await manager.request_approval(
            tool_name="write_file",
            arguments={"path": "notes.txt"},
            tier=PermissionTier.TIER_2_MUTATION,
        )


@pytest.mark.asyncio
async def test_approval_timeout_fails_closed():
    manager = ApprovalManager()

    async def slow_handler(req: ApprovalRequest) -> ApprovalResponse:
        await asyncio.sleep(0.5)
        return ApprovalResponse(request_id=req.request_id, approved=True)

    manager.register_handler(slow_handler)

    with pytest.raises(PermissionDeniedError, match="Approval timed out"):
        await manager.request_approval(
            tool_name="powershell_exec",
            arguments={"command": "dir"},
            tier=PermissionTier.TIER_3_HIGH_IMPACT,
            timeout_sec=0.1,
        )


@pytest.mark.asyncio
async def test_single_use_token_replay_prevention():
    manager = ApprovalManager()

    async def mock_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(request_id=req.request_id, approved=True)

    manager.register_handler(mock_handler)

    token = await manager.request_approval(
        tool_name="write_file",
        arguments={"path": "notes.txt"},
        tier=PermissionTier.TIER_2_MUTATION,
    )

    # First redemption succeeds
    assert manager.verify_and_redeem_token(token, "write_file", {"path": "notes.txt"}) is True

    # Second redemption must raise TokenSecurityError
    with pytest.raises(TokenSecurityError, match="already been redeemed"):
        manager.verify_and_redeem_token(token, "write_file", {"path": "notes.txt"})


@pytest.mark.asyncio
async def test_argument_tampering_rejected():
    manager = ApprovalManager()

    async def mock_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(request_id=req.request_id, approved=True)

    manager.register_handler(mock_handler)

    token = await manager.request_approval(
        tool_name="write_file",
        arguments={"path": "safe.txt", "content": "ok"},
        tier=PermissionTier.TIER_2_MUTATION,
    )

    # Attempt to use the token with altered arguments
    with pytest.raises(TokenSecurityError, match="tampering detected"):
        manager.verify_and_redeem_token(
            token,
            "write_file",
            {"path": "malicious.bat", "content": "format c:"},
        )


@pytest.mark.asyncio
async def test_token_tool_mismatch_rejected():
    manager = ApprovalManager()

    async def mock_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(request_id=req.request_id, approved=True)

    manager.register_handler(mock_handler)

    token = await manager.request_approval(
        tool_name="write_file",
        arguments={"path": "notes.txt"},
        tier=PermissionTier.TIER_2_MUTATION,
    )

    with pytest.raises(TokenSecurityError, match="tool mismatch"):
        manager.verify_and_redeem_token(token, "delete_file", {"path": "notes.txt"})


@pytest.mark.asyncio
async def test_token_expiration_rejected():
    manager = ApprovalManager()

    async def mock_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(request_id=req.request_id, approved=True)

    manager.register_handler(mock_handler)

    token = await manager.request_approval(
        tool_name="write_file",
        arguments={"path": "notes.txt"},
        tier=PermissionTier.TIER_2_MUTATION,
        timeout_sec=0.01,
    )

    # Wait for expiration
    await asyncio.sleep(0.05)

    with pytest.raises(TokenSecurityError, match="expired"):
        manager.verify_and_redeem_token(token, "write_file", {"path": "notes.txt"})


@pytest.mark.asyncio
async def test_token_signature_tampering_rejected():
    manager = ApprovalManager()

    async def mock_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(request_id=req.request_id, approved=True)

    manager.register_handler(mock_handler)

    token = await manager.request_approval(
        tool_name="write_file",
        arguments={"path": "notes.txt"},
        tier=PermissionTier.TIER_2_MUTATION,
    )

    tampered_token = ApprovalToken(
        token_id=token.token_id,
        request_id=token.request_id,
        tool_name=token.tool_name,
        canonical_argument_hash=token.canonical_argument_hash,
        expires_at=token.expires_at,
        signature="deadbeef" * 8,
    )

    with pytest.raises(TokenSecurityError, match="Invalid token signature"):
        manager.verify_and_redeem_token(tampered_token, "write_file", {"path": "notes.txt"})


@pytest.mark.asyncio
async def test_tier_3_double_confirmation_required():
    manager = ApprovalManager()

    # Case A: Approved, but double_confirmed is False
    async def single_confirm_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            request_id=req.request_id,
            approved=True,
            double_confirmed=False,
        )

    manager.register_handler(single_confirm_handler)

    with pytest.raises(PermissionDeniedError, match="requires double confirmation"):
        await manager.request_approval(
            tool_name="powershell_exec",
            arguments={"command": "rmdir /s /q test"},
            tier=PermissionTier.TIER_3_HIGH_IMPACT,
            requires_double_confirmation=True,
        )

    # Case B: Approved with double_confirmed = True
    async def double_confirm_handler(req: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            request_id=req.request_id,
            approved=True,
            double_confirmed=True,
        )

    manager.register_handler(double_confirm_handler)

    token = await manager.request_approval(
        tool_name="powershell_exec",
        arguments={"command": "rmdir /s /q test"},
        tier=PermissionTier.TIER_3_HIGH_IMPACT,
        requires_double_confirmation=True,
    )
    assert isinstance(token, ApprovalToken)
    assert manager.verify_and_redeem_token(
        token,
        "powershell_exec",
        {"command": "rmdir /s /q test"},
    ) is True


def test_approval_manager_evicts_expired_tokens():
    """NEW-SEC-01: ApprovalManager must evict expired tokens to prevent unbounded memory growth."""
    manager = ApprovalManager()

    # Pre-populate with an expired redeemed token
    manager._redeemed_tokens["old-token-1"] = time.time() - 10.0
    manager._redeemed_tokens["old-token-2"] = time.time() - 5.0
    # And an active redeemed token
    manager._redeemed_tokens["active-token"] = time.time() + 60.0

    assert len(manager._redeemed_tokens) == 3

    # Manually invoke eviction
    manager._evict_expired_tokens()

    assert "old-token-1" not in manager._redeemed_tokens
    assert "old-token-2" not in manager._redeemed_tokens
    assert "active-token" in manager._redeemed_tokens
    assert len(manager._redeemed_tokens) == 1

