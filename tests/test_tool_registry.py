import pytest
from shamu.security.tool_registry import (
    ToolRegistry,
    ToolCapability,
    PermissionTier,
    PolicyOutcome,
)


def test_tool_registry_registration_and_lookup():
    reg = ToolRegistry()
    cap = ToolCapability(
        name="custom_read",
        tier=PermissionTier.TIER_1_READ_ONLY,
        description="Reads safe files",
        parameters_schema={
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        },
        path_arg_names=["path"],
    )
    reg.register(cap)
    retrieved = reg.get("custom_read")
    assert retrieved is not None
    assert retrieved.name == "custom_read"
    assert retrieved.tier == PermissionTier.TIER_1_READ_ONLY


def test_tool_registry_validates_valid_arguments():
    reg = ToolRegistry()
    valid, err = reg.validate_call("read_file", {"path": "workspace/doc.txt"})
    assert valid is True
    assert err is None


def test_tool_registry_rejects_missing_required_arguments():
    reg = ToolRegistry()
    # write_file requires both 'path' and 'content'
    valid, err = reg.validate_call("write_file", {"path": "workspace/doc.txt"})
    assert valid is False
    assert "content" in err.lower()


def test_tool_registry_rejects_extra_undeclared_arguments():
    reg = ToolRegistry()
    # read_file only accepts 'path'; extra 'admin_override' must be rejected
    valid, err = reg.validate_call("read_file", {"path": "workspace/doc.txt", "admin_override": True})
    assert valid is False
    assert "additional" in err.lower() or "unexpected" in err.lower()


def test_tool_registry_rejects_invalid_types():
    reg = ToolRegistry()
    # path must be a string, not an integer or dict
    valid, err = reg.validate_call("read_file", {"path": 12345})
    assert valid is False
    assert "string" in err.lower() or "type" in err.lower()


def test_tool_registry_unregistered_tool_default_deny():
    reg = ToolRegistry()
    valid, err = reg.validate_call("unknown_agent_hack_tool", {"arg": "val"})
    assert valid is False
    assert "unregistered" in err.lower()


def test_tool_registry_prohibited_tools_hard_deny():
    reg = ToolRegistry()
    # format_disk is permanently prohibited
    valid, err = reg.validate_call("format_disk", {"drive": "C:"})
    assert valid is False
    assert "prohibited" in err.lower()

    cap = reg.get("format_disk")
    assert cap is not None
    assert cap.is_prohibited is True
    assert cap.tier == PermissionTier.PROHIBITED


def test_builtin_capabilities_populated():
    reg = ToolRegistry()
    read_cap = reg.get("read_file")
    assert read_cap is not None
    assert read_cap.tier == PermissionTier.TIER_1_READ_ONLY

    write_cap = reg.get("write_file")
    assert write_cap is not None
    assert write_cap.tier == PermissionTier.TIER_2_MUTATION

    ps_cap = reg.get("powershell_exec")
    assert ps_cap is not None
    assert ps_cap.tier == PermissionTier.TIER_3_HIGH_IMPACT
    assert ps_cap.requires_double_confirmation is True


def test_get_tool_definitions_filters_prohibited():
    reg = ToolRegistry()
    tool_defs = reg.get_tool_definitions()
    names = [td.name for td in tool_defs]

    # Authorized tools must be present
    assert "read_file" in names
    assert "write_file" in names
    assert "powershell_exec" in names

    # Prohibited tools must never be converted to ToolDefinitions
    assert "format_disk" not in names
    assert "reg_delete" not in names
    assert "disable_firewall" not in names
    assert "elevate_uac" not in names


def test_validate_call_rejects_non_dict_arguments():
    reg = ToolRegistry()
    for bad_args in [None, "invalid_str", [1, 2, 3], 42, 3.14, True]:
        valid, err = reg.validate_call("read_file", bad_args)
        assert valid is False
        assert "dictionary" in err.lower() or "dict" in err.lower()


def test_synchronized_builtin_tools_match_firewall_tiers():
    from shamu.security.firewall import TIER_1_TOOLS, TIER_2_TOOLS, TIER_3_TOOLS

    reg = ToolRegistry()
    expected_all_builtins = {
        "read_file",
        "write_file",
        "append_file",
        "list_directory",
        "get_system_metrics",
        "search_local_history",
        "focus_application",
        "powershell_exec",
        "delete_file",
        "kill_process",
    }

    # Verify disjointness
    assert TIER_1_TOOLS.isdisjoint(TIER_2_TOOLS)
    assert TIER_2_TOOLS.isdisjoint(TIER_3_TOOLS)
    assert TIER_1_TOOLS.isdisjoint(TIER_3_TOOLS)

    # Union must exactly equal the 10 built-in tools
    combined = TIER_1_TOOLS | TIER_2_TOOLS | TIER_3_TOOLS
    assert combined == expected_all_builtins

    # Verify each tool is properly registered with correct tier in ToolRegistry
    for name in TIER_1_TOOLS:
        cap = reg.get(name)
        assert cap is not None, f"Tool {name} missing from ToolRegistry"
        assert cap.tier == PermissionTier.TIER_1_READ_ONLY

    for name in TIER_2_TOOLS:
        cap = reg.get(name)
        assert cap is not None, f"Tool {name} missing from ToolRegistry"
        assert cap.tier == PermissionTier.TIER_2_MUTATION

    for name in TIER_3_TOOLS:
        cap = reg.get(name)
        assert cap is not None, f"Tool {name} missing from ToolRegistry"
        assert cap.tier == PermissionTier.TIER_3_HIGH_IMPACT


def test_permission_tier_single_authoritative_source():
    """Verify PermissionTier is unified into a single authoritative class across all modules."""
    from shamu.types import PermissionTier as TypesTier
    from shamu.security.tool_registry import PermissionTier as RegistryTier
    from shamu.security.firewall import PermissionTier as FirewallTier
    from shamu.security import PermissionTier as SecurityTier
    from shamu.protocols import PermissionTier as ProtocolsTier

    assert TypesTier is RegistryTier
    assert TypesTier is FirewallTier
    assert TypesTier is SecurityTier
    assert TypesTier is ProtocolsTier


def test_permission_tier_backward_compatibility_and_equivalence():
    """Verify PermissionTier supports canonical and legacy aliases with full bidirectional equality and hashing."""
    # Canonical mapping equality
    assert PermissionTier.TIER_1_READ_ONLY == PermissionTier.TIER_1_AUTO
    assert PermissionTier.TIER_2_MUTATION == PermissionTier.TIER_2_AUDITED
    assert PermissionTier.TIER_3_HIGH_IMPACT == PermissionTier.TIER_3_CONFIRMATION

    # String equality
    assert PermissionTier.TIER_1_READ_ONLY == "TIER_1_AUTO"
    assert PermissionTier.TIER_1_AUTO == "TIER_1_READ_ONLY"
    assert PermissionTier.TIER_2_MUTATION == "TIER_2_AUDITED"
    assert PermissionTier.TIER_3_HIGH_IMPACT == "TIER_3_CONFIRMATION"

    # Distinct tiers do not compare equal
    assert PermissionTier.TIER_1_READ_ONLY != PermissionTier.TIER_2_MUTATION
    assert PermissionTier.TIER_2_MUTATION != PermissionTier.TIER_3_HIGH_IMPACT
    assert PermissionTier.TIER_1_READ_ONLY != PermissionTier.PROHIBITED

    # Canonical helper method
    assert PermissionTier.TIER_1_AUTO.canonical() == PermissionTier.TIER_1_READ_ONLY
    assert PermissionTier.TIER_2_AUDITED.canonical() == PermissionTier.TIER_2_MUTATION
    assert PermissionTier.TIER_3_CONFIRMATION.canonical() == PermissionTier.TIER_3_HIGH_IMPACT
    assert PermissionTier.TIER_1_READ_ONLY.canonical() == PermissionTier.TIER_1_READ_ONLY

    # Hash consistency and set membership
    assert hash(PermissionTier.TIER_1_READ_ONLY) == hash(PermissionTier.TIER_1_AUTO)
    assert hash(PermissionTier.TIER_2_MUTATION) == hash(PermissionTier.TIER_2_AUDITED)
    assert hash(PermissionTier.TIER_3_HIGH_IMPACT) == hash(PermissionTier.TIER_3_CONFIRMATION)

    s1 = {PermissionTier.TIER_1_READ_ONLY}
    assert PermissionTier.TIER_1_AUTO in s1

    s2 = {PermissionTier.TIER_2_MUTATION}
    assert PermissionTier.TIER_2_AUDITED in s2

    s3 = {PermissionTier.TIER_3_HIGH_IMPACT}
    assert PermissionTier.TIER_3_CONFIRMATION in s3


def test_registry_and_firewall_tier_alignment(tmp_path):
    """Verify that all capabilities in ToolRegistry align with ExecutionFirewall evaluation."""
    from shamu.security.firewall import ExecutionFirewall

    fw = ExecutionFirewall(workspace_root=tmp_path)
    reg = ToolRegistry.create_default()

    sample_args = {
        "read_file": {"path": str(tmp_path / "test.txt")},
        "list_directory": {"path": str(tmp_path)},
        "get_system_metrics": {},
        "search_local_history": {"query": "test"},
        "write_file": {"path": str(tmp_path / "test.txt"), "content": "hi"},
        "append_file": {"path": str(tmp_path / "test.txt"), "content": "hi"},
        "focus_application": {"window_title": "test"},
        "powershell_exec": {"command": "Get-Date"},
        "delete_file": {"path": str(tmp_path / "test.txt")},
        "kill_process": {"pid": 99999},
        "format_disk": {},
        "reg_delete": {},
        "disable_firewall": {},
        "elevate_uac": {},
    }

    for cap in reg.list_all():
        args = sample_args.get(cap.name, {})
        decision = fw.evaluate_tool_call(cap.name, args)

        if cap.tier == PermissionTier.PROHIBITED or cap.is_prohibited:
            assert decision.tier == PermissionTier.PROHIBITED
            assert not decision.allowed
        elif cap.tier == PermissionTier.TIER_1_READ_ONLY:
            assert decision.tier == PermissionTier.TIER_1_READ_ONLY
            assert decision.tier == PermissionTier.TIER_1_AUTO
            assert decision.allowed
            assert not decision.requires_confirmation
        elif cap.tier == PermissionTier.TIER_2_MUTATION:
            assert decision.tier == PermissionTier.TIER_2_MUTATION
            assert decision.tier == PermissionTier.TIER_2_AUDITED
            assert decision.allowed
        elif cap.tier == PermissionTier.TIER_3_HIGH_IMPACT:
            assert decision.tier == PermissionTier.TIER_3_HIGH_IMPACT
            assert decision.tier == PermissionTier.TIER_3_CONFIRMATION
            assert decision.allowed
            assert decision.requires_confirmation



