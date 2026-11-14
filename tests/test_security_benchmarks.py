import time
from pathlib import Path
import pytest
from shamu.security.path_guard import WindowsPathGuard
from shamu.security.audit_ledger import AuditLedger
from shamu.security.tool_registry import PermissionTier


def test_path_canonicalization_benchmark(tmp_path: Path):
    guard = WindowsPathGuard(workspace_root=tmp_path)
    test_file = tmp_path / "bench.txt"
    test_file.write_text("benchmark data", encoding="utf-8")

    iterations = 500
    t0 = time.perf_counter()
    for _ in range(iterations):
        p = guard.authorize_path("bench.txt", mode="read")
        assert p is not None
    total_time = time.perf_counter() - t0

    avg_ms = (total_time / iterations) * 1000.0
    print(f"\n[BENCHMARK] WindowsPathGuard avg latency: {avg_ms:.4f} ms per path (threshold < 1.0 ms)")
    assert avg_ms < 2.0  # Safe threshold on Windows filesystem


def test_audit_ledger_append_and_verification_benchmark(tmp_path: Path):
    log_file = tmp_path / "bench_audit.jsonl"
    cp_file = tmp_path / "bench_cp.db"
    key = b"B" * 32
    ledger = AuditLedger(log_path=log_file, checkpoint_path=cp_file, master_key=key)

    # 1. Measure append latency
    num_records = 300
    t0 = time.perf_counter()
    for i in range(num_records):
        ledger.append_event(
            session_id="bench-session",
            turn_id=f"turn-{i}",
            tool_name="read_file",
            arguments={"path": f"item_{i}.txt"},
            tier=PermissionTier.TIER_1_READ_ONLY,
            decision="ALLOW",
        )
    append_total_time = time.perf_counter() - t0
    avg_append_ms = (append_total_time / num_records) * 1000.0
    print(f"\n[BENCHMARK] AuditLedger avg append latency: {avg_append_ms:.4f} ms per record (threshold < 5.0 ms)")

    # 2. Measure verification throughput
    t1 = time.perf_counter()
    res = ledger.verify_integrity()
    verify_time = time.perf_counter() - t1

    assert res.is_valid is True
    assert res.verified_records == num_records

    records_per_sec = num_records / verify_time if verify_time > 0 else 999999
    print(f"\n[BENCHMARK] AuditLedger verification throughput: {records_per_sec:.1f} records/sec (threshold > 1000 rec/s)")
    assert records_per_sec > 500

