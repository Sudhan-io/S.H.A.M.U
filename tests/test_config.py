import json
import os
import sys
import pytest
from pathlib import Path
from shamu.config import ShamuConfig, get_config, set_api_key, get_api_key, get_resource_path

def test_default_config():
    cfg = ShamuConfig()
    assert cfg.local_model == "qwen2.5:7b-instruct-q4_K_M"
    assert cfg.fallback_model == "google/gemini-2.5-flash"
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.timeout_seconds == 30
    assert cfg.workspace_root.exists()

def test_save_and_load_config(tmp_path: Path):
    cfg_file = tmp_path / "config.json"
    custom_cfg = ShamuConfig(
        local_model="llama3.2:3b",
        fallback_model="anthropic/claude-3.5-sonnet",
        timeout_seconds=45,
    )
    custom_cfg.save_to_file(cfg_file)

    loaded = ShamuConfig.load_from_file(cfg_file)
    assert loaded.local_model == "llama3.2:3b"
    assert loaded.fallback_model == "anthropic/claude-3.5-sonnet"
    assert loaded.timeout_seconds == 45

def test_keyring_secret_management(monkeypatch):
    mock_store = {}
    import keyring.backend

    class MockKeyring(keyring.backend.KeyringBackend):
        priority = 1
        def get_password(self, service, username):
            return mock_store.get((service, username))
        def set_password(self, service, username, password):
            mock_store[(service, username)] = password
        def delete_password(self, service, username):
            mock_store.pop((service, username), None)

    monkeypatch.setattr("keyring.get_keyring", lambda: MockKeyring())

    set_api_key("openrouter", "sk-test-key-12345")
    retrieved = get_api_key("openrouter")
    assert retrieved == "sk-test-key-12345"


def test_speech_config_defaults():
    cfg = ShamuConfig()
    assert hasattr(cfg, "speech")
    assert cfg.speech.stt_model_size == "large-v3-turbo"
    assert cfg.speech.stt_cpu_threads == 4
    assert cfg.speech.stt_p_cores == [0, 1, 2, 3]
    assert cfg.speech.pre_speech_buffer_ms == 200
    assert cfg.speech.custom_lexicon_path == "data/lexicon.json"


def test_memory_and_inference_config_defaults():
    from shamu.config import MemoryConfig, InferenceConfig
    from shamu.types import MemoryRecord
    from shamu.protocols import MemoryStore, Retriever

    cfg = ShamuConfig()
    assert hasattr(cfg, "memory")
    assert hasattr(cfg, "inference")

    # MemoryConfig defaults
    assert cfg.memory.db_path == "data/memory.db"
    assert cfg.memory.fts_enabled is True
    assert cfg.memory.bm25_k1 == 1.2
    assert cfg.memory.bm25_b == 0.75
    assert cfg.memory.embedding_enabled is True
    assert cfg.memory.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.memory.hybrid_rrf_k == 60
    assert cfg.memory.context_token_budget == 4000
    assert cfg.memory.max_retrieved_memories == 5
    assert cfg.memory.min_relevance_score == 0.0

    # InferenceConfig defaults
    assert cfg.inference.num_ctx == 4096
    assert cfg.inference.temperature == 0.7
    assert cfg.inference.flash_attention is False
    assert cfg.inference.cache_type_k == "f16"
    assert cfg.inference.cache_type_v == "f16"
    assert cfg.inference.keep_alive == "5m"
    assert cfg.inference.prompt_lookup_decoding is False

    # MemoryRecord check
    rec = MemoryRecord(
        id="mem-1",
        content="User preferred dark mode",
        source="conversation",
        category="preference",
        metadata={"user": "Sudhan"},
        created_at=1700000000.0,
        score=0.95,
    )
    assert rec.id == "mem-1"
    assert rec.score == 0.95
    assert rec.metadata["user"] == "Sudhan"


    # Protocols runtime checkable
    assert isinstance(MemoryStore, type)
    assert isinstance(Retriever, type)


def test_get_api_key_from_environment(monkeypatch):
    """PKG-03: Environment variables must be prioritized for API key resolution."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-12345")
    assert get_api_key("openrouter") == "sk-or-env-12345"

    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("SHAMU_OPENROUTER_API_KEY", "sk-shamu-env-67890")
    assert get_api_key("openrouter") == "sk-shamu-env-67890"


def test_resolved_config_paths_anchored_to_home():
    """PKG-02: Default relative config paths resolve anchored under ~/.shamu."""
    cfg = ShamuConfig()
    expected_db = Path.home() / ".shamu" / "data" / "memory.db"
    expected_lexicon = Path.home() / ".shamu" / "data" / "lexicon.json"

    assert cfg.memory.resolved_db_path == expected_db
    assert cfg.speech.resolved_lexicon_path == expected_lexicon

    # Explicit absolute path preserved
    custom_cfg = ShamuConfig(memory={"db_path": "C:/custom/path/db.sqlite"})
    assert custom_cfg.memory.resolved_db_path == Path("C:/custom/path/db.sqlite")


def test_get_resource_path_standard(monkeypatch):
    """Resource path resolves relative to repository root in standard (non-frozen) mode."""
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    repo_root = Path(__file__).resolve().parent.parent
    expected = repo_root / "data" / "lexicon.json"
    res = get_resource_path("data/lexicon.json")
    assert res == expected
    assert res.is_absolute()
    assert res.exists()


def test_get_resource_path_meipass(monkeypatch, tmp_path: Path):
    """Resource path resolves relative to sys._MEIPASS when running in a PyInstaller bundle."""
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    expected = tmp_path / "data" / "lexicon.json"
    res = get_resource_path("data/lexicon.json")
    assert res == expected


def test_save_to_file_atomic(tmp_path: Path, monkeypatch):
    """Config saving uses atomic temporary file replacement."""
    cfg_file = tmp_path / "config.json"
    cfg = ShamuConfig(timeout_seconds=99)

    replace_calls = []
    orig_replace = os.replace

    def mock_replace(src, dst):
        assert Path(src) == tmp_path / ".config.tmp"
        assert Path(src).exists()
        replace_calls.append((src, dst))
        return orig_replace(src, dst)

    monkeypatch.setattr(os, "replace", mock_replace)
    cfg.save_to_file(cfg_file)

    assert len(replace_calls) == 1
    assert cfg_file.exists()
    assert not (tmp_path / ".config.tmp").exists()

    loaded = ShamuConfig.load_from_file(cfg_file)
    assert loaded.timeout_seconds == 99


def test_load_from_file_corrupt_json_fallback(tmp_path: Path):
    """Corrupt JSON file triggers fallback to default config with .corrupted backup."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{malformed: json syntax", encoding="utf-8")

    loaded = ShamuConfig.load_from_file(cfg_file)
    assert loaded.local_model == "qwen2.5:7b-instruct-q4_K_M"

    backup_file = tmp_path / "config.json.corrupted"
    assert backup_file.exists()
    assert backup_file.read_text(encoding="utf-8") == "{malformed: json syntax"


def test_load_from_file_validation_error_fallback(tmp_path: Path):
    """Pydantic validation failure triggers fallback to default config with .corrupted backup."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"timeout_seconds": "not-an-integer"}), encoding="utf-8")

    loaded = ShamuConfig.load_from_file(cfg_file)
    assert loaded.timeout_seconds == 30

    backup_file = tmp_path / "config.json.corrupted"
    assert backup_file.exists()


def test_load_from_file_os_error_fallback(tmp_path: Path, monkeypatch):
    """OSError during read triggers fallback to default config without crashing."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")

    def mock_open(*args, **kwargs):
        raise OSError("Simulated disk I/O error")

    monkeypatch.setattr("builtins.open", mock_open)
    loaded = ShamuConfig.load_from_file(cfg_file)
    assert loaded.local_model == "qwen2.5:7b-instruct-q4_K_M"




