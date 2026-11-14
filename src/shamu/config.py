"""Configuration and secrets management for SHAMU."""

import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Optional, Union
from pydantic import BaseModel, Field, ValidationError
import keyring

logger = logging.getLogger(__name__)

SERVICE_NAME = "shamu_assistant"


def get_resource_path(relative_path: Union[str, Path]) -> Path:
    """Resolve resource path across frozen PyInstaller bundles and source tree.

    If running in a PyInstaller bundle, sys._MEIPASS is used as the base.
    Otherwise, the path is resolved relative to the repository root.
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parent.parent.parent / relative_path


class SpeechConfig(BaseModel):
    """Configuration for local speech recognition, synthesis, and VAD."""
    stt_model_size: str = Field(default="large-v3-turbo", description="Whisper model size/variant")
    stt_cpu_threads: int = Field(default=4, description="Number of CPU threads dedicated to STT decoding")
    stt_p_cores: list[int] = Field(default_factory=lambda: [0, 1, 2, 3], description="Intel physical P-Core affinity list")
    vad_trailing_silence_ms: int = Field(default=400, description="Milliseconds of trailing silence to finalize recording")
    pre_speech_buffer_ms: int = Field(default=200, description="Milliseconds of audio pre-buffering to prevent clipped starts")
    custom_lexicon_path: str = Field(default="data/lexicon.json", description="Path to custom vocabulary and domain terminology file")
    tts_engine: str = Field(default="kokoro", description="Primary TTS engine: kokoro, piper, or sapi5")
    tts_model_path: Optional[str] = Field(default=None, description="Path to Kokoro ONNX model file or Piper onnx file")
    tts_voices_path: Optional[str] = Field(default=None, description="Path to Kokoro voices.bin file or voices directory")
    tts_voice: str = Field(default="af_heart", description="Default voice identifier for Kokoro TTS (e.g. af_heart, am_adam)")
    tts_speed: float = Field(default=1.0, description="Speech playback speed multiplier")
    tts_sample_rate: int = Field(default=24000, description="Audio sample rate (24000 Hz for Kokoro, 22050 Hz for Piper)")

    @property
    def resolved_lexicon_path(self) -> Path:
        """Returns the canonical absolute lexicon path anchored to ~/.shamu if default."""
        p = Path(self.custom_lexicon_path)
        if not p.is_absolute():
            return Path.home() / ".shamu" / p
        return p

    @property
    def resolved_tts_model_path(self) -> Optional[Path]:
        """Resolves Kokoro ONNX model path from config or standard locations."""
        if self.tts_model_path:
            p = Path(self.tts_model_path)
            if p.is_absolute() and p.exists():
                return p
            candidate = Path.home() / ".shamu" / p
            if candidate.exists():
                return candidate
        candidates = [
            Path.home() / ".shamu" / "models" / "kokoro-v1.0.onnx",
            Path.home() / ".shamu" / "models" / "kokoro-v1.0.int8.onnx",
            Path.home() / ".shamu" / "models" / "kokoro.onnx",
            Path("models/kokoro-v1.0.onnx"),
            Path("models/kokoro.onnx"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return None

    @property
    def resolved_tts_voices_path(self) -> Optional[Path]:
        """Resolves Kokoro voices.bin path from config or standard locations."""
        if self.tts_voices_path:
            p = Path(self.tts_voices_path)
            if p.is_absolute() and p.exists():
                return p
            candidate = Path.home() / ".shamu" / p
            if candidate.exists():
                return candidate
        candidates = [
            Path.home() / ".shamu" / "models" / "voices-v1.0.bin",
            Path.home() / ".shamu" / "models" / "voices.bin",
            Path("models/voices-v1.0.bin"),
            Path("models/voices.bin"),
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return None


class MemoryConfig(BaseModel):
    """Configuration for persistent zero-daemon memory storage and hybrid retrieval."""
    db_path: str = Field(default="data/memory.db", description="Path to SQLite memory database")
    fts_enabled: bool = Field(default=True, description="Enable SQLite FTS5 full-text lexical indexing")
    bm25_k1: float = Field(default=1.2, description="BM25 term frequency saturation parameter")
    bm25_b: float = Field(default=0.75, description="BM25 document length normalization parameter")
    embedding_enabled: bool = Field(default=True, description="Enable local semantic embedding search")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", description="Local sentence transformer embedding model")
    hybrid_rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing constant k")
    context_token_budget: int = Field(default=4000, description="Maximum token budget allocated for conversation history")
    max_retrieved_memories: int = Field(default=5, description="Maximum number of memory facts injected into prompt context")
    min_relevance_score: float = Field(default=0.0, description="Minimum hybrid score threshold for injection")

    @property
    def resolved_db_path(self) -> Path:
        """Returns the canonical absolute database path anchored to ~/.shamu if default."""
        p = Path(self.db_path)
        if not p.is_absolute():
            return Path.home() / ".shamu" / p
        return p



class InferenceConfig(BaseModel):
    """Configuration for local Ollama inference optimization parameters."""
    num_ctx: int = Field(default=4096, description="Context window size allocated in VRAM")
    temperature: float = Field(default=0.7, description="Generation sampling temperature")
    flash_attention: bool = Field(default=False, description="Enable Flash Attention in Ollama backend if supported")
    cache_type_k: str = Field(default="f16", description="KV cache key quantization type (e.g. f16, q8_0, q4_0)")
    cache_type_v: str = Field(default="f16", description="KV cache value quantization type (e.g. f16, q8_0, q4_0)")
    keep_alive: str = Field(default="5m", description="Model stay-in-VRAM duration after last inference")
    prompt_lookup_decoding: bool = Field(default=False, description="Enable speculative prompt lookup decoding if supported")


class ShamuConfig(BaseModel):
    """Global SHAMU assistant settings."""
    local_model: str = Field(default="qwen2.5:7b-instruct-q4_K_M", description="Primary local Ollama model")
    fallback_model: str = Field(default="google/gemini-2.5-flash", description="Cloud OpenRouter fallback model")
    ollama_url: str = Field(default="http://localhost:11434", description="Base URL for local Ollama instance")
    timeout_seconds: int = Field(default=30, description="HTTP request timeout in seconds")
    workspace_root: Path = Field(default_factory=Path.cwd, description="Default root directory for workspace file tasks")
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".shamu", description="User data directory")
    speech: SpeechConfig = Field(default_factory=SpeechConfig, description="Speech subsystem configuration")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="Persistent hybrid memory configuration")
    inference: InferenceConfig = Field(default_factory=InferenceConfig, description="Local inference optimization configuration")

    @classmethod
    def load_from_file(cls, path: Path) -> "ShamuConfig":
        """Loads configuration from JSON file with corrupted fallback recovery."""
        target_path = Path(path)
        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data)
            except (json.JSONDecodeError, ValidationError, OSError) as e:
                logger.warning(
                    "Failed to load configuration from %s: %s. Creating .corrupted backup and falling back to default.",
                    target_path,
                    e,
                )
                try:
                    corrupted_path = target_path.with_name(f"{target_path.name}.corrupted")
                    shutil.copy2(target_path, corrupted_path)
                except Exception as backup_err:
                    logger.error("Failed to create corrupted config backup: %s", backup_err)
                return cls()
        return cls()

    def save_to_file(self, path: Path) -> None:
        """Saves configuration to JSON file atomically."""
        dest_path = Path(path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.parent / ".config.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        os.replace(tmp_path, dest_path)


_ACTIVE_CONFIG: Optional[ShamuConfig] = None


def get_config() -> ShamuConfig:
    """Returns the singleton configuration instance, initializing if needed."""
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        default_path = Path.home() / ".shamu" / "config.json"
        _ACTIVE_CONFIG = ShamuConfig.load_from_file(default_path)
    return _ACTIVE_CONFIG


# =====================================================================
# Windows DPAPI Keyring Secret Store
# =====================================================================

def set_api_key(service: str, key: str) -> None:
    """Stores a secret securely in Windows Credential Manager via keyring."""
    kr = keyring.get_keyring()
    kr.set_password(SERVICE_NAME, service, key)


def get_api_key(service: str) -> Optional[str]:
    """Retrieves a secret from environment variables or Windows Credential Manager via keyring."""
    # 1. Check process environment variables (e.g. OPENROUTER_API_KEY, SHAMU_OPENROUTER_API_KEY)
    env_candidates = [
        f"{service.upper()}_API_KEY",
        f"SHAMU_{service.upper()}_API_KEY",
    ]
    for env_var in env_candidates:
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip()

    # 2. Check Windows Credential Manager via keyring
    try:
        kr = keyring.get_keyring()
        return kr.get_password(SERVICE_NAME, service)
    except Exception:
        return None


def delete_api_key(service: str) -> None:
    """Removes a secret from Windows Credential Manager."""
    try:
        kr = keyring.get_keyring()
        kr.delete_password(SERVICE_NAME, service)
    except Exception:
        pass

