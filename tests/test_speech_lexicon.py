"""Tests for DynamicLexicon context biasing engine."""

import json
from pathlib import Path
import pytest
from shamu.speech.lexicon import DynamicLexicon


def test_dynamic_lexicon_defaults(tmp_path: Path):
    lexicon_file = tmp_path / "lexicon.json"
    lex = DynamicLexicon(custom_lexicon_path=lexicon_file)
    prompt = lex.build_initial_prompt()

    assert "S.H.A.M.U." in prompt
    assert "PySide6" in prompt
    assert "Ollama" in prompt
    # Check that lexicon file was automatically created
    assert lexicon_file.exists()


def test_dynamic_lexicon_custom_terms(tmp_path: Path):
    lexicon_file = tmp_path / "lexicon.json"
    lexicon_file.write_text(json.dumps({"terms": ["FastAPI", "Docker", "S.H.A.M.U."]}), encoding="utf-8")

    lex = DynamicLexicon(custom_lexicon_path=lexicon_file)
    prompt = lex.build_initial_prompt()

    assert "FastAPI" in prompt
    assert "Docker" in prompt
    assert "S.H.A.M.U." in prompt


def test_dynamic_lexicon_workspace_extraction(tmp_path: Path):
    # Setup mock workspace files and folders
    (tmp_path / "auth_service.py").touch()
    (tmp_path / "database").mkdir()
    (tmp_path / "__pycache__").mkdir()  # should be excluded
    (tmp_path / ".venv").mkdir()         # should be excluded

    lexicon_file = tmp_path / "lexicon.json"
    lex = DynamicLexicon(custom_lexicon_path=lexicon_file)
    prompt = lex.build_initial_prompt(workspace_dir=tmp_path)

    assert "auth_service" in prompt
    assert "database" in prompt
    assert "__pycache__" not in prompt
    assert ".venv" not in prompt


def test_dynamic_lexicon_deduplication(tmp_path: Path):
    lexicon_file = tmp_path / "lexicon.json"
    lexicon_file.write_text(json.dumps({"terms": ["pyside6", "PySide6", "OLLAMA", "ollama"]}), encoding="utf-8")

    lex = DynamicLexicon(custom_lexicon_path=lexicon_file)
    terms = lex.collect_terms()

    # Case-insensitive deduplication check
    lower_terms = [t.lower() for t in terms]
    assert lower_terms.count("pyside6") == 1
    assert lower_terms.count("ollama") == 1


def test_dynamic_lexicon_token_limit_bounding(tmp_path: Path):
    # Supply 300 terms to verify truncation respects token budget
    huge_terms = [f"customTermNumber{i}" for i in range(300)]
    lexicon_file = tmp_path / "lexicon.json"
    lexicon_file.write_text(json.dumps({"terms": huge_terms}), encoding="utf-8")

    lex = DynamicLexicon(custom_lexicon_path=lexicon_file)
    prompt = lex.build_initial_prompt(max_words=100)

    # Must be bounded under word count
    words = prompt.split()
    assert len(words) <= 100


def test_dynamic_lexicon_malformed_json_graceful_fallback(tmp_path: Path):
    lexicon_file = tmp_path / "corrupt.json"
    lexicon_file.write_text("{this is corrupted json!", encoding="utf-8")

    lex = DynamicLexicon(custom_lexicon_path=lexicon_file)
    prompt = lex.build_initial_prompt()

    # Must not raise exception, must use defaults
    assert "S.H.A.M.U." in prompt
    assert "Gravitas" in prompt

