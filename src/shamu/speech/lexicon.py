"""Dynamic vocabulary context biasing engine for Whisper speech recognition."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Set

from shamu.config import get_resource_path

logger = logging.getLogger(__name__)

DEFAULT_CORE_TERMS: List[str] = [
    "S.H.A.M.U.",
    "Disha",
    "Gravitas",
    "OpenClaw",
    "OpenJarvis",
    "PySide6",
    "Ollama",
    "Whisper",
    "Piper",
    "Pydantic",
    "CTranslate2",
]

EXCLUDED_WORKSPACE_PATTERNS: Set[str] = {
    "__pycache__",
    ".git",
    ".github",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "egg-info",
    ".idea",
    ".vscode",
}


class DynamicLexicon:
    """Extracts, aggregates, and formats domain vocabulary to bias Whisper speech decoding."""

    def __init__(self, custom_lexicon_path: Optional[Path | str] = None):
        self.custom_lexicon_path = (
            Path(custom_lexicon_path) if custom_lexicon_path else get_resource_path("data/lexicon.json")
        )
        self.core_terms = list(DEFAULT_CORE_TERMS)
        self._ensure_lexicon_file()

    def _ensure_lexicon_file(self) -> None:
        """Creates the default lexicon file if it does not exist."""
        if not self.custom_lexicon_path.exists():
            try:
                self.custom_lexicon_path.parent.mkdir(parents=True, exist_ok=True)
                default_data = {
                    "description": "User-customized and project-specific terms to bias speech recognition",
                    "terms": self.core_terms,
                }
                with open(self.custom_lexicon_path, "w", encoding="utf-8") as f:
                    json.dump(default_data, f, indent=2)
            except Exception as e:
                logger.warning("Failed to auto-create lexicon file %s: %s", self.custom_lexicon_path, e)

    def _load_custom_terms(self) -> List[str]:
        """Loads custom vocabulary from JSON file with graceful fallback."""
        if not self.custom_lexicon_path.exists():
            return []
        try:
            with open(self.custom_lexicon_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                terms = data.get("terms", [])
                if isinstance(terms, list):
                    return [str(t).strip() for t in terms if str(t).strip()]
        except Exception as e:
            logger.warning("Error reading lexicon file %s: %s. Using default terms.", self.custom_lexicon_path, e)
        return []

    def _extract_workspace_terms(self, workspace_dir: Path) -> List[str]:
        """Extracts candidate terms from workspace file and directory names."""
        terms: List[str] = []
        if not workspace_dir or not workspace_dir.exists():
            return terms

        try:
            for item in workspace_dir.iterdir():
                name = item.name
                if name.startswith(".") or name in EXCLUDED_WORKSPACE_PATTERNS:
                    continue

                if item.is_dir():
                    terms.append(name)
                elif item.is_file():
                    if item.suffix in (".py", ".md", ".json", ".toml"):
                        terms.append(item.stem)
        except Exception as e:
            logger.warning("Failed scanning workspace directory %s: %s", workspace_dir, e)

        return terms

    def collect_terms(self, workspace_dir: Optional[Path | str] = None) -> List[str]:
        """Collects, deduplicates, and preserves casing for all active terms."""
        all_candidates: List[str] = []
        all_candidates.extend(self.core_terms)
        all_candidates.extend(self._load_custom_terms())

        if workspace_dir:
            ws_path = Path(workspace_dir)
            all_candidates.extend(self._extract_workspace_terms(ws_path))

        # Case-insensitive deduplication, preserving first-seen capitalization
        seen_lower: Set[str] = set()
        deduped: List[str] = []
        for term in all_candidates:
            clean = term.strip()
            if not clean:
                continue
            lower = clean.lower()
            if lower not in seen_lower:
                seen_lower.add(lower)
                deduped.append(clean)

        return deduped

    def build_initial_prompt(
        self,
        workspace_dir: Optional[Path | str] = None,
        max_words: int = 150,
    ) -> str:
        """Constructs an autoregressive priming sentence under Whisper's token budget (~224 tokens)."""
        terms = self.collect_terms(workspace_dir=workspace_dir)

        prefix = "The following is a technical voice conversation with S.H.A.M.U. discussing software engineering, including "
        prefix_words = prefix.split()
        budget = max_words - len(prefix_words)

        selected_terms: List[str] = []
        current_words = 0
        for term in terms:
            term_word_count = len(term.split())
            if current_words + term_word_count > budget:
                break
            selected_terms.append(term)
            current_words += term_word_count

        if not selected_terms:
            return prefix.strip() + "."

        return f"{prefix}{', '.join(selected_terms)}."

