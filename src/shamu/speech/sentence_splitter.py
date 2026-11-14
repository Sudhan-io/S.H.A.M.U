"""Incremental sentence and sub-sentence clause splitter for real-time streaming TTS synthesis."""

import re
from typing import List, Optional

# Common English abbreviations that shouldn't trigger sentence splitting
COMMON_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "eg", "ie", "al", "fig", "st",
}


class SentenceSplitter:
    """Buffers streaming token deltas and yields complete sentences or sub-sentence clauses."""

    def __init__(self, min_clause_words: int = 4):
        self.min_clause_words = min_clause_words
        self._buffer = ""

    def _is_number_internal(self, text: str, match_start: int, match_end: int) -> bool:
        """Checks if punctuation is part of a number (e.g. 3.14 or 10,000)."""
        if match_start > 0 and match_end < len(text):
            prev_char = text[match_start - 1]
            next_char = text[match_end]
            if prev_char.isdigit() and next_char.isdigit():
                return True
        return False

    def _is_abbreviation(self, candidate_words: List[str]) -> bool:
        """Checks if the last word in candidate is an abbreviation."""
        if not candidate_words:
            return False
        last_word = candidate_words[-1].rstrip(".!?").lower()
        return last_word in COMMON_ABBREVIATIONS

    def feed(self, text_chunk: str) -> List[str]:
        """Appends text chunk and returns list of completed sentences or clauses."""
        self._buffer += text_chunk
        results: List[str] = []

        start_pos = 0
        while True:
            # Pattern matching sentence ends (.!?) or clause punctuation (,:;\n—) followed by whitespace or end
            match = re.search(r'([.!?]+|[,;:\n—]+)(\s+|$)', self._buffer[start_pos:])
            if not match:
                break

            punct = match.group(1)
            punct_start = start_pos + match.start(1)
            punct_end = start_pos + match.end(1)
            full_match_end = start_pos + match.end()

            # 1. Skip if internal to a number (e.g. 3.14 or 10,000)
            if self._is_number_internal(self._buffer, punct_start, punct_end):
                start_pos = full_match_end
                continue

            candidate = self._buffer[:punct_end].strip()
            words = candidate.split()
            is_sentence_end = any(c in ".!?" for c in punct)

            # 2. Check for abbreviations on sentence end
            if is_sentence_end and self._is_abbreviation(words):
                start_pos = full_match_end
                continue

            # 3. Check for clause boundary (comma, semicolon, etc.)
            if not is_sentence_end and len(words) < self.min_clause_words:
                start_pos = full_match_end
                continue

            # Valid boundary found!
            if candidate:
                results.append(candidate)
            self._buffer = self._buffer[full_match_end:]
            start_pos = 0

        return results

    def flush(self) -> List[str]:
        """Flushes remaining text in buffer when stream ends."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            return [remaining]
        return []
