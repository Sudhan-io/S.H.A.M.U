"""Tests for enhanced sub-sentence clause splitting in SentenceSplitter."""

import pytest
from shamu.speech.sentence_splitter import SentenceSplitter


def test_clause_splitting_early_emits_for_fast_ttfa():
    splitter = SentenceSplitter(min_clause_words=4)
    # 4 words before the comma -> should emit clause immediately
    chunks = splitter.feed("Checking your local repositories now, please hold on a moment.")
    assert len(chunks) == 2
    assert chunks[0] == "Checking your local repositories now,"
    assert chunks[1] == "please hold on a moment."


def test_clause_splitting_respects_decimal_numbers():
    splitter = SentenceSplitter(min_clause_words=4)
    # Decimal number 3.14 must not be treated as sentence boundary
    chunks = splitter.feed("Python release 3.14 has arrived today.")
    assert len(chunks) == 1
    assert chunks[0] == "Python release 3.14 has arrived today."


def test_clause_splitting_respects_number_commas():
    splitter = SentenceSplitter(min_clause_words=4)
    # 10,000 should not split on comma
    chunks = splitter.feed("We indexed 10,000 files in the workspace.")
    assert len(chunks) == 1
    assert chunks[0] == "We indexed 10,000 files in the workspace."


def test_clause_splitting_preserves_abbreviations():
    splitter = SentenceSplitter(min_clause_words=4)
    chunks = splitter.feed("Dr. Smith visited Prof. Davis yesterday afternoon.")
    assert len(chunks) == 1
    assert chunks[0] == "Dr. Smith visited Prof. Davis yesterday afternoon."


def test_clause_splitting_incremental_token_streaming():
    splitter = SentenceSplitter(min_clause_words=3)
    # Stream token by token
    tokens = ["I ", "am ", "ready, ", "let ", "us ", "begin ", "now."]
    emitted = []
    for token in tokens:
        emitted.extend(splitter.feed(token))

    assert len(emitted) == 2
    assert emitted[0] == "I am ready,"
    assert emitted[1] == "let us begin now."


def test_clause_splitter_flush_remainder():
    splitter = SentenceSplitter(min_clause_words=4)
    chunks = splitter.feed("Incomplete sentence buffer without punctuation")
    assert len(chunks) == 0

    flushed = splitter.flush()
    assert len(flushed) == 1
    assert flushed[0] == "Incomplete sentence buffer without punctuation"

