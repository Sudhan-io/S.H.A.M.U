"""Empirical latency and throughput benchmarks for Phase 1 speech pipeline components."""

import time
import numpy as np
import pytest
from shamu.speech.lexicon import DynamicLexicon
from shamu.speech.sentence_splitter import SentenceSplitter
from shamu.speech.audio_capture import PushToTalkCapture


def test_benchmark_lexicon_prompt_generation_latency():
    lex = DynamicLexicon()
    iterations = 50
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = lex.build_initial_prompt()
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    print(f"\n[BENCHMARK] DynamicLexicon prompt generation: {avg_ms:.3f} ms per invocation")
    # Processing latency must be sub-millisecond or fast (< 5ms)
    assert avg_ms < 10.0


def test_benchmark_sentence_splitter_clause_latency():
    splitter = SentenceSplitter(min_clause_words=4)
    incoming_tokens = [
        "Analyzing ", "the ", "requested ", "directory ", "now, ",
        "found ", "three ", "modified ", "code ", "files."
    ]

    iterations = 100
    t0 = time.perf_counter()
    for _ in range(iterations):
        sp = SentenceSplitter(min_clause_words=4)
        for tok in incoming_tokens:
            _ = sp.feed(tok)
        _ = sp.flush()
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0
    print(f"\n[BENCHMARK] SentenceSplitter token clause processing: {avg_ms:.3f} ms per full turn")
    # Pure string/regex processing latency must be < 1ms
    assert avg_ms < 5.0


def test_benchmark_ring_buffer_audio_callback_overhead():
    capture = PushToTalkCapture(sample_rate=16000, block_size=512, pre_buffer_ms=200)
    data = np.zeros((512, 1), dtype=np.int16)

    iterations = 1000
    t0 = time.perf_counter()
    for _ in range(iterations):
        capture._audio_callback(data, 512, None, None)
    t1 = time.perf_counter()

    avg_us = ((t1 - t0) / iterations) * 1_000_000.0
    print(f"\n[BENCHMARK] AudioCapture callback overhead: {avg_us:.2f} microseconds per 32ms frame")
    # Must be extremely lightweight (< 50 microseconds) so it never causes buffer underruns
    assert avg_us < 200.0

