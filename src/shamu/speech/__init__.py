"""SHAMU Speech and Voice Subsystem (CPU Offloaded)."""

from shamu.speech.audio_capture import PushToTalkCapture
from shamu.speech.vad import VoiceActivityDetector, VADState
from shamu.speech.stt import FasterWhisperSTT
from shamu.speech.tts import TextToSpeech
from shamu.speech.sentence_splitter import SentenceSplitter
from shamu.speech.lexicon import DynamicLexicon
from shamu.speech.pipeline import VoicePipeline

__all__ = [
    "PushToTalkCapture",
    "VoiceActivityDetector",
    "VADState",
    "FasterWhisperSTT",
    "TextToSpeech",
    "SentenceSplitter",
    "DynamicLexicon",
    "VoicePipeline",
]
