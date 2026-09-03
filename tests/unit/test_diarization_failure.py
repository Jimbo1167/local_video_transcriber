"""A failed diarization post-step must not destroy a finished transcription.

Regression tests for docs/bugs/2026-08-30-diarization-audiodecoder.md (Issue
C): transcription completed, then diarization raised, and the whole job was
reported as failed. Instead, the transcriber degrades to an unlabeled
transcript and surfaces the error so callers can attach a warning.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import Config
from src.service import TranscriptionService
from src.transcriber import Transcriber


SEGMENTS = [
    {"start": 0.0, "end": 1.0, "text": "hello"},
    {"start": 1.0, "end": 2.0, "text": "world"},
]


def _stub_engines(transcriber: Transcriber, *, diarize_error: Exception = None):
    transcriber.audio_processor.get_audio_path = (
        lambda _p: ("/tmp/fake_audio.wav", False)
    )
    transcriber.transcription_engine.ensure_model_loaded = lambda: None
    transcriber.diarization_engine.ensure_model_loaded = lambda force=False: None
    transcriber.transcription_engine.transcribe = lambda _p: SEGMENTS

    if diarize_error is not None:
        def failing_diarize(_p, enabled=None):
            raise diarize_error
        transcriber.diarization_engine.diarize = failing_diarize
    else:
        transcriber.diarization_engine.diarize = lambda _p, enabled=None: [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}
        ]


def _build_transcriber(*, diarize_error: Exception = None) -> Transcriber:
    config = Config(include_diarization=True, output_format="txt")
    transcriber = Transcriber(config, test_mode=True)
    _stub_engines(transcriber, diarize_error=diarize_error)
    return transcriber


def test_diarization_failure_returns_transcription():
    transcriber = _build_transcriber(
        diarize_error=Exception("name 'AudioDecoder' is not defined")
    )

    result = transcriber.transcribe("ignored.mp4")

    assert [(s[0], s[1], s[2]) for s in result] == [
        (0.0, 1.0, "hello"),
        (1.0, 2.0, "world"),
    ]
    assert all(speaker == "" for _, _, _, speaker in result)
    assert "AudioDecoder" in transcriber.last_diarization_error


def test_diarization_error_resets_between_calls():
    transcriber = _build_transcriber(diarize_error=Exception("boom"))

    transcriber.transcribe("ignored.mp4")
    assert transcriber.last_diarization_error is not None

    _stub_engines(transcriber)  # healthy diarization again
    result = transcriber.transcribe("ignored.mp4")

    assert transcriber.last_diarization_error is None
    assert result[0][3] == "SPEAKER_00"


def test_transcription_failure_still_raises():
    """Only the post-step degrades; a failed transcription is a failed job."""
    transcriber = _build_transcriber()

    def failing_transcribe(_p):
        raise RuntimeError("asr blew up")

    transcriber.transcription_engine.transcribe = failing_transcribe

    with pytest.raises(RuntimeError, match="asr blew up"):
        transcriber.transcribe("ignored.mp4")


def test_service_result_carries_diarization_error(tmp_path):
    config = Config(include_diarization=True, output_format="txt")
    service = TranscriptionService(config, test_mode=True)
    _stub_engines(service.transcriber, diarize_error=Exception("decoder gone"))

    result = service.transcribe_file(
        "ignored.mp4", output_path=str(tmp_path / "out.txt")
    )

    assert result["diarization_error"] == "decoder gone"
    assert len(result["segments"]) == 2
    assert (tmp_path / "out.txt").exists()


def test_service_result_no_error_on_success(tmp_path):
    config = Config(include_diarization=True, output_format="txt")
    service = TranscriptionService(config, test_mode=True)
    _stub_engines(service.transcriber)

    result = service.transcribe_file(
        "ignored.mp4", output_path=str(tmp_path / "out.txt")
    )

    assert result["diarization_error"] is None
