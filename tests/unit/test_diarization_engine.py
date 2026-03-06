from types import SimpleNamespace

from src.config import Config
from src.diarization.engine import DiarizationEngine


class _FakeAnnotation:
    def __init__(self):
        self._tracks = [
            (SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00"),
        ]

    def itertracks(self, yield_label=False):
        return self._tracks


def test_unwrap_diarization_annotation_direct():
    config = Config(include_diarization=False)
    engine = DiarizationEngine(config, test_mode=True)
    annotation = _FakeAnnotation()

    assert engine._unwrap_diarization_result(annotation) is annotation


def test_unwrap_diarization_output_object():
    config = Config(include_diarization=False)
    engine = DiarizationEngine(config, test_mode=True)
    annotation = _FakeAnnotation()
    diarize_output = SimpleNamespace(speaker_diarization=annotation)

    assert engine._unwrap_diarization_result(diarize_output) is annotation
