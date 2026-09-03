"""Client-side request options (scripts/model_client.py).

Regression test for docs/bugs/2026-08-30-diarization-audiodecoder.md (Issue
B): the client must always send an explicit diarize field so a server whose
INCLUDE_DIARIZATION default is on never diarizes a request that didn't ask.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_CLIENT_PATH = _REPO_ROOT / "scripts" / "model_client.py"
spec = importlib.util.spec_from_file_location("model_client_module", _CLIENT_PATH)
model_client = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(model_client)


def _args(**overrides):
    defaults = dict(diarize=False, model=None, language=None, format=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_diarize_sent_as_false_when_flag_omitted():
    options = model_client.build_transcribe_options(_args())
    assert options["diarize"] == "false"


def test_diarize_sent_as_true_when_flag_given():
    options = model_client.build_transcribe_options(_args(diarize=True))
    assert options["diarize"] == "true"


def test_other_options_forwarded():
    options = model_client.build_transcribe_options(
        _args(model="base", language="en", format="srt")
    )
    assert options == {
        "diarize": "false",
        "model": "base",
        "language": "en",
        "format": "srt",
    }
