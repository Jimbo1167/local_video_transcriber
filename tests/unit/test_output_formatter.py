from src.config import Config
from src.output.formatter import OutputFormatter


def test_pretty_output_merges_adjacent_segments_by_speaker(tmp_path):
    config = Config(output_format="pretty")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "pretty.txt"

    segments = [
        (0.0, 1.5, "I know she would have been honored", "SPEAKER_00"),
        (1.6, 3.0, "to receive this award.", "SPEAKER_00"),
        (6.5, 8.0, "Thank you.", "SPEAKER_00"),
    ]

    formatter.save_transcript(segments, str(output_path))
    content = output_path.read_text(encoding="utf-8")

    assert "I know she would have been honored to receive this award." in content
    assert "Thank you." in content
    assert content.count("SPEAKER_00") == 2


def test_pretty_output_keeps_speaker_breaks(tmp_path):
    config = Config(output_format="pretty")
    formatter = OutputFormatter(config)
    output_path = tmp_path / "pretty.txt"

    segments = [
        (0.0, 1.0, "Hello there.", "SPEAKER_00"),
        (1.1, 2.0, "Hi.", "SPEAKER_01"),
    ]

    formatter.save_transcript(segments, str(output_path))
    content = output_path.read_text(encoding="utf-8")

    assert "SPEAKER_00" in content
    assert "SPEAKER_01" in content
    assert "Hello there." in content
    assert "Hi." in content
