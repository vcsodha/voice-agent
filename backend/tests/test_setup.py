import os

import pytest

import config
import main


@pytest.fixture
def voice_file(tmp_path, monkeypatch):
    path = tmp_path / "voice.onnx"
    path.write_text("pretend model")
    monkeypatch.setattr(config, "PIPER_MODEL_PATH", str(path))
    return path


def test_setup_passes_when_everything_is_there(monkeypatch, voice_file):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    main.check_setup()   # no error


def test_missing_api_key_is_explained(monkeypatch, voice_file):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    with pytest.raises(RuntimeError, match=r"ANTHROPIC_API_KEY.*\.env"):
        main.check_setup()


def test_placeholder_key_from_env_example_is_rejected(monkeypatch, voice_file):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-paste-your-key-here")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main.check_setup()


def test_missing_voice_is_explained_with_the_download_command(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(config, "PIPER_MODEL_PATH", str(tmp_path / "nope.onnx"))
    with pytest.raises(RuntimeError, match="piper.download_voices"):
        main.check_setup()


def test_every_problem_is_reported_at_once(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "PIPER_MODEL_PATH", str(tmp_path / "nope.onnx"))
    with pytest.raises(RuntimeError) as err:
        main.check_setup()
    assert "ANTHROPIC_API_KEY" in str(err.value)
    assert "piper.download_voices" in str(err.value)


def test_default_paths_are_inside_the_project_folder():
    # so it runs the same locally and in docker without setting env vars
    assert os.path.isdir(os.path.join(config.PROJECT_DIR, "backend"))
    assert os.path.isdir(os.path.join(config.PROJECT_DIR, "frontend"))
    assert os.path.isdir(os.path.join(config.PROJECT_DIR, "data", "knowledge"))
