from pathlib import Path

import pytest

from autocircuit.config import load_config


def test_load_repository_config() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "mvp.toml")
    assert config.model == "pythia-70m"
    assert config.seed == 42
    assert (config.discovery_examples, config.validation_examples, config.test_examples) == (
        256,
        128,
        128,
    )


def test_non_positive_split_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[experiment]\nmodel = "pythia-70m"\nseed = 1\n'
        "[splits]\ndiscovery = 1\nvalidation = 0\ntest = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split sizes"):
        load_config(path)


def test_malformed_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('[experiment]\nmodel = "pythia-70m"\nseed = 1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing configuration section"):
        load_config(path)
