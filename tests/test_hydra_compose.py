"""Compose la config Hydra (dry-run, sans GPU ni réseau).

Détecte les casses de typage, les clés manquantes, les références ??? non
résolues — l'équivalent du smoke-train CI mais pour l'assimilation.
"""

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

ABS_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))


@pytest.mark.parametrize(
    "experiment,checkpoint_required",
    [("assim_nudging", False), ("assim_enkf", True), ("assim_dps", True)],
)
def test_assim_experiment_composes(experiment, checkpoint_required):
    with initialize_config_dir(version_base="1.3", config_dir=ABS_CONFIG_DIR):
        overrides = [f"experiment={experiment}"]
        if checkpoint_required:
            overrides.append("checkpoint=fake.ckpt")  # juste pour résoudre ???
        cfg = compose(config_name="config", overrides=overrides)
    assert cfg.fetchers is not None
    assert cfg.window.t0 and cfg.window.t1
    assert cfg.assim._target_


@pytest.mark.parametrize(
    "preset,expected_keys",
    [
        ("synthetic_only", {"synthetic"}),
        ("cmems_ascat", {"cmems", "ascat"}),
        ("all_open", {"cmems", "ascat", "vos", "synthetic"}),
    ],
)
def test_fetcher_presets_compose_and_instantiate(preset, expected_keys):
    with initialize_config_dir(version_base="1.3", config_dir=ABS_CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"fetchers={preset}"])
    assert set(cfg.fetchers.keys()) == expected_keys
    for sub in cfg.fetchers.values():
        instantiate(sub)  # ne doit pas lever


def test_all_open_overrides_synthetic_density():
    # Le preset démo densifie les voiliers virtuels mais hérite du reste du leaf.
    with initialize_config_dir(version_base="1.3", config_dir=ABS_CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["fetchers=all_open"])
    assert cfg.fetchers.synthetic.n_vessels == 30
    assert cfg.fetchers.synthetic.speed_ms == 6.0  # hérité du leaf source/synthetic


@pytest.mark.parametrize("experiment", ["assim_enkf", "assim_dps"])
def test_checkpoint_mandatory_for_model_solvers(experiment):
    # Sans checkpoint, le runner doit échouer tôt (??? non résolu) plutôt que
    # de tourner avec un solveur à moitié branché.
    with initialize_config_dir(version_base="1.3", config_dir=ABS_CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.select(cfg, "checkpoint", throw_on_missing=True)


def test_checkpoint_optional_for_nudging():
    with initialize_config_dir(version_base="1.3", config_dir=ABS_CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["experiment=assim_nudging"])
    assert OmegaConf.select(cfg, "checkpoint", throw_on_missing=True) is None
