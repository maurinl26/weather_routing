"""Compose la config Hydra (dry-run, sans GPU ni réseau).

Détecte les casses de typage, les clés manquantes, les références ??? non
résolues — l'équivalent du smoke-train CI mais pour l'assimilation.
"""

import os

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

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


def test_fetchers_synthetic_instantiates():
    with initialize_config_dir(version_base="1.3", config_dir=ABS_CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=["fetchers=synthetic_only"],
        )
    fetchers = {name: instantiate(sub) for name, sub in cfg.fetchers.items()}
    assert "synthetic" in fetchers
    assert fetchers["synthetic"].name == "synthetic_ais"
