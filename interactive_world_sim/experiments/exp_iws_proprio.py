"""IWS Proprio experiment: trains LatentWorldModelWithProprio on DP demos.

This experiment class bridges the upstream IWS training infrastructure with
the LatentWorldModelWithProprio and DPDemoDataset, enabling
Hydra-based training of the interactive world simulator with proprioceptive
heads on robomimic/RoboCasa expert demonstrations.
"""

from typing import Optional

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf

from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
from interactive_world_sim.datasets.latent_dynamics import (
    DPDemoDataset,
    RealAlohaDataset,
    SimAlohaDataset,
)

from .exp_base import BaseLightningExperiment


class IWSProprioExperiment(BaseLightningExperiment):
    """IWS experiment using LatentWorldModelWithProprio on DP demo data."""

    compatible_algorithms = dict(
        latent_world_model=LatentWorldModel,
        latent_world_model_proprio=LatentWorldModel,  # Overridden in _build_algo
    )

    compatible_datasets = dict(
        dp_demo_dataset=DPDemoDataset,
        sim_aloha_dataset=SimAlohaDataset,
        real_aloha_dataset=RealAlohaDataset,
    )

    # Expected training_stage per experiment config name
    _STAGE_MAP = {
        "exp_iws_proprio_stage1": 1,
        "exp_iws_proprio_stage2": 2,
        "exp_iws_proprio_stage3": 3,
    }

    @staticmethod
    def check_training_stage(exp_name: str, actual_stage: int) -> None:
        """Validate training_stage matches experiment config.

        Raises ValueError if a stage-specific experiment config is selected
        but algorithm.training_stage does not match. No-op for non-stage configs.

        Args:
            exp_name: The experiment._name from Hydra config.
            actual_stage: The algorithm.training_stage value.
        """
        stage_map = IWSProprioExperiment._STAGE_MAP
        if exp_name not in stage_map:
            return
        expected = stage_map[exp_name]
        if actual_stage != expected:
            raise ValueError(
                f"Training stage mismatch: experiment config '{exp_name}' "
                f"expects algorithm.training_stage={expected}, but got "
                f"algorithm.training_stage={actual_stage}. "
                f"Fix: add 'algorithm.training_stage={expected}' to your "
                f"Hydra overrides."
            )

    def _validate_training_stage(self) -> None:
        """Guard: verify algorithm.training_stage matches experiment config."""
        exp_name = self.root_cfg.experiment._name  # noqa
        actual = self.root_cfg.algorithm.get("training_stage", 1)
        self.check_training_stage(exp_name, actual)

    def _build_algo(self) -> pl.LightningModule:
        """Build the model, wiring DPObsContract for proprio if enabled."""
        self._validate_training_stage()
        algo_name = self.root_cfg.algorithm._name  # noqa

        if algo_name == "latent_world_model_proprio":
            # Import our custom model and contract builder
            from offline_red_teaming.utils.iws_latent_world_model_proprio import (
                LatentWorldModelWithProprio,
            )
            from offline_red_teaming.utils.dp_obs_contract import DPObsContract

            # Build DPObsContract from dataset shape_meta + algorithm fields
            dataset_cfg = self.root_cfg.dataset
            shape_meta = OmegaConf.to_container(dataset_cfg.shape_meta, resolve=True)

            # Construct env_runner-like dict from dataset/algorithm config
            env_runner = {
                "n_obs_steps": dataset_cfg.get("n_obs_steps", 1),
                "n_action_steps": dataset_cfg.get("n_action_steps", dataset_cfg.horizon),
                "max_steps": dataset_cfg.get("max_steps", 400),
                "render_obs_key": dataset_cfg.obs_keys[0],
            }

            contract = DPObsContract.from_shape_meta(shape_meta, env_runner)

            # Instantiate model with contract
            return LatentWorldModelWithProprio(
                self.root_cfg.algorithm,
                proprio_contract=contract,
                log_every_n_steps=self.cfg.training.log_every_n_steps,
            )
        else:
            # Fallback to base behavior for standard latent_world_model
            return super()._build_algo()

    @staticmethod
    def _resolve_dataset_class(name: str, compatible_datasets: dict):
        """Resolve dataset class from name with dp_demo_* prefix fallback.

        Handles run_local's runtime-choice overwrite: main.py sets
        cfg.dataset._name to the Hydra group choice key (e.g.
        'dp_demo_robomimic_can') instead of the inner _name field
        ('dp_demo_dataset'). We use a dp_demo_* prefix fallback.

        Args:
            name: The dataset._name value (possibly overwritten by run_local).
            compatible_datasets: Dict mapping exact names to dataset classes.

        Returns:
            Dataset class to instantiate.

        Raises:
            KeyError: If name is not recognized.
        """
        if name in compatible_datasets:
            return compatible_datasets[name]
        elif name.startswith("dp_demo_"):
            return DPDemoDataset
        else:
            raise KeyError(
                f"Unknown dataset '{name}'. Known exact keys: "
                f"{list(compatible_datasets.keys())}. "
                f"Names starting with 'dp_demo_' are also supported."
            )

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        """Build the dataset for a given split."""
        if not hasattr(self, "dataset"):
            name = self.root_cfg.dataset._name  # noqa
            ds_cls = self._resolve_dataset_class(name, self.compatible_datasets)
            self.dataset = ds_cls(self.root_cfg.dataset)
        if split == "training":
            return self.dataset
        elif split == "validation":
            return self.dataset.get_validation_dataset()
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
