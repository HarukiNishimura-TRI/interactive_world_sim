"""DP Demo Dataset for IWS training.

Wraps DP expert demonstrations (HDF5) into the IWS training format expected
by LatentWorldModel / LatentWorldModelWithProprio. Each sample is a temporal
window of (obs, action) pairs from a success-only demonstration trajectory.

This dataset plugs directly into the IWS Hydra-based training pipeline via
interactive_world_sim/main.py, consuming the DPObsContract to
determine observation keys and shapes.

Configuration-driven: dataset path, observation keys, resolution, horizon,
etc. are all specified via Hydra YAML configs.
"""

import copy
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

try:
    from .base_dataset import BaseImageDataset
except ImportError:
    # Fallback for direct import (e.g., in tests without full IWS env)
    BaseImageDataset = torch.utils.data.Dataset  # type: ignore[misc,assignment]


class DPDemoDataset(BaseImageDataset):
    """Dataset that loads DP expert demos (HDF5) for IWS training.

    Expected HDF5 structure (robomimic format):
        data/
            demo_0/
                obs/
                    <image_key>: (T, H, W, C) uint8
                    <proprio_key>: (T, D) float32
                actions: (T, action_dim) float32
            demo_1/ ...

    The dataset produces samples of shape:
        obs[key]: (horizon, C, H, W) for images, (horizon, D) for low_dim
        action: (horizon, action_dim)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.dataset_path = cfg.dataset_path
        self.horizon = cfg.horizon
        self.val_horizon: int = cfg.get("val_horizon", cfg.horizon)
        self.obs_keys: List[str] = list(cfg.obs_keys)
        self.low_dim_keys: List[str] = list(cfg.get("low_dim_keys", []))
        self.resolution: int = cfg.resolution
        self.val_ratio: float = cfg.get("val_ratio", 0.1)
        self.seed: int = cfg.get("seed", 42)
        self.pad_before: int = cfg.get("pad_before", 0)
        self.pad_after: int = cfg.get("pad_after", 0)
        self.action_dim: int = cfg.shape_meta.action.shape[0]
        # crop_ratio: resize to resolution/crop_ratio then center-crop to resolution.
        # 1.0 = no crop (robomimic), 0.9 = 10% center crop (RoboCasa CLIP convention).
        self.crop_ratio: float = cfg.get("crop_ratio", 1.0)

        # Load episode boundaries
        self._load_episodes()

        # Create train/val split
        rng = np.random.default_rng(self.seed)
        n_episodes = len(self._episode_lengths)
        n_val = max(1, int(n_episodes * self.val_ratio))
        perm = rng.permutation(n_episodes)
        self._val_indices = set(perm[:n_val].tolist())
        self._train_indices = set(perm[n_val:].tolist())

        # Build index: (episode_idx, start_step) pairs for windowed sampling
        self._train_samples: List[tuple] = []
        self._val_samples: List[tuple] = []
        for ep_idx in range(n_episodes):
            ep_len = self._episode_lengths[ep_idx]
            if ep_idx in self._train_indices:
                max_start = max(0, ep_len - self.horizon)
                for start in range(0, max_start + 1):
                    self._train_samples.append((ep_idx, start))
            else:
                # Validation uses val_horizon; if val_horizon > ep_len, one
                # sample starting at 0 (padding handles the rest)
                max_start = max(0, ep_len - self.val_horizon)
                for start in range(0, max_start + 1):
                    self._val_samples.append((ep_idx, start))

        self._is_val = False
        self._active_horizon = self.horizon  # switches on get_validation_dataset

    def _load_episodes(self) -> None:
        """Scan the HDF5 file to get episode lengths."""
        self._episode_lengths: List[int] = []
        self._demo_keys: List[str] = []

        with h5py.File(self.dataset_path, "r") as f:
            data_grp = f["data"]
            for demo_key in sorted(data_grp.keys()):
                actions = data_grp[demo_key]["actions"]
                self._episode_lengths.append(actions.shape[0])
                self._demo_keys.append(demo_key)

    def __len__(self) -> int:
        if self._is_val:
            return len(self._val_samples)
        return len(self._train_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        samples = self._val_samples if self._is_val else self._train_samples
        ep_idx, start = samples[idx]
        demo_key = self._demo_keys[ep_idx]
        horizon = self._active_horizon
        end = min(start + horizon, self._episode_lengths[ep_idx])
        actual_len = end - start

        obs_dict: Dict[str, torch.Tensor] = {}

        with h5py.File(self.dataset_path, "r") as f:
            demo_grp = f["data"][demo_key]

            # Load image observations
            for key in self.obs_keys:
                raw = demo_grp["obs"][key][start:end]  # (T, H, W, C) uint8
                # Preprocess: center-crop then resize (matching DP convention)
                raw = self._preprocess_images(raw)
                # (T, H, W, C) -> (T, C, H, W) float32 normalized
                obs_tensor = torch.from_numpy(
                    np.moveaxis(raw.astype(np.float32), -1, 1) / 255.0
                )
                # Pad if needed
                if actual_len < horizon:
                    pad = obs_tensor[-1:].expand(horizon - actual_len, -1, -1, -1)
                    obs_tensor = torch.cat([obs_tensor, pad], dim=0)
                obs_dict[key] = obs_tensor

            # Load low-dim observations
            for key in self.low_dim_keys:
                raw = demo_grp["obs"][key][start:end]  # (T, D)
                obs_tensor = torch.from_numpy(raw.astype(np.float32))
                if actual_len < horizon:
                    pad = obs_tensor[-1:].expand(horizon - actual_len, -1)
                    obs_tensor = torch.cat([obs_tensor, pad], dim=0)
                obs_dict[key] = obs_tensor

            # Load actions (truncate to action_dim — RoboCasa may store wider padded actions)
            actions = demo_grp["actions"][start:end, :self.action_dim]  # (T, action_dim)
            action_tensor = torch.from_numpy(actions.astype(np.float32))
            if actual_len < horizon:
                pad = action_tensor[-1:].expand(horizon - actual_len, -1)
                action_tensor = torch.cat([action_tensor, pad], dim=0)

        return {
            "obs": obs_dict,
            "action": action_tensor,
        }

    def _preprocess_images(self, raw: np.ndarray) -> np.ndarray:
        """Preprocess image frames: center-crop then resize to self.resolution.

        Args:
            raw: (T, H, W, C) uint8 image array.

        Returns:
            Preprocessed (T, resolution, resolution, C) uint8 array.
        """
        import cv2

        T, H, W, C = raw.shape
        target = self.resolution

        if self.crop_ratio < 1.0:
            # Resize to larger intermediate then center-crop (DP RoboCasa convention)
            inter_size = round(target / self.crop_ratio)  # e.g. 128/0.9 ≈ 142
            processed = []
            for frame in raw:
                resized = cv2.resize(frame, (inter_size, inter_size), interpolation=cv2.INTER_AREA)
                # Center crop to target
                offset = (inter_size - target) // 2
                cropped = resized[offset:offset + target, offset:offset + target]
                processed.append(cropped)
            return np.stack(processed)
        else:
            # Simple resize (robomimic convention or already correct size)
            if H == target and W == target:
                return raw
            return np.stack([
                cv2.resize(frame, (target, target), interpolation=cv2.INTER_AREA)
                for frame in raw
            ])

    def get_validation_dataset(self) -> "DPDemoDataset":
        """Return a validation split view using val_horizon."""
        val_ds = copy.copy(self)
        val_ds._is_val = True
        val_ds._active_horizon = self.val_horizon
        return val_ds

    def get_normalizer(self, mode: str = "limits", **kwargs):
        """Build a LinearNormalizer for all obs keys and actions.

        Follows the upstream RealAlohaDataset pattern:
        - action: range normalizer (truncated to action_dim)
        - image obs keys: image range normalizer (maps [0,1] -> [-1,1])
        - low_dim keys ending with pos/quat/vel: identity normalizer
        - low_dim keys ending with qpos: range normalizer
        - unknown low_dim keys: identity normalizer (safe default)
        """
        from interactive_world_sim.utils.normalizer import (
            LinearNormalizer,
            array_to_stats,
            get_identity_normalizer_from_stat,
            get_image_range_normalizer,
            get_range_normalizer_from_stat,
        )

        normalizer = LinearNormalizer()

        # Action normalizer (range)
        all_actions = self.get_all_actions().numpy()  # (N, action_dim)
        stat = array_to_stats(all_actions)
        normalizer["action"] = get_range_normalizer_from_stat(stat)

        # Image obs keys: image range normalizer ([0,1] -> [-1,1])
        for key in self.obs_keys:
            normalizer[key] = get_image_range_normalizer()

        # Low-dim obs keys
        for key in self.low_dim_keys:
            all_obs = []
            with h5py.File(self.dataset_path, "r") as f:
                data_grp = f["data"]
                for demo_key in self._demo_keys:
                    obs = data_grp[demo_key]["obs"][key][:]
                    all_obs.append(obs.astype(np.float32))
            all_obs_arr = np.concatenate(all_obs, axis=0)
            stat = array_to_stats(all_obs_arr)

            if key.endswith("pos"):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                normalizer[key] = get_range_normalizer_from_stat(stat)
            elif key.endswith("vel"):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                # Safe default for unknown keys
                normalizer[key] = get_identity_normalizer_from_stat(stat)

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Load all actions for normalizer fitting (truncated to action_dim)."""
        all_actions = []
        with h5py.File(self.dataset_path, "r") as f:
            data_grp = f["data"]
            for demo_key in self._demo_keys:
                actions = data_grp[demo_key]["actions"][:, :self.action_dim]
                all_actions.append(torch.from_numpy(actions.astype(np.float32)))
        return torch.cat(all_actions, dim=0)
