# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, MLP, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class MlpAdaptModel(nn.Module):
    """History-encoder actor with privileged-target reconstruction loss.

    This model is designed for IsaacLab/RSL-RL v5 TensorDict observations:
    - The actor input is a flattened history vector coming from a dedicated observation group (e.g. ``actor_history``).
    - The reconstruction supervision target is read from a separate observation group (e.g. ``privileged_target``),
      passed via the optional ``privileged_obs`` argument.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        max_length: int = 5,
        cmd_dim: int = 4,
        latent_dim: int = 32,
        privileged_target_key: str = "privileged_target",
        privileged_target_dim: int = 3,
        history_term_dims: tuple[int, ...] | list[int] | None = None,
        mlp_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        actor_hidden_dims: tuple[int, ...] | list[int] = (256, 128, 32),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        **__: Any,
    ) -> None:
        super().__init__()

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        if max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {max_length}")
        if self.obs_dim % max_length != 0:
            raise ValueError(
                f"Actor-history obs_dim ({self.obs_dim}) must be divisible by max_length ({max_length}). "
                "Check observation history flattening."
            )

        self.max_length = int(max_length)
        self.cmd_dim = int(cmd_dim)
        self.privileged_target_key = str(privileged_target_key)
        self.privileged_target_dim = int(privileged_target_dim)

        self.obs_per_step = int(self.obs_dim // self.max_length)
        if self.cmd_dim <= 0 or self.cmd_dim > self.obs_per_step:
            raise ValueError(f"cmd_dim must be in [1, obs_per_step], got {self.cmd_dim} vs {self.obs_per_step}")
        self.proprioception_dim = int(self.obs_per_step - self.cmd_dim)
        self.history_term_dims, self.num_command_terms = self._resolve_history_term_dims(history_term_dims)

        # Observation normalization (applied to flattened history vector).
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        # Distribution head (optional).
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            action_head_out_dim = self.distribution.input_dim
        else:
            self.distribution = None
            action_head_out_dim = output_dim

        # Memory encoder (history -> latent).
        self.mem_encoder = nn.Sequential(
            *MLP(self.proprioception_dim * self.max_length, latent_dim, mlp_hidden_dims, activation, "tanh")
        )

        # Privileged target estimator (latent -> privileged target prediction).
        self.state_estimator = nn.Sequential(
            *MLP(latent_dim, self.privileged_target_dim, [64, 32], activation)
        )

        # Low-level action network (latent + cmd + privileged_pred -> actions).
        self.low_level_net = nn.Sequential(
            *MLP(latent_dim + self.cmd_dim + self.privileged_target_dim, action_head_out_dim, actor_hidden_dims, activation)
        )

        # Cached reconstruction loss for PPO to query.
        self._privileged_recon_loss = torch.tensor(0.0)
        # Last predicted privileged target (e.g. base_lin_vel) after forward; for play/debug tooling.
        self._last_privileged_pred: torch.Tensor | None = None

        # Initialize distribution-specific MLP weights.
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.low_level_net)  # type: ignore[arg-type]

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        privileged_obs: TensorDict | None = None,
        **_: Any,
    ) -> torch.Tensor:
        # Unpad if needed (non-recurrent model).
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs

        # Build flattened history input.
        hist_flat = self.get_latent(obs)
        hist_flat = self.obs_normalizer(hist_flat)

        pro_obs_hist, cmd = self._split_history(hist_flat)

        mem = self.mem_encoder(pro_obs_hist)
        privileged_pred = self.state_estimator(mem)
        self._last_privileged_pred = privileged_pred.detach()

        # Reconstruction loss (if privileged target is available).
        target_obs = privileged_obs if privileged_obs is not None else obs
        self._privileged_recon_loss = self._compute_recon_loss(target_obs, privileged_pred)

        # Actions.
        action_head_in = torch.cat((mem, cmd, privileged_pred), dim=-1)
        action_head_out = self.low_level_net(action_head_in)

        if self.distribution is not None:
            self.distribution.update(action_head_out)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(action_head_out)
        return action_head_out

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        # Select and concatenate observations from the configured groups.
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        return torch.cat(obs_list, dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            hist_flat = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(hist_flat)  # type: ignore[attr-defined]

    def compute_adaptation_pred_loss(self) -> torch.Tensor:
        return self._privileged_recon_loss

    def as_jit(self) -> nn.Module:
        # Export uses deterministic forward without privileged targets.
        return _TorchMlpAdaptModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxMlpAdaptModel(self, verbose)

    def _compute_recon_loss(self, privileged_obs: TensorDict | None, privileged_pred: torch.Tensor) -> torch.Tensor:
        if privileged_obs is None:
            return privileged_pred.new_zeros(())
        if self.privileged_target_key not in privileged_obs.keys():
            return privileged_pred.new_zeros(())
        target = privileged_obs[self.privileged_target_key]
        if target.dim() != 2 or target.shape[-1] != self.privileged_target_dim:
            raise ValueError(
                f"Expected privileged target '{self.privileged_target_key}' with shape [N, {self.privileged_target_dim}], "
                f"got {tuple(target.shape)}"
            )
        # Match old implementation scaling: 2 * mse
        return 2.0 * (privileged_pred - target.detach()).pow(2).mean()

    def _split_history(self, hist_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.history_term_dims is None:
            x = hist_flat.view(hist_flat.shape[0], self.max_length, self.obs_per_step)
            pro_obs_seq = x[..., : self.proprioception_dim]
            cmd = x[:, -1, self.proprioception_dim : self.proprioception_dim + self.cmd_dim]
            return pro_obs_seq.reshape(pro_obs_seq.shape[0], -1), cmd

        pro_terms = []
        offset = 0
        command_start = len(self.history_term_dims) - self.num_command_terms
        for term_dim in self.history_term_dims[:command_start]:
            term_span = term_dim * self.max_length
            pro_terms.append(hist_flat[:, offset : offset + term_span])
            offset += term_span

        cmd_terms = []
        for term_dim in self.history_term_dims[command_start:]:
            term_span = term_dim * self.max_length
            cmd_hist = hist_flat[:, offset : offset + term_span].view(hist_flat.shape[0], self.max_length, term_dim)
            cmd_terms.append(cmd_hist[:, -1, :])
            offset += term_span
        return torch.cat(pro_terms, dim=-1), torch.cat(cmd_terms, dim=-1)

    def _resolve_history_term_dims(
        self, history_term_dims: tuple[int, ...] | list[int] | None
    ) -> tuple[tuple[int, ...] | None, int]:
        if history_term_dims is None:
            return None, 1
        dims = tuple(int(dim) for dim in history_term_dims)
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"history_term_dims must contain positive dimensions, got {dims}")
        if sum(dims) != self.obs_per_step:
            raise ValueError(
                f"history_term_dims must sum to obs_per_step ({self.obs_per_step}), got {dims} with sum {sum(dims)}"
            )
        command_dim = 0
        num_command_terms = 0
        for dim in reversed(dims):
            command_dim += dim
            num_command_terms += 1
            if command_dim >= self.cmd_dim:
                break
        if command_dim != self.cmd_dim:
            raise ValueError(f"Trailing history command terms must sum to cmd_dim ({self.cmd_dim}), got {command_dim}")
        if num_command_terms == len(dims):
            raise ValueError("history_term_dims must contain at least one non-command term")
        return dims, num_command_terms

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"MlpAdaptModel expects 1D (2D batched) observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchMlpAdaptModel(nn.Module):
    """Exportable deterministic actor (JIT)."""

    def __init__(self, model: MlpAdaptModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mem_encoder = copy.deepcopy(model.mem_encoder)
        self.state_estimator = copy.deepcopy(model.state_estimator)
        self.low_level_net = copy.deepcopy(model.low_level_net)
        self.max_length = model.max_length
        self.obs_per_step = model.obs_per_step
        self.proprioception_dim = model.proprioception_dim
        self.cmd_dim = model.cmd_dim
        self.term_major_history = model.history_term_dims is not None
        self.pro_history_dim = model.proprioception_dim * model.max_length
        cmd_indices: list[int] = []
        if model.history_term_dims is not None:
            command_start = len(model.history_term_dims) - model.num_command_terms
            self.pro_history_dim = sum(model.history_term_dims[:command_start]) * model.max_length
            offset = self.pro_history_dim
            for term_dim in model.history_term_dims[command_start:]:
                cmd_indices.extend(range(offset + (model.max_length - 1) * term_dim, offset + model.max_length * term_dim))
                offset += term_dim * model.max_length
        self.register_buffer("_cmd_indices", torch.tensor(cmd_indices, dtype=torch.long), persistent=False)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        pro_obs_hist, cmd = self._split_history(x)
        mem = self.mem_encoder(pro_obs_hist)
        privileged_pred = self.state_estimator(mem)
        out = self.low_level_net(torch.cat((mem, cmd, privileged_pred), dim=-1))
        return self.deterministic_output(out)

    def _split_history(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.term_major_history:
            x = x.view(x.shape[0], self.max_length, self.obs_per_step)
            pro_obs_seq = x[..., : self.proprioception_dim]
            cmd = x[:, -1, self.proprioception_dim : self.proprioception_dim + self.cmd_dim]
            return pro_obs_seq.reshape(pro_obs_seq.shape[0], -1), cmd

        pro_obs_hist = x[:, : self.pro_history_dim]
        cmd = x.index_select(1, self._cmd_indices)
        return pro_obs_hist, cmd

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxMlpAdaptModel(nn.Module):
    """Exportable deterministic actor (ONNX)."""

    is_recurrent: bool = False

    def __init__(self, model: MlpAdaptModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mem_encoder = copy.deepcopy(model.mem_encoder)
        self.state_estimator = copy.deepcopy(model.state_estimator)
        self.low_level_net = copy.deepcopy(model.low_level_net)
        self.max_length = model.max_length
        self.obs_per_step = model.obs_per_step
        self.proprioception_dim = model.proprioception_dim
        self.cmd_dim = model.cmd_dim
        self.term_major_history = model.history_term_dims is not None
        self.pro_history_dim = model.proprioception_dim * model.max_length
        cmd_indices: list[int] = []
        if model.history_term_dims is not None:
            command_start = len(model.history_term_dims) - model.num_command_terms
            self.pro_history_dim = sum(model.history_term_dims[:command_start]) * model.max_length
            offset = self.pro_history_dim
            for term_dim in model.history_term_dims[command_start:]:
                cmd_indices.extend(range(offset + (model.max_length - 1) * term_dim, offset + model.max_length * term_dim))
                offset += term_dim * model.max_length
        self.register_buffer("_cmd_indices", torch.tensor(cmd_indices, dtype=torch.long), persistent=False)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

        self.input_names = ["obs"]
        self.output_names = ["act"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        pro_obs_hist, cmd = self._split_history(x)
        mem = self.mem_encoder(pro_obs_hist)
        privileged_pred = self.state_estimator(mem)
        out = self.low_level_net(torch.cat((mem, cmd, privileged_pred), dim=-1))
        return self.deterministic_output(out)

    def _split_history(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.term_major_history:
            x = x.view(x.shape[0], self.max_length, self.obs_per_step)
            pro_obs_seq = x[..., : self.proprioception_dim]
            cmd = x[:, -1, self.proprioception_dim : self.proprioception_dim + self.cmd_dim]
            return pro_obs_seq.reshape(pro_obs_seq.shape[0], -1), cmd

        pro_obs_hist = x[:, : self.pro_history_dim]
        cmd = x.index_select(1, self._cmd_indices)
        return pro_obs_hist, cmd

    def get_dummy_inputs(self) -> torch.Tensor:
        return torch.zeros(1, self.input_size)
