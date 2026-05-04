from __future__ import annotations

from typing import Any

import numpy as np

from uav_benchmark.algorithms.sac_smopso.controller_types import (
    TemporalRelationalStateSpec,
    TensorStructuredStateBatch,
)
from uav_benchmark.algorithms.sac_smopso.torch_support import _TORCH_AVAILABLE, F, Normal, nn, torch

Tensor = Any

if _TORCH_AVAILABLE and torch is not None:

    def _prepare_attention_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
        valid_mask = mask > 0.5
        safe_mask = valid_mask.clone()
        empty_rows = ~torch.any(valid_mask, dim=1)
        if torch.any(empty_rows):
            safe_mask[empty_rows, 0] = True
        return valid_mask, safe_mask

    class _MaskedSelfAttentionBlock(nn.Module):
        def __init__(self, hidden_dim: int, num_heads: int) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.norm2 = nn.LayerNorm(hidden_dim)

        def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
            valid_mask, safe_mask = _prepare_attention_mask(mask)
            key_padding_mask = ~safe_mask
            attn_out, _weights = self.attn(
                tokens, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False
            )
            x = self.norm1(tokens + attn_out)
            x = self.norm2(x + self.ff(x))
            return x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)

    class _LearnedSetEncoder(nn.Module):
        def __init__(self, token_dim: int, hidden_dim: int, num_heads: int = 4, seed_count: int = 2) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.context_proj = nn.Linear(hidden_dim, hidden_dim)
            self.blocks = nn.ModuleList(
                [
                    _MaskedSelfAttentionBlock(hidden_dim, num_heads=num_heads),
                    _MaskedSelfAttentionBlock(hidden_dim, num_heads=num_heads),
                ]
            )
            self.seed_vectors = nn.Parameter(torch.randn(seed_count, hidden_dim) * 0.02)
            self.pool = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
            self.out = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )

        def forward(self, tokens: Tensor, mask: Tensor, context: Tensor) -> Tensor:
            valid_mask, safe_mask = _prepare_attention_mask(mask)
            x = self.input_proj(tokens)
            x = x + self.context_proj(context).unsqueeze(1)
            x = x * safe_mask.unsqueeze(-1).to(dtype=x.dtype)
            for block in self.blocks:
                x = block(x, safe_mask.to(dtype=mask.dtype))
            seeds = self.seed_vectors.unsqueeze(0).expand(tokens.shape[0], -1, -1)
            pooled, _weights = self.pool(seeds, x, x, key_padding_mask=~safe_mask, need_weights=False)
            pooled = torch.mean(pooled, dim=1)
            pooled = self.out(pooled)
            has_any = torch.any(valid_mask, dim=1, keepdim=True).to(dtype=pooled.dtype)
            return pooled * has_any

    class _PooledSetEncoder(nn.Module):
        def __init__(self, token_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.token_proj = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, tokens: Tensor, mask: Tensor, context: Tensor) -> Tensor:
            valid_mask, safe_mask = _prepare_attention_mask(mask)
            token_hidden = self.token_proj(tokens) * safe_mask.unsqueeze(-1).to(dtype=tokens.dtype)
            query_hidden = self.query_proj(context).unsqueeze(1)
            scores = torch.sum(token_hidden * query_hidden, dim=-1) / np.sqrt(max(1, token_hidden.shape[-1]))
            scores = scores.masked_fill(~safe_mask, -1e4)
            weights = torch.softmax(scores, dim=1)
            weights = weights * valid_mask.to(dtype=weights.dtype)
            weights = weights / torch.clamp(torch.sum(weights, dim=1, keepdim=True), min=1e-6)
            pooled = torch.sum(weights.unsqueeze(-1) * token_hidden, dim=1)
            has_any = torch.any(valid_mask, dim=1, keepdim=True).to(dtype=pooled.dtype)
            return pooled * has_any

    class _LearnedTemporalEncoder(nn.Module):
        def __init__(self, token_dim: int, hidden_dim: int, num_heads: int = 4, max_steps: int = 8) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.pos_embedding = nn.Parameter(torch.randn(1, max_steps, hidden_dim) * 0.02)
            self.blocks = nn.ModuleList(
                [
                    _MaskedSelfAttentionBlock(hidden_dim, num_heads=num_heads),
                    _MaskedSelfAttentionBlock(hidden_dim, num_heads=num_heads),
                ]
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.pool = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)

        def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
            valid_mask, safe_mask = _prepare_attention_mask(mask)
            x = self.input_proj(tokens)
            length = x.shape[1]
            x = x + self.pos_embedding[:, :length, :]
            x = x * safe_mask.unsqueeze(-1).to(dtype=x.dtype)
            for block in self.blocks:
                x = block(x, safe_mask.to(dtype=mask.dtype))
            query = self.pool_query.expand(tokens.shape[0], -1, -1)
            pooled, _weights = self.pool(query, x, x, key_padding_mask=~safe_mask, need_weights=False)
            pooled = pooled.squeeze(1)
            has_any = torch.any(valid_mask, dim=1, keepdim=True).to(dtype=pooled.dtype)
            return pooled * has_any

    class _PooledTemporalEncoder(nn.Module):
        def __init__(self, token_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.ReLU(),
            )
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
            valid_mask, safe_mask = _prepare_attention_mask(mask)
            x = self.input_proj(tokens)
            x = x * safe_mask.unsqueeze(-1).to(dtype=x.dtype)
            output, _hidden = self.gru(x)
            lengths = torch.clamp(torch.sum(valid_mask, dim=1).long(), min=1)
            batch_index = torch.arange(tokens.shape[0], device=tokens.device)
            pooled = output[batch_index, lengths - 1]
            has_any = torch.any(valid_mask, dim=1, keepdim=True).to(dtype=pooled.dtype)
            return pooled * has_any

    class _FlatStructuredStateEncoder(nn.Module):
        def __init__(self, spec: TemporalRelationalStateSpec, hidden_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(spec.global_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )

        def forward(self, state_batch: TensorStructuredStateBatch) -> Tensor:
            return self.net(state_batch.global_features)

    class _LearnedStructuredStateEncoder(nn.Module):
        def __init__(self, spec: TemporalRelationalStateSpec, hidden_dim: int) -> None:
            super().__init__()
            self.global_proj = nn.Sequential(
                nn.Linear(spec.global_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.population_pool = _LearnedSetEncoder(spec.population_dim, hidden_dim)
            self.archive_pool = _LearnedSetEncoder(spec.archive_dim, hidden_dim)
            self.topology_pool = _LearnedSetEncoder(spec.topology_dim, hidden_dim)
            self.interaction_pool = _LearnedSetEncoder(spec.interaction_dim, hidden_dim)
            self.environment_pool = _LearnedSetEncoder(spec.environment_dim, hidden_dim)
            self.temporal_pool = _LearnedTemporalEncoder(spec.temporal_dim, hidden_dim)
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 7, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            )

        def forward(self, state_batch: TensorStructuredStateBatch) -> Tensor:
            global_ctx = self.global_proj(state_batch.global_features)
            population_ctx = self.population_pool(
                state_batch.population_tokens, state_batch.population_mask, global_ctx
            )
            archive_ctx = self.archive_pool(state_batch.archive_tokens, state_batch.archive_mask, global_ctx)
            topology_ctx = self.topology_pool(state_batch.topology_tokens, state_batch.topology_mask, global_ctx)
            interaction_ctx = self.interaction_pool(
                state_batch.interaction_tokens,
                state_batch.interaction_mask,
                global_ctx,
            )
            environment_ctx = self.environment_pool(
                state_batch.environment_tokens,
                state_batch.environment_mask,
                global_ctx,
            )
            temporal_ctx = self.temporal_pool(state_batch.temporal_tokens, state_batch.temporal_mask)
            fused = torch.cat(
                [
                    global_ctx,
                    population_ctx,
                    archive_ctx,
                    topology_ctx,
                    interaction_ctx,
                    environment_ctx,
                    temporal_ctx,
                ],
                dim=-1,
            )
            return self.fuse(fused)

    class _HandcraftedStructuredStateEncoder(nn.Module):
        def __init__(self, spec: TemporalRelationalStateSpec, hidden_dim: int) -> None:
            super().__init__()
            self.global_proj = nn.Sequential(
                nn.Linear(spec.global_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.population_pool = _PooledSetEncoder(spec.population_dim, hidden_dim)
            self.archive_pool = _PooledSetEncoder(spec.archive_dim, hidden_dim)
            self.topology_pool = _PooledSetEncoder(spec.topology_dim, hidden_dim)
            self.interaction_pool = _PooledSetEncoder(spec.interaction_dim, hidden_dim)
            self.environment_pool = _PooledSetEncoder(spec.environment_dim, hidden_dim)
            self.temporal_pool = _PooledTemporalEncoder(spec.temporal_dim, hidden_dim)
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 7, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )

        def forward(self, state_batch: TensorStructuredStateBatch) -> Tensor:
            global_ctx = self.global_proj(state_batch.global_features)
            population_ctx = self.population_pool(
                state_batch.population_tokens, state_batch.population_mask, global_ctx
            )
            archive_ctx = self.archive_pool(state_batch.archive_tokens, state_batch.archive_mask, global_ctx)
            topology_ctx = self.topology_pool(state_batch.topology_tokens, state_batch.topology_mask, global_ctx)
            interaction_ctx = self.interaction_pool(
                state_batch.interaction_tokens,
                state_batch.interaction_mask,
                global_ctx,
            )
            environment_ctx = self.environment_pool(
                state_batch.environment_tokens,
                state_batch.environment_mask,
                global_ctx,
            )
            temporal_ctx = self.temporal_pool(state_batch.temporal_tokens, state_batch.temporal_mask)
            fused = torch.cat(
                [
                    global_ctx,
                    population_ctx,
                    archive_ctx,
                    topology_ctx,
                    interaction_ctx,
                    environment_ctx,
                    temporal_ctx,
                ],
                dim=-1,
            )
            return self.fuse(fused)

    def _build_state_encoder(
        state_spec: TemporalRelationalStateSpec,
        hidden_dim: int,
        encoder_mode: str,
    ) -> Any:
        normalized = str(encoder_mode).strip().lower()
        if normalized == "flat":
            return _FlatStructuredStateEncoder(state_spec, hidden_dim)
        if normalized == "handcrafted":
            return _HandcraftedStructuredStateEncoder(state_spec, hidden_dim)
        if normalized == "learned":
            return _LearnedStructuredStateEncoder(state_spec, hidden_dim)
        raise ValueError(f"Unsupported SAC state encoder mode: {encoder_mode}")

    class _Actor(nn.Module):
        def __init__(
            self,
            state_spec: TemporalRelationalStateSpec,
            action_dim: int,
            operator_count: int,
            hidden_dim: int,
            encoder_mode: str,
        ) -> None:
            super().__init__()
            self.encoder = _build_state_encoder(state_spec, hidden_dim, encoder_mode)
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)
            self.logit_head = nn.Linear(hidden_dim, operator_count)

        def forward(self, state_batch: TensorStructuredStateBatch) -> tuple[Tensor, Tensor, Tensor]:
            latent = self.encoder(state_batch)
            mean = self.mean_head(latent)
            log_std = torch.clamp(self.log_std_head(latent), -5.0, 2.0)
            logits = self.logit_head(latent)
            return mean, log_std, logits

        def _disabled_operator_outputs(self, batch_size: int, dtype: Any, device: Any) -> tuple[Tensor, Tensor, Tensor]:
            probs = torch.full(
                (batch_size, self.logit_head.out_features),
                1.0 / max(1, self.logit_head.out_features),
                dtype=dtype,
                device=device,
            )
            operator_id = torch.zeros(batch_size, dtype=torch.long, device=device)
            operator_one_hot = torch.zeros_like(probs)
            return probs, operator_one_hot, operator_id

        def sample(
            self,
            state_batch: TensorStructuredStateBatch,
            deterministic: bool = False,
            *,
            use_operator_head: bool = True,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            mean, log_std, logits = self.forward(state_batch)
            if deterministic:
                continuous = torch.tanh(mean)
                if use_operator_head:
                    probs = torch.softmax(logits, dim=-1)
                    operator_id = torch.argmax(probs, dim=-1)
                    operator_one_hot = F.one_hot(operator_id, num_classes=probs.shape[-1]).to(dtype=continuous.dtype)
                else:
                    probs, operator_one_hot, operator_id = self._disabled_operator_outputs(
                        mean.shape[0],
                        continuous.dtype,
                        continuous.device,
                    )
                log_prob = torch.zeros(mean.shape[0], dtype=mean.dtype, device=mean.device)
                return continuous, operator_one_hot, operator_id, log_prob, probs

            std = torch.exp(log_std)
            normal = Normal(mean, std)
            raw = normal.rsample()
            continuous = torch.tanh(raw)
            log_prob_cont = normal.log_prob(raw) - torch.log(torch.clamp(1.0 - continuous.pow(2), min=1e-6))
            log_prob_cont = torch.sum(log_prob_cont, dim=-1)

            if use_operator_head:
                operator_one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                log_prob_disc = torch.sum(operator_one_hot * torch.log(torch.clamp(probs, min=1e-8)), dim=-1)
                operator_id = torch.argmax(operator_one_hot, dim=-1)
                log_prob = log_prob_cont + log_prob_disc
            else:
                probs, operator_one_hot, operator_id = self._disabled_operator_outputs(
                    mean.shape[0],
                    continuous.dtype,
                    continuous.device,
                )
                log_prob = log_prob_cont
            return continuous, operator_one_hot, operator_id, log_prob, probs

    class _Critic(nn.Module):
        def __init__(
            self,
            state_spec: TemporalRelationalStateSpec,
            action_dim: int,
            operator_count: int,
            hidden_dim: int,
            encoder_mode: str,
        ) -> None:
            super().__init__()
            self.encoder = _build_state_encoder(state_spec, hidden_dim, encoder_mode)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + action_dim + operator_count, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            state_batch: TensorStructuredStateBatch,
            continuous: Tensor,
            operator_one_hot: Tensor,
        ) -> Tensor:
            latent = self.encoder(state_batch)
            features = torch.cat([latent, continuous, operator_one_hot], dim=-1)
            return self.net(features).squeeze(-1)
else:
    _Actor = None  # type: ignore[assignment]
    _Critic = None  # type: ignore[assignment]
