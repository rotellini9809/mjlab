from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LatentModelConfig:
  obs_dim: int
  act_dim: int
  k_future: int
  z_dim: int = 32
  hidden_dim: int = 256
  encoder_hidden_dim: int | None = None
  decoder_hidden_dim: int | None = None


class MLP(nn.Module):
  def __init__(
    self, in_dim: int, out_dim: int, hidden_dim: int, depth: int = 2
  ) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
      layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
      d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class GaussianEncoder(nn.Module):
  """q(z | future_obs)."""

  def __init__(self, obs_dim: int, k_future: int, z_dim: int, hidden_dim: int) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.k_future = k_future
    self.z_dim = z_dim
    in_dim = obs_dim * k_future
    self.backbone = MLP(in_dim=in_dim, out_dim=2 * z_dim, hidden_dim=hidden_dim, depth=2)

  def forward(self, obs_future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    obs_future: [B, k_future, obs_dim]
    returns: mu [B, z_dim], logvar [B, z_dim]
    """
    b, k, d = obs_future.shape
    assert k == self.k_future and d == self.obs_dim
    x = obs_future.reshape(b, k * d)
    h = self.backbone(x)
    mu, logvar = torch.chunk(h, 2, dim=-1)
    return mu, logvar


class ActionDecoder(nn.Module):
  """pi(a | obs_t, z)."""

  def __init__(self, obs_dim: int, z_dim: int, act_dim: int, hidden_dim: int) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.z_dim = z_dim
    self.act_dim = act_dim
    self.net = MLP(in_dim=obs_dim + z_dim, out_dim=act_dim, hidden_dim=hidden_dim, depth=2)

  def forward(self, obs_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    x = torch.cat([obs_t, z], dim=-1)
    return self.net(x)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
  # z = mu + std * eps
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return mu + std * eps


def kl_to_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
  """
  KL( N(mu, sigma) || N(0, I) ) for diagonal Gaussian.
  returns: [B] (per-sample KL)
  """
  return 0.5 * torch.sum(torch.exp(logvar) + mu * mu - 1.0 - logvar, dim=-1)


class LatentMotorPrimitive(nn.Module):
  def __init__(self, cfg: LatentModelConfig) -> None:
    super().__init__()
    enc_h = cfg.encoder_hidden_dim or cfg.hidden_dim
    dec_h = cfg.decoder_hidden_dim or cfg.hidden_dim
    self.cfg = cfg
    self.encoder = GaussianEncoder(cfg.obs_dim, cfg.k_future, cfg.z_dim, enc_h)
    self.decoder = ActionDecoder(cfg.obs_dim, cfg.z_dim, cfg.act_dim, dec_h)

  def forward(self, obs_t: torch.Tensor, obs_future: torch.Tensor) -> dict[str, torch.Tensor]:
    mu, logvar = self.encoder(obs_future)
    z = reparameterize(mu, logvar)
    a_pred = self.decoder(obs_t, z)
    kl = kl_to_std_normal(mu, logvar)
    return {"a_pred": a_pred, "z": z, "mu": mu, "logvar": logvar, "kl": kl}


def kl_diag_gaussians(
  mu_q: torch.Tensor,
  logvar_q: torch.Tensor,
  mu_p: torch.Tensor,
  logvar_p: torch.Tensor,
) -> torch.Tensor:
  """KL( N(mu_q, diag(exp(logvar_q))) || N(mu_p, diag(exp(logvar_p))) ), summed over z_dim."""
  var_q = torch.exp(logvar_q)
  var_p = torch.exp(logvar_p)
  return 0.5 * torch.sum(
    logvar_p
    - logvar_q
    + (var_q + (mu_q - mu_p) ** 2) / var_p
    - 1.0,
    dim=-1,
  )


class NPMPLatentMotorPrimitive(nn.Module):
  """Temporal latent model with posterior and learned prior."""

  def __init__(self, cfg: LatentModelConfig) -> None:
    super().__init__()
    self.cfg = cfg
    enc_h = cfg.encoder_hidden_dim or cfg.hidden_dim
    dec_h = cfg.decoder_hidden_dim or cfg.hidden_dim

    in_dim = cfg.k_future * cfg.obs_dim + cfg.z_dim
    self.posterior = MLP(in_dim=in_dim, out_dim=2 * cfg.z_dim, hidden_dim=enc_h, depth=2)
    self.prior = MLP(in_dim=cfg.z_dim, out_dim=2 * cfg.z_dim, hidden_dim=enc_h, depth=2)
    self.decoder = ActionDecoder(cfg.obs_dim, cfg.z_dim, cfg.act_dim, dec_h)

  def forward(
    self, obs_chunk: torch.Tensor, obs_future: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    """
    obs_chunk: [B, T, obs_dim]
    obs_future: [B, T, k_future, obs_dim]
    returns dict with a_pred [B, T, act_dim] and kl_t [B, T].
    """
    b, t, d = obs_chunk.shape
    assert d == self.cfg.obs_dim
    assert obs_future.shape[:3] == (b, t, self.cfg.k_future)

    z_prev = torch.zeros(b, self.cfg.z_dim, device=obs_chunk.device, dtype=obs_chunk.dtype)

    a_preds = []
    kls = []
    mu_qs = []
    logvar_qs = []
    mu_ps = []
    logvar_ps = []
    zs = []

    for step in range(t):
      prior_h = self.prior(z_prev)
      mu_p, logvar_p = torch.chunk(prior_h, 2, dim=-1)

      obs_future_t = obs_future[:, step]  # [B, k_future, obs_dim]
      obs_future_flat = obs_future_t.reshape(b, self.cfg.k_future * self.cfg.obs_dim)
      post_in = torch.cat([obs_future_flat, z_prev], dim=-1)
      post_h = self.posterior(post_in)
      mu_q, logvar_q = torch.chunk(post_h, 2, dim=-1)

      z_t = reparameterize(mu_q, logvar_q)
      a_pred = self.decoder(obs_chunk[:, step], z_t)
      kl_t = kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)

      a_preds.append(a_pred)
      kls.append(kl_t)
      mu_qs.append(mu_q)
      logvar_qs.append(logvar_q)
      mu_ps.append(mu_p)
      logvar_ps.append(logvar_p)
      zs.append(z_t)

      z_prev = z_t

    return {
      "a_pred": torch.stack(a_preds, dim=1),
      "kl_t": torch.stack(kls, dim=1),
      "kl": torch.stack(kls, dim=1),
      "mu_q": torch.stack(mu_qs, dim=1),
      "logvar_q": torch.stack(logvar_qs, dim=1),
      "mu_p": torch.stack(mu_ps, dim=1),
      "logvar_p": torch.stack(logvar_ps, dim=1),
      "z": torch.stack(zs, dim=1),
    }
