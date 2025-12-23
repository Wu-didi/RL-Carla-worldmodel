from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DreamerEncoder(nn.Module):
    """Simple CNN encoder that follows Dreamer-style latent learning."""

    def __init__(self, latent_dim: int, proprio_dim: int, image_size: tuple[int, int]):
        super().__init__()
        self.image_size = image_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        conv_h = image_size[0] // 16
        conv_w = image_size[1] // 16
        conv_out = 256 * conv_h * conv_w
        self.fc = nn.Sequential(
            nn.Linear(conv_out + proprio_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        x = x.view(image.size(0), -1)
        if proprio is not None:
            x = torch.cat([x, proprio], dim=-1)
        return self.fc(x)


class RSSMTransition(nn.Module):
    """GRU transition that predicts the next latent given action and current latent."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prior = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gru_in = torch.cat([latent, action], dim=-1)
        hidden_state = self.gru(gru_in, hidden_state)
        prior_latent = self.prior(hidden_state)
        return prior_latent, hidden_state


class DreamerWorldModel(nn.Module):
    """Lightweight Dreamer-style latent world model with self-supervised rollout loss."""

    def __init__(
        self,
        image_size: tuple[int, int],
        latent_dim: int,
        action_dim: int,
        proprio_dim: int,
        rssm_hidden_dim: int,
        lr: float,
        device: torch.device,
    ):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.encoder = DreamerEncoder(latent_dim=latent_dim, proprio_dim=proprio_dim, image_size=image_size)
        self.transition = RSSMTransition(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=rssm_hidden_dim)
        self.rssm_hidden_dim = rssm_hidden_dim
        # Move modules to device before creating optimizer to avoid dtype/device mismatches
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def init_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.rssm_hidden_dim, device=self.device)

    def _prep_image(self, rgb: np.ndarray) -> torch.Tensor:
        if rgb is None:
            raise ValueError("RGB frame is None; camera not initialized.")
        if rgb.ndim == 3:
            rgb = rgb[None, ...]
        img = torch.as_tensor(rgb, dtype=torch.float32, device=self.device) / 255.0
        # NHWC -> NCHW
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"RGB shape expected (..., H, W, 3), got {tuple(img.shape)}")
        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, size=self.image_size, mode="bilinear", align_corners=False)
        return img

    def encode(self, rgb: np.ndarray, proprio: np.ndarray) -> torch.Tensor:
        img = self._prep_image(rgb)
        proprio_t = torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
        if proprio_t.ndim == 1:
            proprio_t = proprio_t.unsqueeze(0)
        return self.encoder(img, proprio_t)

    def predict(
        self,
        latent: torch.Tensor,
        action: np.ndarray | torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        hidden_state = hidden_state.detach()
        prior, hidden_state = self.transition(latent, act, hidden_state)
        return prior, hidden_state

    def optimize_world(self, pred_latent: torch.Tensor, target_latent: torch.Tensor) -> float:
        loss = F.mse_loss(pred_latent, target_latent.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def agent_state(self, latent: torch.Tensor, proprio: np.ndarray) -> np.ndarray:
        proprio = np.asarray(proprio, dtype=np.float32)
        if latent.dim() == 2:
            lat_np = latent.detach().cpu().numpy()
        else:
            lat_np = latent.detach().cpu().view(1, -1).numpy()
        lat_np = lat_np.reshape(lat_np.shape[0], -1)
        proprio = proprio.reshape(lat_np.shape[0], -1)
        return np.concatenate([lat_np, proprio], axis=-1)

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
