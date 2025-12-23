import gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
标准的gan模型
'''
class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid()
        )

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)  # 直接拼接连续动作
        return self.fc(x)





class GAIL:
    def __init__(self, state_dim, hidden_dim, action_dim, lr_d, device):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.device = device

    
    def learn(self, expert_s, expert_a, agent_s, agent_a):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_a, dtype=torch.float).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(agent_a, dtype=torch.float).to(self.device)

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)
        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
                expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10.0)
        self.discriminator_optimizer.step()
        return {
        'discriminator_loss'    : discriminator_loss.item(),
        'discriminator_grad'    : torch.nn.utils.\
            clip_grad_norm_(self.discriminator.parameters(), 1e6).item(),
        }


    def get_reward(self, agent_s, agent_a):
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(agent_a, dtype=torch.float).to(self.device)
        if agent_actions.dim() == 0:
            agent_actions = agent_actions.unsqueeze(0)
        agent_prob = self.discriminator(agent_states, agent_actions)
        reward = -torch.log(agent_prob + 0.5).detach().cpu().numpy()
        return reward

    def save_model(self, dir, id=None): #new way to load and save
        ckpt = {
            "discriminator": self.discriminator.state_dict(),
            "opt": self.discriminator_optimizer.state_dict(),
            }
        torch.save(ckpt, f"{dir}/gail_ckpt_{id}.pt") # type: ignore
    

    def load_model(self, dir, id=None):
        ckpt = torch.load(f"{dir}/gail_ckpt_{id}.pt", map_location=self.device)
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.discriminator_optimizer.load_state_dict(ckpt["opt"])


##############################################################################
'''
加入扩散的gan模型
'''
##############################################################################

import torch
import torch.nn as nn
import math

# ------------------------------------------------
# 1. 小型扩散判别器
# ------------------------------------------------
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class MiniDiffusionModel(nn.Module):
    """
    极简 MLP 条件扩散模型
    输入 x = [state; action]，条件 cond = label(0/1)
    输出预测的噪声 epsilon_theta
    """
    def __init__(self, input_dim, hidden_dim=128, n_steps=100):
        super().__init__()
        self.n_steps = n_steps

        # 预计算 schedule
        betas = cosine_beta_schedule(n_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1 - alphas_cumprod))

        # 网络
        self.time_embed = nn.Embedding(n_steps, hidden_dim)
        self.label_embed = nn.Embedding(2, hidden_dim)          # 0/1
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, label, t):
        """
        x:      [B, input_dim]
        label:  [B] 0 or 1
        t:      [B] timestep
        return: [B, input_dim] 预测噪声
        """
        t_emb = self.time_embed(t)
        l_emb = self.label_embed(label)
        cond = t_emb + l_emb
        return self.net(torch.cat([x, cond], dim=-1))


class DiffDiscriminator(nn.Module):
    """
    外部接口与原 Discriminator 完全一致：
    forward(s, a) -> [B,1] scalar reward
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_steps=50, device='cpu'):
        super().__init__()
        self.device = device
        self.input_dim = state_dim + action_dim
        self.n_steps = n_steps
        self.model = MiniDiffusionModel(self.input_dim, hidden_dim, n_steps).to(device)

        # 缓存 schedule
        betas = cosine_beta_schedule(n_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1 - alphas_cumprod).to(device))

    # ------------------------------------------------
    # 对外唯一接口：输入 s,a -> 标量 reward
    # ------------------------------------------------
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        B = x.size(0)
        log_p1 = self._log_prob(x, torch.ones(B, dtype=torch.long, device=self.device))
        log_p0 = self._log_prob(x, torch.zeros(B, dtype=torch.long, device=self.device))
        # log p(label=1|x)
        log_p = torch.log_softmax(torch.stack([log_p0, log_p1], dim=-1), dim=-1)[:, 1]
        return log_p.unsqueeze(-1)

    # ------------------------------------------------
    # 内部：近似 log p(x|label)
    # ------------------------------------------------
    def _log_prob(self, x, label):
        """负 L_simple 作为 log-likelihood 的近似"""
        loss = torch.zeros(x.size(0), device=self.device)
        cnt = 0
        # 每 10 步采一次，减少计算
        for t in range(0, self.n_steps, 10):
            cnt += 1
            t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
            noise = torch.randn_like(x)
            alpha = self.sqrt_alphas_cumprod[t]
            sigma = self.sqrt_one_minus_alphas_cumprod[t]
            x_t = alpha * x + sigma * noise
            pred_noise = self.model(x_t, label, t_batch)
            loss += (pred_noise - noise).pow(2).mean(dim=1)
        return -(loss / cnt)       


# ------------------------------------------------
# 2. DGAIL 主类（基本保持原接口）
# ------------------------------------------------
class DGAIL:
    def __init__(self, state_dim, hidden_dim, action_dim, lr_d, device):
        self.device = device
        # 替换为扩散判别器
        self.discriminator = DiffDiscriminator(
            state_dim, action_dim, hidden_dim, n_steps=50, device=device)
        self.opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)

    # ------------------------------------------------
    # 训练判别器
    # ------------------------------------------------
    def learn(self, expert_s, expert_a, agent_s, agent_a):
        expert_x = torch.cat([
            torch.tensor(expert_s, dtype=torch.float, device=self.device),
            torch.tensor(expert_a, dtype=torch.float, device=self.device)
        ], dim=-1)
        agent_x = torch.cat([
            torch.tensor(agent_s, dtype=torch.float, device=self.device),
            torch.tensor(agent_a, dtype=torch.float, device=self.device)
        ], dim=-1)

        # 扩散损失：专家=1，agent=0
        loss = self._diffusion_loss(expert_x, torch.ones(expert_x.size(0), device=self.device, dtype=torch.long)) + \
               self._diffusion_loss(agent_x, torch.zeros(agent_x.size(0), device=self.device, dtype=torch.long))

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10.0)
        self.opt.step()
        return {'discriminator_loss': loss.item(),
                'discriminator_grad': torch.nn.utils.\
            clip_grad_norm_(self.discriminator.parameters(), max_norm=10.0).item()}

    # ------------------------------------------------
    # 计算奖励（给 rl_module 用）
    # ------------------------------------------------
    def get_reward(self, agent_s, agent_a):
        s = torch.tensor(agent_s, dtype=torch.float).unsqueeze(0).to(self.device)
        a = torch.tensor(agent_a, dtype=torch.float).unsqueeze(0).to(self.device)
        if a.dim() == 0:
            a = a.unsqueeze(0)
        reward = self.discriminator(s, a).item()
        return np.clip((reward+10)/10,0,1)

    # ------------------------------------------------
    # 内部：单条样本的扩散损失
    # ------------------------------------------------
    def _diffusion_loss(self, x, label):
        B = x.size(0)
        t = torch.randint(0, self.discriminator.n_steps, (B,), device=self.device)
        noise = torch.randn_like(x)
        alpha = self.discriminator.sqrt_alphas_cumprod[t]
        sigma = self.discriminator.sqrt_one_minus_alphas_cumprod[t]
        x_t = alpha[:, None] * x + sigma[:, None] * noise
        pred_noise = self.discriminator.model(x_t, label, t)
        return torch.mean((pred_noise - noise) ** 2)
    
    def save_model(self, dir, id=None): #new way to load and save
        ckpt = {
            "discriminator": self.discriminator.state_dict(),
            "opt": self.opt.state_dict(),
            }
        torch.save(ckpt, f"{dir}/gail_ckpt_{id}.pt") # type: ignore

    def load_model(self, dir, id=None):
        ckpt = torch.load(f"{dir}/gail_ckpt_{id}.pt", map_location=self.device)
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.opt.load_state_dict(ckpt["opt"])


##############################################################################
'''
加入扩散的gan模型2.0
'''
##############################################################################


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# # ---------- 1. 调度器 ----------
# def cosine_beta_schedule(timesteps, s=0.008):
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0.0001, 0.9999)

# # ---------- 2. 扩散骨干网络（不变） ----------
# class MiniDiffusionModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim=128, n_steps=100):
#         super().__init__()
#         self.n_steps = n_steps
#         betas = cosine_beta_schedule(n_steps)
#         alphas = 1 - betas
#         alphas_cumprod = torch.cumprod(alphas, 0)
#         self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
#         self.register_buffer('sqrt_one_minus_alphas_cumprod',
#                              torch.sqrt(1 - alphas_cumprod))

#         self.time_embed = nn.Embedding(n_steps, hidden_dim)
#         self.label_embed = nn.Embedding(2, hidden_dim)          # 0/1
#         self.net = nn.Sequential(
#             nn.Linear(input_dim + hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )

#     def forward(self, x, label, t):
#         t_emb = self.time_embed(t)
#         l_emb = self.label_embed(label)
#         cond = t_emb + l_emb
#         return self.net(torch.cat([x, cond], dim=-1))

# # ---------- 3. 孪生条件判别器 ----------
# class DiffDiscriminator(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128, n_steps=50, device='cpu'):
#         super().__init__()
#         self.device = device
#         self.input_dim = state_dim + action_dim
#         self.n_steps = n_steps
#         self.model = MiniDiffusionModel(self.input_dim, hidden_dim, n_steps).to(device)

#         betas = cosine_beta_schedule(n_steps)
#         alphas = 1 - betas
#         alphas_cumprod = torch.cumprod(alphas, 0)
#         self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(device))
#         self.register_buffer('sqrt_one_minus_alphas_cumprod',
#                              torch.sqrt(1 - alphas_cumprod).to(device))

#     # ---------- 推理：一次前向，返回 logit ----------
#     def forward(self, x):
#         B = x.size(0)
#         t = torch.randint(0, self.n_steps, (B,), device=self.device)
#         noise = torch.randn_like(x)
#         alpha = self.sqrt_alphas_cumprod[t]
#         sigma = self.sqrt_one_minus_alphas_cumprod[t]
#         x_t = alpha[:, None] * x + sigma[:, None] * noise

#         # 孪生前向：同时求 c⁺ 与 c⁻
#         eps_plus  = self.model(x_t, torch.ones(B, dtype=torch.long, device=self.device), t)
#         eps_minus = self.model(x_t, torch.zeros(B, dtype=torch.long, device=self.device), t)

#         # DRAIL 式 logit = L⁻ − L⁺
#         l_plus  = (eps_plus  - noise).pow(2).mean(dim=1)   # (B,)
#         l_minus = (eps_minus - noise).pow(2).mean(dim=1)   # (B,)
#         logit = l_minus - l_plus                            # (B,)

#         return logit.unsqueeze(-1)        # (B,1) 未过 sigmoid
    

# # ---------- 4. DRGAIL 主类 ----------
# class DRGAIL:
#     def __init__(self, state_dim, hidden_dim, action_dim, lr_d, device):
#         self.device = device
#         self.discriminator = DiffDiscriminator(
#             state_dim, action_dim, hidden_dim, n_steps=50, device=device)
#         self.opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
#         self.bce = nn.BCEWithLogitsLoss()

#     # ---------- 判别器训练 ----------
#     def learn(self, expert_s, expert_a, agent_s, agent_a):
#         expert_x = torch.cat([
#             torch.tensor(expert_s, dtype=torch.float, device=self.device),
#             torch.tensor(expert_a, dtype=torch.float, device=self.device)
#         ], dim=-1)
#         agent_x = torch.cat([
#             torch.tensor(agent_s, dtype=torch.float, device=self.device),
#             torch.tensor(agent_a, dtype=torch.float, device=self.device)
#         ], dim=-1)

#         expert_logit = self.discriminator(expert_x)   # 直接调 forward 即可
#         agent_logit  = self.discriminator(agent_x)

#         loss = self.bce(expert_logit, torch.ones_like(expert_logit)) + \
#                self.bce(agent_logit, torch.zeros_like(agent_logit))

#         self.opt.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
#         self.opt.step()
#         return {'discriminator_loss': loss.item()}

#     # ---------- 奖励：DRAIL 式 ----------
#     def get_reward(self, agent_s, agent_a):
#         s = torch.as_tensor(agent_s, dtype=torch.float, device=self.device)
#         a = torch.as_tensor(agent_a, dtype=torch.float, device=self.device)
#         if s.ndim == 1:
#              s = s.unsqueeze(0)
#         if a.ndim == 1:
#              a = a.unsqueeze(0)
#         logit = self.discriminator(torch.cat([s, a], -1)).squeeze(-1)   # 仍是 Tensor
#         return float(torch.sigmoid(-logit))                              # 最后再 .item()

#     # ---------- 保存 / 加载 ----------
#     def save_model(self, dir, id=None):
#         ckpt = {
#             "discriminator": self.discriminator.state_dict(),
#             "opt": self.opt.state_dict(),
#         }
#         torch.save(ckpt, f"{dir}/gail_ckpt_{id}.pt")

#     def load_model(self, dir, id=None):
#         ckpt = torch.load(f"{dir}/gail_ckpt_{id}.pt", map_location=self.device)
#         self.discriminator.load_state_dict(ckpt["discriminator"])
#         self.opt.load_state_dict(ckpt["opt"])








def Gan_setup(state_dim, hidden_dim, action_dim, lr_d, device, agent = 'GAIL'):
    if agent == 'GAIL':
        return GAIL(state_dim, hidden_dim, action_dim, lr_d, device=device)
    elif agent =='DGAIL':
        return DGAIL(state_dim, hidden_dim, action_dim, lr_d, device=device)
    elif agent =='DRGAIL':
        return DRGAIL(state_dim, hidden_dim, action_dim, lr_d, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent}")