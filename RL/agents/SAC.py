import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import os



#attention_bc
class PolicyNet_BC(torch.nn.Module): 
    def __init__(self, hidden_dim, action_dim, action_bound, latent_dim, proprio_dim):
        super().__init__()
        self.attn   = SemanticAttention(latent_dim=latent_dim, proprio_dim=proprio_dim, d_key=64)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(64, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()
        )
        self.fc_mu  = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def get_dist(self, obs):
        z, alpha = self.attn(obs)          # 注意力输出 + 权重
        x = self.backbone(z)
        mu  = torch.tanh(self.fc_mu(x)) * self.action_bound
        std = F.softplus(self.fc_std(x)) + 1e-4
        std = torch.clamp(std, 1e-4, self.action_bound)
        return Normal(mu, std), alpha     # 多返回 alpha，用于可视化


###################### Attention SAC network ######################


class SemanticAttention(torch.nn.Module):
    """Latent + proprio → 64 维，带可解释权重 alpha (B,2)"""
    def __init__(self, latent_dim, proprio_dim, d_key=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.proprio_dim = proprio_dim
        # 投影到同一维度
        self.proj_latent = torch.nn.Linear(latent_dim, d_key)
        self.proj_proprio = torch.nn.Linear(proprio_dim, d_key)
        # 加性注意力
        d_query = d_key
        self.W_q = torch.nn.Linear(d_query, d_key)
        self.W_k = torch.nn.Linear(d_key, d_key, bias=False)
        self.v   = torch.nn.Linear(d_key, 1,  bias=False)

    def forward(self, obs):
        # obs: (B, latent + proprio)
        latent = obs[:, :self.latent_dim].unsqueeze(1)            # (B,1,latent_dim)
        proprio = obs[:, self.latent_dim:].unsqueeze(1)           # (B,1,proprio_dim)

        # 投影到 d_key 并拼接成两个 token
        key = torch.cat([
            self.proj_latent(latent),
            self.proj_proprio(proprio)
        ], dim=1)  # (B, 2, d_key)

        # 加性注意力
        query = key.mean(dim=1)                       # (B,d_key) 全局平均当查询
        q = self.W_q(query).unsqueeze(1)              # (B,1,d_key)
        k = self.W_k(key)                             # (B,2,d_key)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # (B,2)
        alpha  = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B,2,1)
        z = torch.sum(alpha * key, dim=1)             # (B,d_key)
        return z, alpha.squeeze(-1)                   # 返回 64 维向量 + 权重


class PolicyNetContinuousAttn(torch.nn.Module):
    def __init__(self, hidden_dim, action_dim, action_bound, latent_dim, proprio_dim):
        super().__init__()
        self.attn   = SemanticAttention(latent_dim=latent_dim, proprio_dim=proprio_dim, d_key=64)
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(64, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()
        )
        self.fc_mu  = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, obs):
        z, alpha = self.attn(obs)          # 注意力输出 + 权重
        x = self.backbone(z)
        mu  = torch.tanh(self.fc_mu(x)) * self.action_bound
        std = F.softplus(self.fc_std(x)) + 1e-4
        std = torch.clamp(std, 1e-4, self.action_bound)
        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample).sum(-1, keepdim=True)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7).sum(-1, keepdim=True)
        action = action * self.action_bound
        return action, log_prob, alpha     # 多返回 alpha，用于可视化


class QValueNetContinuousAttn(torch.nn.Module):
    def __init__(self, hidden_dim, action_dim, latent_dim, proprio_dim):
        super().__init__()
        self.attn = SemanticAttention(latent_dim=latent_dim, proprio_dim=proprio_dim, d_key=64)
        self.net  = torch.nn.Sequential(
            torch.nn.Linear(64 + action_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        z, _ = self.attn(obs)
        x = torch.cat([z, action], dim=-1)
        return self.net(x)


########################## basic SAC network ##########################

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = self.net(x)
        mu = torch.tanh(self.fc_mu(x)) * self.action_bound
        std = F.softplus(self.fc_std(x))  + 1e-4
        std = torch.clamp(std, 1e-4, self.action_bound)

        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample).sum(dim=-1, keepdim=True)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)   # 输出 Q 值
        )

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        return self.net(cat)

###################### Basic SAC ######################


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return action.detach().cpu().numpy().flatten()


    def take_action_bc(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # 全局 μ、σ
        norm_path = "/home/dt/carla_0.9.13/PythonAPI/EasyCarla-RL-main/example/params_dql/bc_norm_726.npz"

        norm = np.load(norm_path)
        mu_np  = torch.from_numpy(norm["mu"]).to(self.device)
        std_np = torch.from_numpy(norm["std"]).to(self.device).clamp(min=1e-4)
        state = (state - mu_np) / std_np

        action = self.actor(state)[0]
        return action.detach().cpu().numpy().flatten()


    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
 
        # print("states shape:", states.shape)
        # print("actions shape:", actions.shape)
        # print("rewards:", rewards)
        # print("next_states shape:", next_states.shape)
        # print("dones shape:", dones.shape)
        
        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=10.0)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=10.0)
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        return {
        'actor_loss'    : actor_loss.item(),
        'critic_1_loss' : critic_1_loss.item(),
        'critic_2_loss' : critic_2_loss.item(),
        'alpha_loss'    : alpha_loss.item(),
        'alpha'         : self.log_alpha.exp().item(),
        'avg_q'         : torch.min(q1_value, q2_value).mean().item(),
        'entropy'       : -log_prob.mean().item(),
        'actor_grad'    : torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1e6).item(),
        'critic_grad'   : torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1e6).item(),
        }



    def save_model(self, dir, id=None): #new way to load and save
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,                 # ← 温度系数值
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            }
        torch.save(ckpt, f"{dir}/sac_ckpt_{id}.pt") # type: ignore


    def load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(ckpt["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(ckpt["critic_2_optimizer"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        self.log_alpha_optimizer.load_state_dict(ckpt["log_alpha_optimizer"])


    def load_model_only_basic(self, dir, id=None):
        ckpt_path = os.path.join(dir, f"sac_ckpt_{id}.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])

    def load_model_path(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_1_optimizer" in ckpt and "critic_2_optimizer" in ckpt:
            self.critic_1_optimizer.load_state_dict(ckpt["critic_1_optimizer"])
            self.critic_2_optimizer.load_state_dict(ckpt["critic_2_optimizer"])
        if "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"])
        if "log_alpha_optimizer" in ckpt:
            self.log_alpha_optimizer.load_state_dict(ckpt["log_alpha_optimizer"])


    def load_bc_model(self, dir, id=None):
        ckpt_path = os.path.join(dir, f"bc_ckpt_{id}.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["bc_policy"], strict=False)
        print('Successfully load bc network')


###################### Attention SAC ######################
class SACContinuous_attention_bc:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, hidden_dim, action_dim, action_bound,
                 device, gamma, tau, actor_lr, critic_lr, alpha_lr,
                 target_entropy, latent_dim, proprio_dim):
        self.actor = PolicyNetContinuousAttn(
            hidden_dim, action_dim, action_bound, latent_dim, proprio_dim
        ).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuousAttn(
            hidden_dim, action_dim, latent_dim, proprio_dim
        ).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuousAttn(
            hidden_dim, action_dim, latent_dim, proprio_dim
        ).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuousAttn(
            hidden_dim, action_dim, latent_dim, proprio_dim
        ).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuousAttn(
            hidden_dim, action_dim, latent_dim, proprio_dim
        ).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return action.detach().cpu().numpy().flatten()


    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob, _ = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
 
        # print("states shape:", states.shape)
        # print("actions shape:", actions.shape)
        # print("rewards:", rewards)
        # print("next_states shape:", next_states.shape)
        # print("dones shape:", dones.shape)
        
        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=10.0)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=10.0)
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob, atten_alpha = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        attention_a = atten_alpha.detach().cpu().numpy().mean(axis=0)   # (41,)
        attention_a = (attention_a - attention_a.min()) / (attention_a.max() + 1e-8)
        attention_img = torch.from_numpy(attention_a.astype(np.float32))[None, None, :]


        return {
        'actor_loss'    : actor_loss.item(),
        'critic_1_loss' : critic_1_loss.item(),
        'critic_2_loss' : critic_2_loss.item(),
        'alpha_loss'    : alpha_loss.item(),
        'alpha'         : self.log_alpha.exp().item(),
        'avg_q'         : torch.min(q1_value, q2_value).mean().item(),
        'entropy'       : -log_prob.mean().item(),
        'actor_grad'    : torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1e6).item(),
        'critic_grad'   : torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 1e6).item(),
        'attention_img' : attention_img,
        }



    def save_model(self, dir, id=None): #new way to load and save
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,                 # ← 温度系数值
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            }
        torch.save(ckpt, f"{dir}/sac_ckpt_{id}.pt") # type: ignore


    def load_model(self, dir, id=None):
        ckpt_path = os.path.join(dir, f"sac_ckpt_{id}.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(ckpt["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(ckpt["critic_2_optimizer"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        self.log_alpha_optimizer.load_state_dict(ckpt["log_alpha_optimizer"])


    def load_model_only_basic(self, dir, id=None):
        ckpt_path = os.path.join(dir, f"sac_ckpt_{id}.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])

    def load_model_path(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.target_critic_1.load_state_dict(ckpt["target_critic_1"])
        self.target_critic_2.load_state_dict(ckpt["target_critic_2"])
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_1_optimizer" in ckpt and "critic_2_optimizer" in ckpt:
            self.critic_1_optimizer.load_state_dict(ckpt["critic_1_optimizer"])
            self.critic_2_optimizer.load_state_dict(ckpt["critic_2_optimizer"])
        if "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"])
        if "log_alpha_optimizer" in ckpt:
            self.log_alpha_optimizer.load_state_dict(ckpt["log_alpha_optimizer"])


    def load_bc_model(self, dir, id=None):
        ckpt_path = os.path.join(dir, f"bc_ckpt_{id}.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["bc_policy"], strict=False)
        print('Successfully load bc network')


def SAC_setup(state_dim, hidden_dim, action_dim, action_bound, device, \
              gamma=0.99, tau=0.05, actor_lr=1e-4, critic_lr=4e-4, \
              alpha_lr=1e-4, target_entropy=-3, network = 'SAC', \
              latent_dim=None, proprio_dim=None):
    if network == 'SAC':
        sacagent = SACContinuous(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            device=device,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            )
        return sacagent
    
    elif network == 'Attention_SAC':
        if latent_dim is None or proprio_dim is None:
            raise ValueError("latent_dim and proprio_dim are required for Attention_SAC.")
        attasacagent = SACContinuous_attention_bc(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            device=device,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            latent_dim=latent_dim,
            proprio_dim=proprio_dim,
            )
        return attasacagent
    else:
        raise ValueError(f"Unknown agent type: {network}")
    
