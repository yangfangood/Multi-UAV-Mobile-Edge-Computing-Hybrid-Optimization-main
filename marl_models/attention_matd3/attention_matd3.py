from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.attention_matd3.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, GaussianNoise, get_state_dict, load_safe
import config
import torch
import torch.nn.functional as F
import numpy as np
from typing import cast
import os


class AttentionMATD3(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        # 5个actor网络，每个智能体一个，10个critic网络，每个无人机两个，用于双Q学习---改进ddpg
        # 网络创建 （策略网络） 输入：自己的观测 输出：动作（二维）[角度比例，距离比例]  功能：决定无人机怎么飞
        self.actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        # 输入：所有无人机的观测+所有无人机的动作---Q值：评估当前状态，动作对的好坏
        self.critics_1: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics_2: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        # 对应目标网络，用于稳定训练
        self.target_actors: list[ActorNetwork] = [ActorNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_1: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics_2: list[CriticNetwork] = [CriticNetwork(obs_dim, action_dim).to(device) for _ in range(num_agents)]
        self._init_target_networks()# 初始化目标网络参数（复制当前网络）
        # 性能优化 把PyTorch代码编译成更高效的机器码 但是被我禁用了
        self.actors = [cast(ActorNetwork, torch.compile(actor)) for actor in self.actors]
        self.critics_1 = [cast(CriticNetwork, torch.compile(critic)) for critic in self.critics_1]
        self.critics_2 = [cast(CriticNetwork, torch.compile(critic)) for critic in self.critics_2]
        # 优化器--最小化损失函数
        self.actor_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(actor.parameters(), lr=config.ACTOR_LR) for actor in self.actors] #5个Actor优化器，lr=3e-4
        self.critic_1_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics_1] # 5个Critic1优化器，lr=3e-4

        self.critic_2_optimizers: list[torch.optim.Adam] = [torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR) for critic in self.critics_2]  # 5个Critic2优化器，lr=3e-4
        #  在训练时给动作加噪声，鼓励探索    避免总是选择相同动作，陷入局部最优
        self.noise: list[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

        # Delayed Updates Counter
        # 作用：记录更新次数，用于实现延迟Actor更新
        self.update_counter: int = 0

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        """
           在训练和测试过程中 被反复调用，负责把观测变成动作
           参数:
                observations: 5个无人机的观测，每个形状(265,)
                exploration: True=训练模式（加噪声探索），False=测试模式（不加噪声）

           返回:
               actions: 形状(5,2)的动作数组
        """
        actions: list[np.ndarray] = []
        with torch.no_grad(): # 关闭梯度计算
            for i, obs in enumerate(observations):
                obs_tensor: torch.Tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) # 将numpy数组转为pytorch张量
                action: np.ndarray = self.actors[i](obs_tensor).squeeze(0).cpu().numpy()

                if exploration: # exploration：控制噪声开关的参数
                    action += self.noise[i].sample() # 给动作加上高斯噪声，让无人机尝试不同的动作
                actions.append(np.clip(action, -1.0, 1.0)) # 剪裁动作范围
        return np.array(actions) # 返回所有动作

    def update(self, batch: ExperienceBatch) -> dict:
        """
        MATD3期望接收OffPolicyExperienceBatch数据结构（一个包含5个元素的元组）
        """

        assert isinstance(batch, tuple) and len(batch) == 5, "MATD3 expects OffPolicyExperienceBatch (tuple of 5 elements)"
        self.update_counter += 1
        # 解包batch
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch
        #转换为tensor 创建张量，开启记忆记忆功能
        obs_tensor: torch.Tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
        rewards_tensor: torch.Tensor = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        next_obs_tensor: torch.Tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        dones_tensor: torch.Tensor = torch.as_tensor(dones_batch, dtype=torch.float32, device=self.device)
        # CRITICAL CHANGE: We DO NOT flatten obs/actions here. 重大变更：我们不会在这里对观测值/动作进行扁平化处理。

        agent_losses: list[float] = []
        agent_critic_losses: list[float] = []
        # 对每个智能体更新critic
        for agent_idx in range(self.num_agents):
            # Update Critic
            with torch.no_grad():
                # Get next actions from target actors and add clipped noise 用目标网络为每个无人机计算下一步动作
                next_actions_list: list[torch.Tensor] = [self.target_actors[i](next_obs_tensor[:, i, :]) for i in range(self.num_agents)] # 得到了下一步动作集合
                # 给动作加噪声，防止Critic对某些动作产生尖锐峰值 并对噪声进行剪裁
                noise: list[torch.Tensor] = [torch.randn_like(next_action_i) * config.TARGET_POLICY_NOISE for next_action_i in next_actions_list]
                clipped_noise: list[torch.Tensor] = [torch.clamp(n, -config.NOISE_CLIP, config.NOISE_CLIP) for n in noise]
                next_actions_list = [torch.clamp(next_actions_list[i] + clipped_noise[i], -1.0, 1.0) for i in range(self.num_agents)] # 在次更新（加了噪声之后）
                # 转换为tensor 创建张量，开启记忆功能
                next_actions_tensor: torch.Tensor = torch.stack(next_actions_list, dim=1)
                # Compute target Q-value using the minimum of the two target critics 用两个目标critic计算q值，取最小值
                target_q1: torch.Tensor = self.target_critics_1[agent_idx](next_obs_tensor, next_actions_tensor, agent_idx)
                target_q2: torch.Tensor = self.target_critics_2[agent_idx](next_obs_tensor, next_actions_tensor, agent_idx)
                target_q_min: torch.Tensor = torch.min(target_q1, target_q2)
                # 贝尔曼方程计算目标值
                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_done: torch.Tensor = dones_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q_min * (1 - agent_done)

            # Update both critic networks 更新网络的时候要开梯度

            current_q1: torch.Tensor = self.critics_1[agent_idx](obs_tensor, actions_tensor, agent_idx) # 计算当前q值--评价当前无人机的动作
            critic_1_loss: torch.Tensor = F.mse_loss(current_q1, y) # 计算损失--当前q值和目标q值的均方误差
            # 反向传播更新
            self.critic_1_optimizers[agent_idx].zero_grad() # 清空之前的梯度
            critic_1_loss.backward() # 计算梯度
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.critics_1[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_1_optimizers[agent_idx].step() # 更新参数
            # 更新 critic 2
            current_q2: torch.Tensor = self.critics_2[agent_idx](obs_tensor, actions_tensor, agent_idx)
            critic_2_loss: torch.Tensor = F.mse_loss(current_q2, y)
            self.critic_2_optimizers[agent_idx].zero_grad()
            critic_2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics_2[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_2_optimizers[agent_idx].step() # 根据计算好的参数，更新模型的参数
            # 记录损失
            avg_critic_loss = (float(critic_1_loss.detach().item()) + float(critic_2_loss.detach().item())) / 2.0
            agent_critic_losses.append(avg_critic_loss)

        # Delayed Policy and Target Network Updates 延迟更新actor
        if self.update_counter % config.POLICY_UPDATE_FREQ == 0: # 每2步一次
            for agent_idx in range(self.num_agents):
                # Update Actor
                # 复制动作tensor
                pred_actions_tensor: torch.Tensor = actions_tensor.detach().clone()
                # 只替换当前agent的动作 （用当前actor重新计算）  self.actors[agent_idx] 是当前agent的Actor网络，obs_tensor[:, agent_idx, :] 是当前agent的观测
                pred_actions_tensor[:, agent_idx, :] = self.actors[agent_idx](obs_tensor[:, agent_idx, :])
                # 实际用的是 self.critics_1[agent_idx](...) 直接计算损失
                actor_loss: torch.Tensor = -self.critics_1[agent_idx](obs_tensor, pred_actions_tensor, agent_idx).mean()
                # 更新actor
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward() # 计算梯度
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM) # 梯度剪裁
                self.actor_optimizers[agent_idx].step() # 更新参数
                agent_losses.append(float(actor_loss.detach().item()))

                # Soft update all target networks 软更新目标网络
                # 软更新公式： target = τ × current + (1 - τ) × target  # τ很小，如0.01
                soft_update(self.target_actors[agent_idx], self.actors[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_1[agent_idx], self.critics_1[agent_idx], config.UPDATE_FACTOR)
                soft_update(self.target_critics_2[agent_idx], self.critics_2[agent_idx], config.UPDATE_FACTOR)
            # 衰减探索噪声
            for n in self.noise:
                n.decay()

        # Return averaged losses across all agents (same format as standard MATD3) 返回损失值
        return {
            "actor": float(np.mean(agent_losses)) if agent_losses else 0.0,
            "critic": float(np.mean(agent_critic_losses)),
        }
    # 把目标网络的参数初始化为和主网络一样
    def _init_target_networks(self) -> None:
        # 初始化 Actor的目标网络
        """
        self.actors = [actor0, actor1, actor2, actor3, actor4]
        self.target_actors = [target0, target1, target2, target3, target4]

        zip之后：
                第1次循环: actor=actor0, target_actor=target0
                第2次循环: actor=actor1, target_actor=target1
                第3次循环: actor=actor2, target_actor=target2
                第4次循环: actor=actor3, target_actor=target3
                第5次循环: actor=actor4, target_actor=target4
        """
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        # 初始化 critic 1 的目标网络
        for critic1, target_critic1 in zip(self.critics_1, self.target_critics_1):
            target_critic1.load_state_dict(critic1.state_dict())
        # 初始化 critic 2 的目标网络
        for critic2, target_critic2 in zip(self.critics_2, self.target_critics_2):
            target_critic2.load_state_dict(critic2.state_dict())
    # 重置状态
    def reset(self) -> None:
        for n in self.noise: # 重置噪声状态：每个新回合开始时，重置探索噪声的内部状态。
            n.reset()
    # 保存模型
    """
    saved_models/
├── agent_0.pth          # 无人机0的所有网络
├── agent_1.pth          # 无人机1的所有网络
├── agent_2.pth          # 无人机2的所有网络
├── agent_3.pth          # 无人机3的所有网络
├── agent_4.pth          # 无人机4的所有网络
└── update_counter.txt   # 当前更新次数
    """
    def save(self, directory: str) -> None:
        for i in range(self.num_agents): # 对每个无人机
            torch.save(
                {
                    # 网络参数
                    "actor": get_state_dict(self.actors[i]),
                    "critic_1": get_state_dict(self.critics_1[i]),
                    "critic_2": get_state_dict(self.critics_2[i]),
                    # 目标网络参数
                    "target_actor": self.target_actors[i].state_dict(),
                    "target_critic_1": self.target_critics_1[i].state_dict(),
                    "target_critic_2": self.target_critics_2[i].state_dict(),
                    # 优化器状态
                    "actor_optimizer": self.actor_optimizers[i].state_dict(),
                    "critic_1_optimizer": self.critic_1_optimizers[i].state_dict(),
                    "critic_2_optimizer": self.critic_2_optimizers[i].state_dict(),
                },
                os.path.join(directory, f"agent_{i}.pth"), # 每个agent单独文件
            )
            # 保存更新计数器（用于恢复训练时的步数）
        update_counter_path: str = os.path.join(directory, "update_counter.txt")
        with open(update_counter_path, "w") as f:
            f.write(str(self.update_counter))
    # 加载模型

    def load(self, directory: str) -> None:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"❌ Model directory not found: {directory}")

        for i in range(self.num_agents):
            agent_path: str = os.path.join(directory, f"agent_{i}.pth")
            if not os.path.exists(agent_path):
                raise FileNotFoundError(f"❌ Model file not found: {agent_path}")
            # 加载checkpoint
            checkpoint: dict = torch.load(agent_path, map_location=self.device, weights_only=True)
            # 加载网络参数
            load_safe(self.actors[i], checkpoint["actor"])
            load_safe(self.critics_1[i], checkpoint["critic_1"])
            load_safe(self.critics_2[i], checkpoint["critic_2"])
            # 加载目标网络
            self.target_actors[i].load_state_dict(checkpoint["target_actor"])
            self.target_critics_1[i].load_state_dict(checkpoint["target_critic_1"])
            self.target_critics_2[i].load_state_dict(checkpoint["target_critic_2"])
            # 加载优化器状态
            self.actor_optimizers[i].load_state_dict(checkpoint["actor_optimizer"])
            self.critic_1_optimizers[i].load_state_dict(checkpoint["critic_1_optimizer"])
            self.critic_2_optimizers[i].load_state_dict(checkpoint["critic_2_optimizer"])
        # 加载更新计数器
        update_counter_path: str = os.path.join(directory, "update_counter.txt")
        if os.path.exists(update_counter_path):
            with open(update_counter_path, "r") as f:
                self.update_counter = int(f.read())
        else:
            self.update_counter = 0
            print(f"⚠️ Update counter file not found: {update_counter_path}. Setting update_counter to 0.")
