from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Any, Dict, Optional

# 离线策略算法的经验格式（如：MADDPG、MATD3、DQN）
# 顺序固定：(obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch)
OffPolicyExperienceBatch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# 在线策略算法的经验格式（如：PPO、A2C）
# 字典形式，包含 obs, actions, log_probs, rewards, advantages 等
OnPolicyExperienceBatch = dict[str, torch.Tensor]
# 联合类型：update 方法可以接受任何一种
ExperienceBatch = OffPolicyExperienceBatch | OnPolicyExperienceBatch


class MARLModel(ABC):
    """
    Abstract Base Class for Multi-Agent Reinforcement Learning models.
    This class defines the essential methods that any MARL algorithm implementation
    must have to be compatible with the training framework.
    """

    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        self.model_name = model_name
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device   # CPU or CUDA

    # PPO专用
    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets actions, log probabilities, and state values.
        Essential for on-policy algorithms like PPO.
        获取动作、log概率和状态价值
    PPO等在线策略算法需要

    返回:
        actions: 选择的动作
        log_probs: 动作的对数概率
        values: 状态价值估计
        """
        raise NotImplementedError("This method is required for on-policy algorithms.")

    @abstractmethod
    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        """
        Selects actions for all agents based on their observations.
        所有智能体根据观察结果，为他们选择动作
        参数:
        observations: 每个无人机的观测列表，长度=5，每个元素形状(265,)
        exploration: 是否添加探索噪声（训练=True，测试=False）

        返回:
        actions: 形状(5,2)的数组，每个无人机[角度比例,距离比例]
        训练时：需要探索（加噪声）
        测试时：只用最优动作（不加噪声）
        输入：5个265维观测 → 输出：5个2维动作
        """
        pass

    @abstractmethod
    def update(self, batch: ExperienceBatch) -> Optional[Dict[str, Any]]:
        """

        这是学习发生的的地方！不同算法的更新逻辑完全不同：

        MADDPG：每个智能体有自己的Actor-Critic，用TD误差更新

        MATD3：两个Critic，取最小值，延迟Actor更新

        MAPPO：用GAE计算优势，裁剪目标函数

        Performs a learning update on the model's networks using a batch of experiences.
        利用一批训练数据对模型的网络结构进行学习更新。

        Args:
            batch (ExperienceBatch): A dictionary (for on-policy) or a tuple (for off-policy).
            批次数据（ExperienceBatch）：对于基于策略的学习方式，其为字典形式；而对于不基于策略的学习方式，则为元组形式。
        用一批经验更新网络参数

    参数:
        batch: 离线策略用元组，在线策略用字典

    返回:
        可选：损失值字典，用于日志记录

        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the model's internal state (if any) for a new episode.重置状态
        某些算法（如RNN-based）有内部状态，每个新回合需要重置。
        """
        pass

    @abstractmethod
    def save(self, directory: str) -> None:# 保存模型参数到指定目录
        pass

    @abstractmethod
    def load(self, directory: str) -> None:# 从指定目录加载模型参数
        pass
