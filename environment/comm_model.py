import config
import numpy as np

# 计算信道增益 信号在空间中传播时会衰减。距离越远，信号越弱。
def calculate_channel_gain(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculates channel gain based on the free-space path loss model.根据自由空间路径损耗模型来计算信道增益。
    距离增加一倍，增益降到1/4。这就是为什么无人机要尽量靠近用户！"""
    distance_sq: float = np.sum((pos1 - pos2) ** 2) # 计算距离平方
    return (config.G_CONSTS_PRODUCT) / (distance_sq + config.EPSILON)

# 计算信噪比  信号功率 vs 噪声功率。信噪比越高，通信质量越好。
# 信号很强 + 噪声很弱 → SNR高 → 通信质量好
# 信号很弱 + 噪声很强 → SNR低 → 可能传错数据
def calculate_ue_uav_rate(channel_gain: float, num_associated_ues: int) -> float:
    """Calculates data rate between a UE and a UAV."""
    assert num_associated_ues != 0
    bandwidth_per_ue: float = config.BANDWIDTH_EDGE / num_associated_ues # 带宽共享！ 无人机覆盖的用户越多，每个用户分到的带宽越少。
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return bandwidth_per_ue * np.log2(1 + snr)

# 香农公式 - 传输速率 独占带宽——每个无人机独立连接基站，不共享。 基站是“内容仓库”，无人机需要时从这里获取文件，但回程带宽是瓶颈（只有10MHz）。
# 给定信噪比和带宽，理论上能达到的最大传输速率。
def calculate_uav_mbs_rate(channel_gain: float) -> float:
    """Calculates data rate between a UAV and the MBS."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_BACKHAUL * np.log2(1 + snr)

# 三种链路详细分析
# 用于无人机协作，带宽独立。
def calculate_uav_uav_rate(channel_gain: float) -> float:
    """Calculates data rate between two UAVs."""
    snr: float = (config.TRANSMIT_POWER * channel_gain) / config.AWGN
    return config.BANDWIDTH_INTER * np.log2(1 + snr)
