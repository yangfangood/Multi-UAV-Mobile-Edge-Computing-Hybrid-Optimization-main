import numpy as np

# Training Parameters
MODEL: str = "matd3"  # options: 'maddpg', 'matd3', 'mappo', 'masac', 'attention_<model>', 'random'
SEED: int = 42  # random seed for reproducibility：随机种子，保证可复现
np.random.seed(SEED)  # set numpy random seed
STEPS_PER_EPISODE: int = 1000  # total T  每个回合1000步，每个步是一个时隙
LOG_FREQ: int = 10  # episodes 每个10和回合记录一次日志
IMG_FREQ: int = 1000  # steps
TEST_LOG_FREQ: int = 1  # episodes (for testing)
TEST_IMG_FREQ: int = 100  # steps (for testing)

# Simulation Parameters 理解问题有多大
MBS_POS: np.ndarray = np.array([350.0, 350.0, 30.0])  # (X_mbs, Y_mbs, Z_mbs) in meters 基站位置
NUM_UAVS: int = 5  # U
MAX_LOAD = 30   # 每个无人机最多同时服务30个任务（不超过 MAX_ASSOCIATED_UES）
NUM_UES: int = 100  # M  用户数量100个
AREA_WIDTH: int = 700  # X_max in meters
AREA_HEIGHT: int = 700  # Y_max in meters
TIME_SLOT_DURATION: float = 1.0  # tau in seconds 每个时隙是1s
UE_MAX_DIST: float = 15.0  # d_max^UE in meters
UE_MAX_WAIT_TIME: int = 10  # in time slots

# UAV Parameters  每个无人机能在100米半径内服务用户，存储40-80MB数据
UAV_ALTITUDE: int = 100  # H in meters
UAV_SPEED: float = 10.0  # v^UAV in m/s
UAV_STORAGE_CAPACITY: np.ndarray = np.random.choice(np.arange(40 * 10**6, 80 * 10**6, 10**6), size=NUM_UAVS).astype(np.int64)  # S_u in bytes 存储容量
UAV_COMPUTING_CAPACITY: np.ndarray = np.random.choice(np.arange(5 * 10**9, 20 * 10**9, 10**9), size=NUM_UAVS).astype(np.int64)  # F_u in cycles/sec
UAV_SENSING_RANGE: float = 300.0  # R^sense in meters
UAV_COVERAGE_RADIUS: float = 100.0  # R in meters 无人机覆盖半径
MIN_UAV_SEPARATION: float = 200.0  # d_min in meters 无人机之间的最小安全距离
assert np.all(UAV_STORAGE_CAPACITY > 0)
assert np.all(UAV_COMPUTING_CAPACITY > 0)
assert UAV_COVERAGE_RADIUS * 2 <= MIN_UAV_SEPARATION
assert UAV_SENSING_RANGE >= MIN_UAV_SEPARATION

# Collisions and Penalties
COLLISION_AVOIDANCE_ITERATIONS: int = 20  # number of iterations to resolve collisions
COLLISION_PENALTY: float = 1.0  # penalty per collision
BOUNDARY_PENALTY: float = 1.0  # penalty for going out of bounds
NON_SERVED_LATENCY_PENALTY: float = 20.0  # penalty in latency for non-served requests
# IMPORTANT : Reconfigurable, should try for various values including : NUM_UAVS - 1 and NUM_UES
MAX_UAV_NEIGHBORS: int = NUM_UAVS - 1
MAX_ASSOCIATED_UES: int = min(30, NUM_UES // NUM_UAVS + 10)
assert MAX_UAV_NEIGHBORS >= 1 and MAX_UAV_NEIGHBORS <= NUM_UAVS - 1
assert MAX_ASSOCIATED_UES >= 1 and MAX_ASSOCIATED_UES <= NUM_UES

POWER_MOVE: float = 100.0  # P_move in Watts
POWER_HOVER: float = 80.0  # P_hover in Watts

# Request Parameters
NUM_SERVICES: int = 25  # S 服务类文件数量
NUM_CONTENTS: int = 50  # K  下载类文件数量
NUM_FILES: int = NUM_SERVICES + NUM_CONTENTS  # S + K 总文件数量
CPU_CYCLES_PER_BYTE: np.ndarray = np.random.randint(2000, 4000, size=NUM_SERVICES)  # omega_s_m
FILE_SIZES: np.ndarray = np.random.randint(10**6, 5 * 10**6, size=NUM_FILES).astype(np.int64)  # in bytes
MIN_INPUT_SIZE: int = 1 * 10**6  # in bytes
MAX_INPUT_SIZE: int = 5 * 10**6  # in bytes
ZIPF_BETA: float = 0.8  # beta^Zipf
K_CPU: float = 1e-27  # CPU capacitance coefficient

# Caching Parameters
T_CACHE_UPDATE_INTERVAL: int = 50  # T_cache 缓存更新频率
GDSF_SMOOTHING_FACTOR: float = 0.75  # beta^gdsf

# Probabilistic Caching Parameters
AVG_FILE_SIZE: float = float(np.mean(FILE_SIZES))
PROB_GAMMA: float = 0.5  # gamma

# Communication Parameters 通信参数
G_CONSTS_PRODUCT: float = 2.2846 * 1.42 * 1e-4  # G_0 * g_0
TRANSMIT_POWER: float = 0.5  # P^comm in Watts 电池的传输能耗
AWGN: float = 1e-13  # sigma^2
BANDWIDTH_INTER: int = 20 * 10**6  # B^inter in Hz 无人机间通信带宽
BANDWIDTH_EDGE: int = 40 * 10**6  # B^edge in Hz 无人机-用户通信带宽
BANDWIDTH_BACKHAUL: int = 10 * 10**6  # B^backhaul in Hz

# WPT Parameters 无线能量传输
UE_BATTERY_CAPACITY: float = 100.0  # B_max in Joules
UE_CRITICAL_THRESHOLD: float = 0.3 * UE_BATTERY_CAPACITY  # B_low in Joules
WPT_TRANSMIT_POWER: float = 5000.0  # P^WPT in Watts (UAV WPT transmit power for energy harvesting) 无人机发射功率
WPT_EFFICIENCY: float = 0.6  # eta (energy harvesting efficiency, 60%) 能量转换效率
UE_STATIC_POWER: float = 0.05  # Idle power consumption in Watts  用户设备静态能耗

# Model Parameters 模型参数----理解优化目标
# Reward formula: reward = ALPHA_3*log(fairness) - ALPHA_1*log(latency) - ALPHA_2*log(energy) - ALPHA_4*log(1+offline_rate)
# 奖励函数：reward = α₃*log(公平性) - α₁*log(延迟) - α₂*log(能耗) - α₄*log(1+离线率)
# Then scaled by REWARD_SCALING_FACTOR. All log terms ∈ [0, log(max_value)] to keep rewards bounded.
# 然后乘以奖励缩放因子REWARD_SCALING_FACTOR。所有对数项的值都保持在[0, log(max_value)]的范围内，从而确保奖励值处于可控范围内。
ALPHA_1 = 1.0  # weightage for latency (negative term, higher = stronger penalty for latency) 延迟权重（越大越惩罚延迟）
ALPHA_2 = 0.6  # weightage for energy (negative term, lower priority than latency)  能耗权重（比延迟优先级低）
ALPHA_3 = 2.0  # weightage for fairness (positive term, encourage equal service)   公平性权重（鼓励公平服务）
ALPHA_4 = 8.0 # weightage for offline rate (negative term, penalizes UEs running out of battery) 离线率权重（强烈惩罚设备掉电）最怕设备没电掉线
#REWARD_SCALING_FACTOR: float = 0.01  # scaling factor for rewards (prevents exploding values)奖励的缩放因子（用于防止数值过高而导致的异常情况）
REWARD_SCALING_FACTOR: float = 0.01

# 根据训练前期经验设定合理上限（可调） 归一化的基准值
#MAX_LATENCY = 3.5e9  # 最大总延迟（秒）
#MAX_ENERGY = 1.2e7  # 最大总能耗（焦耳）
MAX_LATENCY = 4e9   # 不要低于4e9
MAX_ENERGY  = 1.5e7   # 不要低于1.5e7
#观测空间（理解智能体能看到什么）
UAV_TYPE_OBS_DIM = 3  # 新增，放在 SELF_OBS_DIM 之前
# 新增两个能力数值的维度
UAV_CAPABILITY_OBS_DIM = 2   # 计算能力 + 存储容量 新增这两个维度
UAV_LOAD_OBS_DIM = 1
SELF_OBS_DIM: int = 2 + NUM_FILES + UAV_TYPE_OBS_DIM + UAV_CAPABILITY_OBS_DIM + UAV_LOAD_OBS_DIM# pos (2) + cache (NUM_FILES)  自身状态：位置(2) + 缓存状态(75维)
UE_OBS_DIM: int = 2 + 3 + 1  # pos (2) + request_tuple (3) + battery level (1) 用户状态：位置(2) + 请求(3) + 电量(1)
NEIGHBOR_OBS_DIM: int = 2  # pos (2) 邻居位置(2)
#每个无人机总的观测维度
OBS_DIM_SINGLE: int = SELF_OBS_DIM + (MAX_UAV_NEIGHBORS * NEIGHBOR_OBS_DIM) + (MAX_ASSOCIATED_UES * UE_OBS_DIM)
ACTION_DIM: int = 2  # angle, distance from [-1, 1] # 角度, 距离，范围都在[-1, 1]  第1维：飞行方向（-1到1映射到0°到360°），飞行距离比例（-1到1映射到0到最大速度）
MLP_HIDDEN_DIM: int = 128
#计算一下：
# MAX_UAV_NEIGHBORS = 4（最多看4个邻居）
# MAX_ASSOCIATED_UES = 30（最多看30个用户）
#所以每个无人机的观测向量维度 =
#• 自身：2 + 75 = 77
#• 邻居：4 × 2 = 8
#• 用户：30 × (2+3+1=6) = 180
#• 总计：77 + 8 + 180 = 265维


ACTOR_LR: float = 3e-4
CRITIC_LR: float = 3e-4
DISCOUNT_FACTOR: float = 0.99  # gamma
UPDATE_FACTOR: float = 0.01  # tau
MAX_GRAD_NORM: float = 0.5  # maximum norm for gradient clipping to prevent exploding gradients
LOG_STD_MAX: float = 2  # maximum log standard deviation for stochastic policies
LOG_STD_MIN: float = -20  # minimum log standard deviation for stochastic policies
EPSILON: float = 1e-9  # small value to prevent division by zero 一个超级小的量，避免除0

# Off-policy algorithm hyperparameters
REPLAY_BUFFER_SIZE: int = 10**6  # B
REPLAY_BATCH_SIZE: int = 256  # minibatch size
INITIAL_RANDOM_STEPS: int = 5000  # steps of random actions for exploration
LEARN_FREQ: int = 10  # steps to learn after

# Gaussian Noise Parameters (for MADDPG and MATD3)
INITIAL_NOISE_SCALE: float = 0.15 #初始探索噪声
MIN_NOISE_SCALE: float = 0.01  #噪声衰减（越训练越少探索）
NOISE_DECAY_RATE: float = 0.995

# MATD3 Specific Hyperparameters
POLICY_UPDATE_FREQ: int = 2  # delayed policy update frequency   MATD3特有：每2次Critic更新才更新1次Actor
TARGET_POLICY_NOISE: float = 0.2  # standard deviation of target policy smoothing noise.
NOISE_CLIP: float = 0.5  # range to clip target policy smoothing noise

# MAPPO Specific Hyperparameters
PPO_ROLLOUT_LENGTH: int = STEPS_PER_EPISODE  # number of steps to collect per rollout before updating
PPO_GAE_LAMBDA: float = 0.95  # lambda parameter for GAE
PPO_EPOCHS: int = 10  # number of epochs to run on the collected rollout data 每个batch训练10轮
PPO_BATCH_SIZE: int = 200  # size of mini-batches to use during the update step
PPO_CLIP_EPS: float = 0.2  # clipping parameter (epsilon) for the PPO surrogate objective 裁剪范围
PPO_ENTROPY_COEF: float = 0.01  # coefficient for the entropy bonus to encourage exploration  熵奖励（鼓励探索）

# MASAC Specific Hyperparameters
ALPHA_LR: float = 3e-4  # learning rate for the entropy temperature alpha

# Attention Hyperparameters
ATTN_HIDDEN_DIM: int = 64  # Embedding size for internal attention representations 注意力机制的隐层维度
ATTN_NUM_HEADS: int = 4  # Number of attention heads 4个注意力头

assert ATTN_HIDDEN_DIM % ATTN_NUM_HEADS == 0, f"ATTN_HIDDEN_DIM ({ATTN_HIDDEN_DIM}) must be divisible by ATTN_NUM_HEADS ({ATTN_NUM_HEADS})"


# ========== 障碍物参数（版本1-最小改动）==========

# 实验开关：0=无障碍物，>0=有障碍物
NUM_STATIC_OBSTACLES: int = 5    # 静态障碍物数量（建筑物）
NUM_DYNAMIC_OBSTACLES: int = 0   # 动态障碍物数量（移动物体）

# 障碍物尺寸
STATIC_OBSTACLE_RADIUS: float = 30.0   # 静态障碍物半径（米）
DYNAMIC_OBSTACLE_RADIUS: float = 20.0  # 动态障碍物半径（米）

# 避障参数
OBSTACLE_COLLISION_PENALTY: float = 20.0  # 碰撞障碍物惩罚

# 动态障碍物速度范围
DYNAMIC_OBSTACLE_SPEED_RANGE: tuple = (-3.0, 3.0)  # 速度范围（米/秒）

#=======================异构无人机配置=======================
# 无人机类型定义
UAV_TYPE_COMPUTE = 0    # 计算型
UAV_TYPE_STORAGE = 1    # 存储型
UAV_TYPE_BALANCED = 2   # 均衡型

# 5个无人机的类型分配（可根据需要修改）
UAV_TYPE_ASSIGN = [
    UAV_TYPE_COMPUTE,   # UAV0
    UAV_TYPE_STORAGE,   # UAV1
    UAV_TYPE_BALANCED,  # UAV2
    UAV_TYPE_COMPUTE,   # UAV3
    UAV_TYPE_STORAGE    # UAV4
]
# UAV 能力归一化常数（必须在 UAV_COMPUTING_CAPACITY 和 UAV_STORAGE_CAPACITY 定义之后）
# 添加最大值常数，供env.py中归一化使用
MAX_UAV_COMPUTING = np.max(UAV_COMPUTING_CAPACITY)
MAX_UAV_STORAGE = np.max(UAV_STORAGE_CAPACITY)
# 增加负载归一化参数  也可以根据实际最大值动态计算MAX_UAV_COMPUTING
MAX_LOAD = MAX_ASSOCIATED_UES   # 假设最大服务请求数，可根据场景设定
MAX_QUEUE = 20 # 最大队列长度

# 硬件参数
# 计算型
COMPUTE_UAV_COMPUTING = 20 * 10**9   # 20 GHz
COMPUTE_UAV_STORAGE   = 20 * 10**6   # 20 MB

# 存储型
STORAGE_UAV_COMPUTING = 5 * 10**9    # 5 GHz
STORAGE_UAV_STORAGE   = 80 * 10**6   # 80 MB

# 均衡型
BALANCED_UAV_COMPUTING = 12 * 10**9
BALANCED_UAV_STORAGE   = 50 * 10**6

# 能力匹配因子（用于延迟修正）
# 服务任务对计算型的加速因子
SERVICE_COMPUTE_FACTOR = 1.2   # 延迟缩短20%
SERVICE_STORAGE_FACTOR = 0.8   # 延迟增加20%
SERVICE_BALANCED_FACTOR = 1.0

# 内容任务对存储型的加速因子
CONTENT_STORAGE_FACTOR = 1.2
CONTENT_COMPUTE_FACTOR = 0.8
CONTENT_BALANCED_FACTOR = 1.0

# 匹配奖励（可选）
MATCH_REWARD_SERVICE_TO_COMPUTE = 0.5
MATCH_REWARD_CONTENT_TO_STORAGE = 0.5
MATCH_REWARD_BALANCED_BONUS = 0.2

# === 新增 G2A 信道模型参数 ===
# 场景设置
ENVIRONMENT = "urban"  # "urban", "suburban", "rural"
CARRIER_FREQUENCY = 2.4e9  # Hz

# LoS 概率模型参数 (以城市环境为例)
LOS_A = 9.61
LOS_B = 0.16

# 路径损耗指数
PATH_LOSS_EXP_LOS = 2.0
PATH_LOSS_EXP_NLOS = 3.5

# 额外衰减因子 (dB)
ETA_LOS = 0.0  # dB
ETA_NLOS = 20.0  # dB

# ==================HGAT 集成方案 ===================

# 节点类型名称
NODE_TYPE_COMPUTE = 'compute'
NODE_TYPE_STORAGE = 'storage'
NODE_TYPE_BALANCED = 'balanced'

# 每种节点类型的特征维度（可能不同，这里假设相同）
NODE_FEATURE_DIM = 128   # 与 hidden_dim 一致