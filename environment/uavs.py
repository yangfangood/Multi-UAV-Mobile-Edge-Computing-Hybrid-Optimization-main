from __future__ import annotations
from dataclasses import dataclass
from environment.user_equipments import UE
from environment import comm_model as comms
import config
import numpy as np


@dataclass(frozen=True)
class RequestPlan:
    source_uav: "UAV"
    ue: UE
    request_type: int          # 0=service, 1=content, 2=energy
    ue_uav_rate: float | None = None
    target_idx: int | None = None  # 0=local, 1=neighbor, 2=MBS, None=energy
    target_uav: "UAV | None" = None
    estimate: "RouteEstimate | None" = None  # 能量请求为 None
    order_index: int = 0


@dataclass(frozen=True)
class RouteEstimate:
    """一次卸载路径的完整时延和能耗估计（纯计算，无副作用）"""

    ue_upload_time: float       # UE→UAV 上传（服务 >0，内容 =0）
    ue_download_time: float     # UAV→UE 下发（内容 >0，服务 =0）
    uav_uav_time: float         # UAV↔UAV 传输
    backhaul_time: float        # UAV↔MBS 回传
    compute_time: float         # 计算时长

    ue_tx_energy: float
    source_tx_energy: float
    source_rx_energy: float
    target_tx_energy: float
    target_rx_energy: float
    compute_energy: float

    compute_node_load: int
    # target_idx ∈ (0,1): execution_load
    # target_idx == 2:   mbs_execution_load

    @property
    def total_latency(self) -> float:
        return (self.ue_upload_time + self.ue_download_time
                + self.uav_uav_time + self.backhaul_time
                + self.compute_time)

    @property
    def controllable_energy(self) -> float:
        return (self.ue_tx_energy + self.source_tx_energy
                + self.source_rx_energy + self.target_tx_energy
                + self.target_rx_energy + self.compute_energy)


# 用逻辑回归 估计邻居缓存某个文件的概率，用于卸载决策
def _get_belief_probability(file_id: int, neighbor_id: int) -> float:
    """Returns the estimated probability P_{v,i} that a neighbor has file_i."""
    rank = UE.id_to_rank_map[file_id]  # 文件流行度排名
    c_hat_v: float = config.UAV_STORAGE_CAPACITY[neighbor_id] / config.AVG_FILE_SIZE  # 邻居能存多少文件
    exponent: float = config.PROB_GAMMA * (rank - c_hat_v)  # γ=0.5
    probability: float = 1.0 / (1.0 + np.exp(exponent))  # Sigmoid函数
    return probability


# 计算无人机 处理任务的计算时延和能耗
def _get_computing_latency_and_energy(uav: UAV, cpu_cycles: float) -> tuple[float, float]:
    """使用执行负载计算计算时延和能耗"""
    load = uav.execution_service_load
    if load <= 0:
        raise RuntimeError(
            f"UAV{uav.id} executes a service but execution load is {load}."
        )
    frequency = uav.computing_capacity / load
    latency = cpu_cycles / frequency
    energy = config.K_CPU * cpu_cycles * (frequency ** 2)
    return latency, energy


def _try_add_file_to_cache(uav: UAV, file_id: int) -> None:
    """Try to add a file to UAV cache if there's enough space."""
    if uav._working_cache[file_id]:
        return  # Already in cache
    used_space: int = np.sum(uav._working_cache * config.FILE_SIZES)  # 已用空间
    if used_space + config.FILE_SIZES[file_id] <= uav.storage_capacity:
        uav._working_cache[file_id] = True  # 空间够就加入


# ============================================================
# estimate_route() 纯函数系（v1.5 联合卸载优化）
# ============================================================

def _capability_factor(uav_type: int, req_type: int) -> float:
    """只作用于 compute_time。content 不使用（与当前执行一致）。"""
    if req_type == 0:  # service
        if uav_type == config.UAV_TYPE_COMPUTE:
            return config.SERVICE_COMPUTE_FACTOR    # 1.2
        elif uav_type == config.UAV_TYPE_STORAGE:
            return config.SERVICE_STORAGE_FACTOR    # 0.8
        else:
            return config.SERVICE_BALANCED_FACTOR   # 1.0
    return 1.0


def _estimate_service_route(
    source_uav: UAV,
    ue: UE,
    target_idx: int,
    target_uav: UAV | None,
    execution_load: int,
    ue_uav_rate: float,
    source_mbs_rate: float,
    target_mbs_rate: float | None,
    mbs_execution_load: int,
    req_id: int,
    req_size: int,
    cpu_cycles: float,
    file_size: int,
) -> RouteEstimate:
    """服务请求的完整时延能耗估计（纯计算，无副作用）。"""
    # === 防御性校验 ===
    if req_id >= config.NUM_SERVICES:
        raise ValueError(f"Service req_id {req_id} exceeds NUM_SERVICES.")
    if not np.isfinite(ue_uav_rate) or ue_uav_rate <= 0.0:
        raise ValueError(f"Invalid ue_uav_rate: {ue_uav_rate}")
    if not np.isfinite(source_mbs_rate) or source_mbs_rate <= 0.0:
        raise ValueError(f"Invalid source_mbs_rate: {source_mbs_rate}")

    # === 按目标分支检查负载 ===
    if target_idx in (0, 1):
        if execution_load <= 0:
            raise ValueError(f"UAV execution_load must be positive, got {execution_load}.")
    elif target_idx == 2:
        if mbs_execution_load <= 0:
            raise ValueError(f"MBS execution_load must be positive, got {mbs_execution_load}.")
    else:
        raise ValueError(f"Invalid service target: {target_idx}")

    # === 公共项 ===
    ue_upload_time = req_size / ue_uav_rate
    ue_tx_energy = config.UE_TX_POWER * ue_upload_time
    source_rx_energy = config.UAV_RX_POWER * ue_upload_time

    uav_uav_time = 0.0
    backhaul_time = 0.0
    compute_time = 0.0
    source_tx_energy = 0.0
    target_tx_energy = 0.0
    target_rx_energy = 0.0
    compute_energy = 0.0
    effective_load = execution_load

    if target_idx == 0:
        # 本地
        fetch_time = 0.0 if source_uav.cache[req_id] else file_size / source_mbs_rate
        backhaul_time = fetch_time
        source_rx_energy += config.UAV_RX_POWER * fetch_time

        frequency = source_uav.computing_capacity / execution_load
        raw_compute = cpu_cycles / frequency
        compute_time = raw_compute / _capability_factor(source_uav.type, req_type=0)
        compute_energy = config.K_CPU * cpu_cycles * frequency ** 2

    elif target_idx == 1:
        # 邻居
        if target_uav is None:
            raise ValueError("Neighbor target requires target_uav.")
        if target_mbs_rate is None:
            raise ValueError("Neighbor path requires target_mbs_rate.")
        if not np.isfinite(target_mbs_rate) or target_mbs_rate <= 0.0:
            raise ValueError(f"Invalid target_mbs_rate: {target_mbs_rate}")

        uav_uav_rate = comms.calculate_uav_uav_rate(
            comms.calculate_channel_gain(source_uav.pos, target_uav.pos))
        if not np.isfinite(uav_uav_rate) or uav_uav_rate <= 0.0:
            raise RuntimeError(
                f"Invalid UAV-UAV rate: source={source_uav.id}, target={target_uav.id}")

        uav_uav_time = req_size / uav_uav_rate
        source_tx_energy = config.UAV_TX_POWER * uav_uav_time
        target_rx_energy = config.UAV_RX_POWER * uav_uav_time

        fetch_time = 0.0 if target_uav.cache[req_id] else file_size / target_mbs_rate
        backhaul_time = fetch_time
        target_rx_energy += config.UAV_RX_POWER * fetch_time

        frequency = target_uav.computing_capacity / execution_load
        raw_compute = cpu_cycles / frequency
        compute_time = raw_compute / _capability_factor(target_uav.type, req_type=0)
        compute_energy = config.K_CPU * cpu_cycles * frequency ** 2

    elif target_idx == 2:
        # MBS
        mbs_upload_time = req_size / source_mbs_rate
        backhaul_time = mbs_upload_time
        source_tx_energy = config.UAV_TX_POWER * mbs_upload_time

        mbs_frequency = config.MBS_COMPUTING_CAPACITY / mbs_execution_load
        compute_time = cpu_cycles / mbs_frequency
        effective_load = mbs_execution_load

    return RouteEstimate(
        ue_upload_time=ue_upload_time, ue_download_time=0.0,
        uav_uav_time=uav_uav_time, backhaul_time=backhaul_time,
        compute_time=compute_time,
        ue_tx_energy=ue_tx_energy, source_tx_energy=source_tx_energy,
        source_rx_energy=source_rx_energy, target_tx_energy=target_tx_energy,
        target_rx_energy=target_rx_energy, compute_energy=compute_energy,
        compute_node_load=effective_load,
    )


def _estimate_content_route(
    source_uav: UAV,
    ue: UE,
    target_idx: int,
    target_uav: UAV | None,
    ue_uav_rate: float,
    source_mbs_rate: float,
    target_mbs_rate: float | None,
    req_id: int,
    file_size: int,
) -> RouteEstimate:
    """内容请求的完整时延能耗估计（纯计算，无副作用）。"""
    # === 防御性校验 ===
    if req_id < config.NUM_SERVICES:
        raise ValueError(f"Content req_id {req_id} is in service range.")
    if not np.isfinite(ue_uav_rate) or ue_uav_rate <= 0.0:
        raise ValueError(f"Invalid ue_uav_rate: {ue_uav_rate}")
    if not np.isfinite(source_mbs_rate) or source_mbs_rate <= 0.0:
        raise ValueError(f"Invalid source_mbs_rate: {source_mbs_rate}")

    ue_download_time = file_size / ue_uav_rate
    source_tx_energy = config.UAV_TX_POWER * ue_download_time

    uav_uav_time = 0.0
    backhaul_time = 0.0
    source_rx_energy = 0.0
    target_tx_energy = 0.0
    target_rx_energy = 0.0

    if target_idx == 0:
        # 本地
        if not source_uav.cache[req_id]:
            fetch_time = file_size / source_mbs_rate
            backhaul_time = fetch_time
            source_rx_energy = config.UAV_RX_POWER * fetch_time

    elif target_idx == 1:
        # 邻居
        if target_uav is None:
            raise ValueError("Neighbor target requires target_uav.")
        if target_mbs_rate is None:
            raise ValueError("Neighbor path requires target_mbs_rate.")
        if not np.isfinite(target_mbs_rate) or target_mbs_rate <= 0.0:
            raise ValueError(f"Invalid target_mbs_rate: {target_mbs_rate}")

        uav_uav_rate = comms.calculate_uav_uav_rate(
            comms.calculate_channel_gain(source_uav.pos, target_uav.pos))
        if not np.isfinite(uav_uav_rate) or uav_uav_rate <= 0.0:
            raise RuntimeError(
                f"Invalid UAV-UAV rate: source={source_uav.id}, target={target_uav.id}")

        uav_uav_time = file_size / uav_uav_rate
        target_tx_energy = config.UAV_TX_POWER * uav_uav_time
        source_rx_energy = config.UAV_RX_POWER * uav_uav_time

        if not target_uav.cache[req_id]:
            fetch_time = file_size / target_mbs_rate
            backhaul_time = fetch_time
            target_rx_energy = config.UAV_RX_POWER * fetch_time

    elif target_idx == 2:
        # MBS
        mbs_download_time = file_size / source_mbs_rate
        backhaul_time = mbs_download_time
        source_rx_energy = config.UAV_RX_POWER * mbs_download_time

    else:
        raise ValueError(f"Invalid content target: {target_idx}")

    return RouteEstimate(
        ue_upload_time=0.0, ue_download_time=ue_download_time,
        uav_uav_time=uav_uav_time, backhaul_time=backhaul_time,
        compute_time=0.0,
        ue_tx_energy=0.0, source_tx_energy=source_tx_energy,
        source_rx_energy=source_rx_energy, target_tx_energy=target_tx_energy,
        target_rx_energy=target_rx_energy, compute_energy=0.0,
        compute_node_load=0,
    )


def estimate_route(
    source_uav: UAV,
    ue: UE,
    target_idx: int,
    target_uav: UAV | None,
    execution_load: int,
    ue_uav_rate: float,
    source_mbs_rate: float,
    target_mbs_rate: float | None = None,
    mbs_execution_load: int = 0,
) -> RouteEstimate:
    """统一入口：为一条卸载路径生成 RouteEstimate。"""
    req_type, req_size, req_id = ue.current_request

    # 统一 req_id 范围检查
    if not (0 <= req_id < config.NUM_FILES):
        raise ValueError(f"Invalid req_id: {req_id} (must be 0..{config.NUM_FILES - 1})")

    if req_type == 0:
        if req_id >= config.NUM_SERVICES:
            raise ValueError(f"Service req_id {req_id} exceeds NUM_SERVICES.")
        cpu_cycles = float(config.CPU_CYCLES_PER_BYTE[req_id]) * float(req_size)
        file_size = config.FILE_SIZES[req_id]
        return _estimate_service_route(
            source_uav, ue, target_idx, target_uav,
            execution_load, ue_uav_rate, source_mbs_rate,
            target_mbs_rate, mbs_execution_load,
            req_id, req_size, cpu_cycles, file_size,
        )
    elif req_type == 1:
        if req_id < config.NUM_SERVICES:
            raise ValueError(f"Content req_id {req_id} is in service range.")
        file_size = config.FILE_SIZES[req_id]
        return _estimate_content_route(
            source_uav, ue, target_idx, target_uav,
            ue_uav_rate, source_mbs_rate, target_mbs_rate,
            req_id, file_size,
        )
    else:
        raise ValueError(f"estimate_route does not support request type {req_type}")


class UAV:
    # 初始化无人机属性
    def __init__(self, uav_id: int) -> None:
        self.id: int = uav_id  # 唯一标识
        # 获取类型
        self.type = config.UAV_TYPE_ASSIGN[uav_id]
        # 根据类型设置计算和存储能力
        if self.type == config.UAV_TYPE_COMPUTE:
            self.computing_capacity = config.COMPUTE_UAV_COMPUTING
            self.storage_capacity = config.COMPUTE_UAV_STORAGE
        elif self.type == config.UAV_TYPE_STORAGE:
            self.computing_capacity = config.STORAGE_UAV_COMPUTING
            self.storage_capacity = config.STORAGE_UAV_STORAGE
        else:  # BALANCED
            self.computing_capacity = config.BALANCED_UAV_COMPUTING
            self.storage_capacity = config.BALANCED_UAV_STORAGE
        # 三维位置
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH),  # x坐标 0-700米随机
                                         np.random.uniform(0, config.AREA_HEIGHT),  # y坐标 0-700米随机
                                         config.UAV_ALTITUDE])  # 固定高度100米

        self._dist_moved: float = 0.0  # Distance moved in the current time slot 当前时隙移动的距离
        self._current_covered_ues: list[UE] = []  # 当前无人机对应的ue队列
        self._neighbors: list[UAV] = []  # 相邻无人机
        self._associated_service_load: int = 0  # 关联服务请求数（观测用）
        self._execution_service_load: int = 0  # 实际执行服务数（计算用）
        self.collision_violation: bool = False  # Track if UAV has violated minimum separation 跟踪无人机是否违反了最小碰撞距离
        self.boundary_violation: bool = False  # Track if UAV has gone out of bounds 跟踪无人机是否越界

        # Cache and request tracking 无人机物理存储了哪些文件，可以直接服务用户
        self.cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)  # 物理缓存 哪些文件真的存了
        self._working_cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)  # 临时缓存，处理请求时用
        self._freq_counts: np.ndarray = np.zeros(config.NUM_FILES)  # 当前时间步各文件被请求次数
        self._ema_scores: np.ndarray = np.zeros(config.NUM_FILES)  # 指数移动平均，平滑后的文件热度

        self._uav_mbs_rate: float = 0.0  # 无人机到基站的传输速率

        self.total_fly_energy = 0.0  # 累计飞行+悬停能耗
        self.total_wpt_tx_energy = 0.0  # 累计无线充电发射能耗

        self.total_compute_energy = 0.0

        # === 时隙能耗分项（6 个） ===
        self._fly_energy_slot: float = 0.0
        self._compute_energy_slot: float = 0.0
        self._comm_tx_energy_slot: float = 0.0
        self._comm_rx_energy_slot: float = 0.0
        self._wpt_energy_slot: float = 0.0
        self._harvested_energy_slot: float = 0.0

        # === 累计字段（新增 2 个） ===
        self.total_comm_tx_energy: float = 0.0
        self.total_comm_rx_energy: float = 0.0

        # ✅ 新增：记录本 UAV 服务的延迟列表（用于 Credit Assignment 验证）
        self._served_latencies: list[float] = []

    @property
    def communication_energy(self) -> float:
        return self._comm_tx_energy_slot + self._comm_rx_energy_slot

    @property
    def energy(self) -> float:
        return (self._fly_energy_slot + self._compute_energy_slot +
                self.communication_energy + self._wpt_energy_slot)

    @property
    def current_covered_ues(self) -> list[UE]:
        return self._current_covered_ues

    @property
    def neighbors(self) -> list[UAV]:
        return self._neighbors

    @property
    def associated_service_load(self) -> int:
        return self._associated_service_load

    @property
    def execution_service_load(self) -> int:
        return self._execution_service_load

    def set_execution_service_load(self, load: int) -> None:
        if load < 0:
            raise ValueError("Execution load cannot be negative.")
        self._execution_service_load = load

    # ✅ 新增：记录一次服务的延迟
    def add_served_latency(self, latency: float) -> None:
        """记录一次服务的延迟（用于 Credit Assignment 验证）"""
        self._served_latencies.append(latency)

    # ✅ 新增：获取本回合服务的平均延迟
    def get_avg_served_latency(self) -> float:
        """获取本回合服务的平均延迟"""
        if not self._served_latencies:
            return 0.0
        return sum(self._served_latencies) / len(self._served_latencies)

    def add_tx_energy(self, duration: float) -> None:
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError(f"Invalid TX duration: {duration}")
        energy = config.UAV_TX_POWER * duration
        self._comm_tx_energy_slot += energy
        self.total_comm_tx_energy += energy

    def add_rx_energy(self, duration: float) -> None:
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError(f"Invalid RX duration: {duration}")
        energy = config.UAV_RX_POWER * duration
        self._comm_rx_energy_slot += energy
        self.total_comm_rx_energy += energy

    def add_tx_energy_value(self, energy: float) -> None:
        """按能量值直接记账（用于新执行器从 RouteEstimate 取值）。"""
        if not np.isfinite(energy) or energy < 0.0:
            raise ValueError(f"Invalid TX energy: {energy}")
        self._comm_tx_energy_slot += energy
        self.total_comm_tx_energy += energy

    def add_rx_energy_value(self, energy: float) -> None:
        """按能量值直接记账（用于新执行器从 RouteEstimate 取值）。"""
        if not np.isfinite(energy) or energy < 0.0:
            raise ValueError(f"Invalid RX energy: {energy}")
        self._comm_rx_energy_slot += energy
        self.total_comm_rx_energy += energy

    # 清空临时数据，为下一秒做准备
    def reset_for_next_step(self) -> None:
        """Reset UAV state for a new step."""
        self._current_covered_ues = []
        self._neighbors = []
        self._associated_service_load = 0
        self._execution_service_load = 0
        self._freq_counts = np.zeros(config.NUM_FILES)
        self._fly_energy_slot = 0.0
        self._compute_energy_slot = 0.0
        self._comm_tx_energy_slot = 0.0
        self._comm_rx_energy_slot = 0.0
        self._wpt_energy_slot = 0.0
        self._harvested_energy_slot = 0.0
        self.collision_violation = False
        self.boundary_violation = False
        self._served_latencies = []  # ✅ 新增：清空延迟记录

    def update_position(self, next_pos: np.ndarray) -> None:
        """
        Update the UAV's position to the new location chosen by the MARL agent.
        更新无人机位置
        """
        new_pos: np.ndarray = np.append(next_pos, config.UAV_ALTITUDE)  # 新的(x,y) 加上高度H
        self._dist_moved = float(np.linalg.norm(new_pos - self.pos))  # 记录移动距离
        self.pos = new_pos  # 跟新位置

    def set_neighbors(self, all_uavs: list[UAV]) -> None:
        """
        Set neighboring UAVs within sensing range for this UAV.
        找出感知范围内的邻居无人机
        """
        self._neighbors = []
        for other_uav in all_uavs:
            if other_uav.id != self.id:
                distance = float(np.linalg.norm(self.pos - other_uav.pos))
                if distance <= config.UAV_SENSING_RANGE:
                    self._neighbors.append(other_uav)

    def update_associated_service_load(self) -> None:
        self._associated_service_load = sum(
            1 for ue in self._current_covered_ues
            if ue.current_request[0] == 0
        )

    def prepare_request_slot(self) -> None:
        """时隙开始时统一初始化临时缓存和回传速率"""
        self._working_cache = self.cache.copy()
        self._uav_mbs_rate = comms.calculate_uav_mbs_rate(
            comms.calculate_channel_gain(self.pos, config.MBS_POS)
        )
        if not np.isfinite(self._uav_mbs_rate) or self._uav_mbs_rate <= 0.0:
            raise RuntimeError(
                f"Invalid UAV-MBS rate: UAV={self.id}, rate={self._uav_mbs_rate}"
            )

    def plan_requests(self) -> list[RequestPlan]:
        """生成卸载计划，不更新缓存、能耗、电池和频率"""
        plans: list[RequestPlan] = []
        shuffled_indices = np.random.permutation(len(self._current_covered_ues))
        for idx in shuffled_indices:
            ue = self._current_covered_ues[idx]
            req_type, _, _ = ue.current_request

            if req_type == 2:
                plans.append(RequestPlan(
                    source_uav=self, ue=ue, request_type=2,
                    ue_uav_rate=None, target_idx=None, target_uav=None,
                ))
                continue

            num_associated = len(self._current_covered_ues)
            if num_associated <= 0:
                raise RuntimeError(
                    f"UAV{self.id} plans a request without associated UEs."
                )

            ue_uav_rate = comms.calculate_ue_uav_rate(
                comms.calculate_channel_gain(ue.pos, self.pos),
                num_associated,
            )

            if not np.isfinite(ue_uav_rate) or ue_uav_rate <= 0.0:
                raise RuntimeError(
                    f"Invalid UE-UAV rate: UAV={self.id}, UE={ue.id}, rate={ue_uav_rate}"
                )

            target_idx, target_uav = self._decide_offloading_target(
                ue.current_request, ue_uav_rate
            )

            plans.append(RequestPlan(
                source_uav=self, ue=ue, request_type=req_type,
                ue_uav_rate=ue_uav_rate, target_idx=target_idx, target_uav=target_uav,
            ))

        return plans

    def execute_legacy_request_plan(self, plan: RequestPlan) -> None:
        """旧执行器：完整保留当前所有校验和副作用。仅 legacy 模式使用。"""
        if plan.source_uav is not self:
            raise RuntimeError("Request plan executed by the wrong source UAV.")

        if plan.request_type == 2:
            if plan.target_idx is not None:
                raise RuntimeError("Energy request should not have an offloading target.")
            self._process_energy_request(plan.ue)
            return

        if plan.request_type not in (0, 1):
            raise ValueError(f"Invalid request type: {plan.request_type}")
        if plan.target_idx not in (0, 1, 2):
            raise ValueError(f"Invalid offloading target: {plan.target_idx}")
        if plan.target_idx == 1 and plan.target_uav is None:
            raise RuntimeError("Neighbor offloading plan has no target UAV.")
        if plan.ue_uav_rate is None:
            raise RuntimeError("Service/content plan has no UE-UAV rate.")

        _, _, req_id = plan.ue.current_request

        self._freq_counts[req_id] += 1

        if plan.target_idx == 1:
            plan.target_uav._freq_counts[req_id] += 1

        if plan.request_type == 0:
            self._process_service_request(
                plan.ue, plan.ue_uav_rate, plan.target_idx, plan.target_uav
            )
        else:
            self._process_content_request(
                plan.ue, plan.ue_uav_rate, plan.target_idx, plan.target_uav
            )

    def execute_request_plan(self, plan: RequestPlan) -> None:
        """新执行器：从 RouteEstimate 统一记账。unified_* / iterative_* 模式使用。"""
        # === 保留旧执行器的全部前置校验 ===
        if plan.source_uav is not self:
            raise RuntimeError("Request plan executed by wrong source UAV.")

        if plan.request_type == 2:
            if plan.target_idx is not None:
                raise RuntimeError("Energy request should not have offloading target.")
            if plan.estimate is not None:
                raise RuntimeError("Energy request should not have RouteEstimate.")
            self._process_energy_request(plan.ue)
            return

        if plan.request_type not in (0, 1):
            raise ValueError(f"Invalid request type: {plan.request_type}")
        if plan.target_idx not in (0, 1, 2):
            raise ValueError(f"Invalid offloading target: {plan.target_idx}")
        if plan.target_idx == 1 and plan.target_uav is None:
            raise RuntimeError("Neighbor plan has no target UAV.")
        if plan.ue_uav_rate is None:
            raise RuntimeError("Service/content plan has no UE-UAV rate.")
        if plan.estimate is None:
            raise RuntimeError("Service/content plan must have RouteEstimate.")

        est = plan.estimate
        _, _, req_id = plan.ue.current_request

        # === 通信能耗 ===
        self.add_tx_energy_value(est.source_tx_energy)
        self.add_rx_energy_value(est.source_rx_energy)

        if plan.target_idx == 1:
            if plan.target_uav is None:
                raise RuntimeError("Neighbor plan has no target UAV.")
            plan.target_uav.add_tx_energy_value(est.target_tx_energy)
            plan.target_uav.add_rx_energy_value(est.target_rx_energy)

        # === 计算能耗 ===
        if plan.request_type == 0:
            if plan.target_idx == 0:
                self._compute_energy_slot += est.compute_energy
                self.total_compute_energy += est.compute_energy
            elif plan.target_idx == 1:
                plan.target_uav._compute_energy_slot += est.compute_energy
                plan.target_uav.total_compute_energy += est.compute_energy

        # === UE 电池 ===
        if plan.request_type == 0:
            plan.ue.update_battery(harv_energy=0.0, ue_transmit_time=est.ue_upload_time)
        elif plan.request_type == 1:
            plan.ue.update_battery(harv_energy=0.0, ue_transmit_time=0.0)

        # === 延迟归属 ===
        plan.ue.latency_current_request = est.total_latency
        self.add_served_latency(est.total_latency)

        # === 频率统计（验证 G 依赖） ===
        self._freq_counts[req_id] += 1
        if plan.target_idx == 1:
            plan.target_uav._freq_counts[req_id] += 1

        # === 缓存更新 ===
        if plan.request_type == 0:
            if plan.target_idx == 0:
                if not self.cache[req_id]:
                    _try_add_file_to_cache(self, req_id)
            elif plan.target_idx == 1:
                if not plan.target_uav.cache[req_id]:
                    _try_add_file_to_cache(plan.target_uav, req_id)
                _try_add_file_to_cache(self, req_id)
            elif plan.target_idx == 2:
                _try_add_file_to_cache(self, req_id)
        elif plan.request_type == 1:
            if plan.target_idx == 0:
                if not self.cache[req_id]:
                    _try_add_file_to_cache(self, req_id)
            elif plan.target_idx == 1:
                if not plan.target_uav.cache[req_id]:
                    _try_add_file_to_cache(plan.target_uav, req_id)
                _try_add_file_to_cache(self, req_id)
            elif plan.target_idx == 2:
                _try_add_file_to_cache(self, req_id)

    # 返回值 best_target_idx:
    # 0: 自己处理
    # 1: 卸载给其他无人机
    # 2: 卸载给基站
    # 3: 本地计算（用户自己算）

    # 核心决策算法 ！ 最关键的核心决策算法
    def _decide_offloading_target(self, current_req: tuple[int, int, int], ue_uav_rate: float) -> tuple[
        int, UAV | None]:
        """Returns (target_idx, target_uav_obj); Id: 0 = Local, 1 = Collaborating UAV, 2 = MBS
           返回值：(目标索引, 目标无人机标识)；其中，0表示本地无人机，1表示协作无人机，2表示多机器人系统中的其他无人机。
        """
        req_type, req_size, req_id = current_req
        file_size: int = config.FILE_SIZES[req_id]
        cpu_cycles: float = float(config.CPU_CYCLES_PER_BYTE[req_id]) * float(req_size) if req_type == 0 else -1.0

        # Associated UAV (Local) Expected Latency
        p_local: float = 1.0 if self.cache[req_id] else 0.0
        ue_uav_upload_latency: float = req_size / ue_uav_rate  # For service
        ue_uav_download_latency: float = file_size / ue_uav_rate  # For content
        exp_fetch_latency: float = (1.0 - p_local) * (file_size / self._uav_mbs_rate)  # For both
        exp_local_latency: float = exp_fetch_latency + ue_uav_download_latency  # For content
        if req_type == 0:  # Service
            local_estimated_load = max(self.associated_service_load, 1)
            est_comp_latency: float = cpu_cycles / (
                        config.UAV_COMPUTING_CAPACITY[self.id] / local_estimated_load)
            exp_local_latency = ue_uav_upload_latency + exp_fetch_latency + est_comp_latency  # Overwrite for service
            # ========== 在这里插入以下 6 行（补上本地无人机的因子修正） ==========
            # 获取本地无人机自己的类型匹配因子
            if self.type == config.UAV_TYPE_COMPUTE:
                factor = config.SERVICE_COMPUTE_FACTOR
            elif self.type == config.UAV_TYPE_STORAGE:
                factor = config.SERVICE_STORAGE_FACTOR
            else:
                factor = config.SERVICE_BALANCED_FACTOR
            # 用因子修正预期延迟（因子>1 代表加速，所以除以因子）
            exp_local_latency = exp_local_latency / factor
        # 在这里加
        best_exp_latency: float = exp_local_latency
        best_target_idx: int = 0
        best_target_uav: UAV | None = None

        # MBS Offloading Expected Latency
        uav_mbs_download_latency: float = file_size / self._uav_mbs_rate
        exp_mbs_latency: float = uav_mbs_download_latency + ue_uav_download_latency  # For content
        if req_type == 0:
            uav_mbs_upload_latency: float = req_size / self._uav_mbs_rate
            exp_mbs_latency = ue_uav_upload_latency + uav_mbs_upload_latency  # Overwrite for service

        if exp_mbs_latency < best_exp_latency:
            best_exp_latency = exp_mbs_latency
            best_target_idx = 2

        # Collaborating UAV Expected Latency  计算协作无人机的预期延迟
        for neighbor in self._neighbors:
            belief_prob: float = _get_belief_probability(req_id, neighbor.id)

            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, neighbor.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(
                comms.calculate_channel_gain(neighbor.pos, config.MBS_POS))
            uav_uav_download_latency: float = file_size / uav_uav_rate
            exp_neighbor_fetch_latency: float = (1.0 - belief_prob) * (file_size / uav_mbs_rate)  # For both
            exp_neighbor_latency: float = exp_neighbor_fetch_latency + uav_uav_download_latency + ue_uav_download_latency  # For content
            if req_type == 0:  # Service
                neigh_load: int = max(neighbor.associated_service_load + 1, 1)
                est_comp_latency: float = cpu_cycles / (config.UAV_COMPUTING_CAPACITY[neighbor.id] / neigh_load)
                uav_uav_upload_latency: float = req_size / uav_uav_rate
                exp_neighbor_latency = ue_uav_upload_latency + uav_uav_upload_latency + exp_neighbor_fetch_latency + est_comp_latency

            # 能力感知因子调整（先定义默认因子，再覆盖）
            factor = 1.0
            if req_type == 0:  # service
                if neighbor.type == config.UAV_TYPE_COMPUTE:
                    factor = config.SERVICE_COMPUTE_FACTOR
                elif neighbor.type == config.UAV_TYPE_STORAGE:
                    factor = config.SERVICE_STORAGE_FACTOR
                else:
                    factor = config.SERVICE_BALANCED_FACTOR
            else:  # content (req_type == 1)
                if neighbor.type == config.UAV_TYPE_STORAGE:
                    factor = config.CONTENT_STORAGE_FACTOR
                elif neighbor.type == config.UAV_TYPE_COMPUTE:
                    factor = config.CONTENT_COMPUTE_FACTOR
                else:
                    factor = config.CONTENT_BALANCED_FACTOR

            adjusted_latency = exp_neighbor_latency / factor

            if adjusted_latency < best_exp_latency:
                best_exp_latency = adjusted_latency
                best_target_idx = 1
                best_target_uav = neighbor

        assert best_exp_latency >= 0.0
        return best_target_idx, best_target_uav

    def _process_service_request(self, ue: UE, ue_uav_rate: float, target_idx: int, target_uav: UAV | None) -> None:
        _, req_size, req_id = ue.current_request
        assert req_id < config.NUM_SERVICES
        cpu_cycles: float = float(config.CPU_CYCLES_PER_BYTE[req_id]) * float(req_size)
        file_size: int = config.FILE_SIZES[req_id]

        ue_uav_upload_latency: float = req_size / ue_uav_rate
        ue.update_battery(0.0, ue_uav_upload_latency)
        self.add_rx_energy(ue_uav_upload_latency)

        if target_idx == 0:  # Associated UAV
            fetch_latency: float = 0.0
            if not self.cache[req_id]:
                fetch_latency = file_size / self._uav_mbs_rate
                self.add_rx_energy(fetch_latency)
                _try_add_file_to_cache(self, req_id)

            comp_latency, comp_energy = _get_computing_latency_and_energy(self, cpu_cycles)

            # ========== 能力匹配因子（本地无人机） ==========
            if self.type == config.UAV_TYPE_COMPUTE:
                factor = config.SERVICE_COMPUTE_FACTOR  # 1.2 (加速)
            elif self.type == config.UAV_TYPE_STORAGE:
                factor = config.SERVICE_STORAGE_FACTOR  # 0.8 (减速)
            else:  # BALANCED
                factor = config.SERVICE_BALANCED_FACTOR  # 1.0 (不变)
            comp_latency = comp_latency / factor

            ue.latency_current_request = ue_uav_upload_latency + fetch_latency + comp_latency
            self.add_served_latency(ue.latency_current_request)
            self._compute_energy_slot += comp_energy
            self.total_compute_energy += comp_energy

        elif target_idx == 1:  # Collaborating UAV
            assert target_uav is not None
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, target_uav.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(
                comms.calculate_channel_gain(target_uav.pos, config.MBS_POS))
            uav_uav_upload_latency: float = req_size / uav_uav_rate

            self.add_tx_energy(uav_uav_upload_latency)
            target_uav.add_rx_energy(uav_uav_upload_latency)

            fetch_latency: float = 0.0
            if not target_uav.cache[req_id]:
                fetch_latency = file_size / uav_mbs_rate
                target_uav.add_rx_energy(fetch_latency)
                _try_add_file_to_cache(target_uav, req_id)

            comp_latency, comp_energy = _get_computing_latency_and_energy(target_uav, cpu_cycles)

            # ========== 能力匹配因子（协作无人机） ==========
            if target_uav.type == config.UAV_TYPE_COMPUTE:
                factor = config.SERVICE_COMPUTE_FACTOR  # 1.2 (加速)
            elif target_uav.type == config.UAV_TYPE_STORAGE:
                factor = config.SERVICE_STORAGE_FACTOR  # 0.8 (减速)
            else:  # BALANCED
                factor = config.SERVICE_BALANCED_FACTOR  # 1.0 (不变)
            comp_latency = comp_latency / factor

            ue.latency_current_request = ue_uav_upload_latency + uav_uav_upload_latency + fetch_latency + comp_latency
            self.add_served_latency(ue.latency_current_request)
            target_uav._compute_energy_slot += comp_energy
            target_uav.total_compute_energy += comp_energy
            _try_add_file_to_cache(self, req_id)

        else:  # MBS
            uav_mbs_upload_latency: float = req_size / self._uav_mbs_rate
            self.add_tx_energy(uav_mbs_upload_latency)
            ue.latency_current_request = ue_uav_upload_latency + uav_mbs_upload_latency
            self.add_served_latency(ue.latency_current_request)
            _try_add_file_to_cache(self, req_id)

    def _process_content_request(self, ue: UE, ue_uav_rate: float, target_idx: int, target_uav: UAV | None) -> None:
        req_id: int = ue.current_request[2]
        assert req_id >= config.NUM_SERVICES
        file_size: int = config.FILE_SIZES[req_id]

        ue_uav_download_latency: float = file_size / ue_uav_rate
        ue.update_battery(0.0, 0.0)
        self.add_tx_energy(ue_uav_download_latency)

        if target_idx == 0:  # Associated UAV
            fetch_latency: float = 0.0
            if not self.cache[req_id]:
                fetch_latency = file_size / self._uav_mbs_rate
                self.add_rx_energy(fetch_latency)
                _try_add_file_to_cache(self, req_id)

            ue.latency_current_request = fetch_latency + ue_uav_download_latency
            self.add_served_latency(ue.latency_current_request)

        elif target_idx == 1:  # Collaborating UAV
            assert target_uav is not None
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, target_uav.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(
                comms.calculate_channel_gain(target_uav.pos, config.MBS_POS))
            uav_uav_download_latency: float = file_size / uav_uav_rate

            target_uav.add_tx_energy(uav_uav_download_latency)
            self.add_rx_energy(uav_uav_download_latency)

            fetch_latency: float = 0.0
            if not target_uav.cache[req_id]:
                fetch_latency = file_size / uav_mbs_rate
                target_uav.add_rx_energy(fetch_latency)
                _try_add_file_to_cache(target_uav, req_id)

            ue.latency_current_request = fetch_latency + uav_uav_download_latency + ue_uav_download_latency
            self.add_served_latency(ue.latency_current_request)
            _try_add_file_to_cache(self,
                                   req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

        else:  # MBS
            uav_mbs_download_latency: float = file_size / self._uav_mbs_rate
            self.add_rx_energy(uav_mbs_download_latency)
            ue.latency_current_request = uav_mbs_download_latency + ue_uav_download_latency
            self.add_served_latency(ue.latency_current_request)
            _try_add_file_to_cache(self,req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

    def _process_energy_request(self, ue: UE) -> None:
        """Process an emergency energy request from a UE."""
        actual_harvested = ue.update_battery(
            harv_energy=config.HARVEST_ENERGY_PER_REQUEST,
            ue_transmit_time=0.0,
        )
        wpt_cost = actual_harvested / config.WPT_EFFICIENCY
        self._wpt_energy_slot += wpt_cost
        self._harvested_energy_slot += actual_harvested
        self.total_wpt_tx_energy += wpt_cost
        ue.latency_current_request = 0.0  # No latency deadline for energy requests
        self.add_served_latency(0.0)

    # 无人机学习用户偏好的核心函数
    def update_ema_and_cache(self) -> None:
        """Update EMA scores and cache reactively."""
        # 新EMA = α × 当前请求数 + (1 - α) × 旧EMA
        # 其中 α = GDSF_SMOOTHING_FACTOR = 0.75
        self._ema_scores = config.GDSF_SMOOTHING_FACTOR * self._freq_counts + (
                    1 - config.GDSF_SMOOTHING_FACTOR) * self._ema_scores
        self.cache = self._working_cache.copy()  # Update cache after processing all requests of all UAVs 更新缓存：把处理过程中用的临时缓存变成正式缓存

    def gdsf_cache_update(self) -> None:  # 缓存更新函数
        """Update cache using the GDSF caching policy at a longer timescale."""
        priority_scores = self._ema_scores / config.FILE_SIZES
        sorted_file_ids = np.argsort(-priority_scores)
        self.cache = np.zeros(config.NUM_FILES, dtype=bool)
        used_space = 0.0
        for file_id in sorted_file_ids:
            file_size = config.FILE_SIZES[file_id]
            if used_space + file_size <= self.storage_capacity:
                self.cache[file_id] = True
                used_space += file_size
            else:
                break

    # 无人机能量会计的核心函数
    #   def update_energy_consumption(self) -> None:
    #       """Update UAV energy consumption for the current time slot."""
    #      time_moving = self._dist_moved / config.UAV_SPEED
    #       time_hovering = config.TIME_SLOT_DURATION - time_moving
    #       fly_energy = config.POWER_MOVE * time_moving + config.POWER_HOVER * time_hovering

    # 累加飞行能耗
    #       self.total_fly_energy += fly_energy
    #       self._energy_current_slot += fly_energy

    #      has_energy_request = any(ue.current_request[0] == 2 for ue in self._current_covered_ues)
    #       if has_energy_request:
    #           wpt_energy = config.WPT_TRANSMIT_POWER * config.TIME_SLOT_DURATION
    #           self.total_wpt_tx_energy += wpt_energy
    #           self._energy_current_slot += wpt_energy

    # self._energy_current_slot += fly_energy
    # has_energy_request = any(ue.current_request[0] == 2 for ue in self._current_covered_ues)
    # if has_energy_request:
    #    self._energy_current_slot += config.WPT_TRANSMIT_POWER * config.TIME_SLOT_DURATION

    def update_energy_consumption(self) -> None:
        time_moving = self._dist_moved / config.UAV_SPEED
        time_hovering = max(0.0, config.TIME_SLOT_DURATION - time_moving)
        self._fly_energy_slot = (
            config.POWER_MOVE * time_moving
            + config.POWER_HOVER * time_hovering
        )
        self.total_fly_energy += self._fly_energy_slot