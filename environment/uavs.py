from __future__ import annotations
from environment.user_equipments import UE
from environment import comm_model as comms
import config
import numpy as np

# 用逻辑回归 估计邻居缓存某个文件的概率，用于卸载决策
def _get_belief_probability(file_id: int, neighbor_id: int) -> float:
    """Returns the estimated probability P_{v,i} that a neighbor has file_i."""
    rank = UE.id_to_rank_map[file_id] # 文件流行度排名
    c_hat_v: float = config.UAV_STORAGE_CAPACITY[neighbor_id] / config.AVG_FILE_SIZE #邻居能存多少文件
    exponent: float = config.PROB_GAMMA * (rank - c_hat_v) # γ=0.5
    probability: float = 1.0 / (1.0 + np.exp(exponent)) # Sigmoid函数
    return probability

# 计算无人机 处理任务的计算时延和能耗
def _get_computing_latency_and_energy(uav: UAV, cpu_cycles: float) -> tuple[float, float]:
    """Calculate computing latency and energy for a UAV processing request."""
    assert uav._current_service_request_count > 0 # 断言 当前无人机的服务队列是否为空
    # 当前无人机的计算能力/需要服务的任务数量
    computing_capacity_per_request: float = config.UAV_COMPUTING_CAPACITY[uav.id] / uav._current_service_request_count
    latency: float = cpu_cycles / computing_capacity_per_request # 延迟 = 计算量 / 算力
    energy: float = config.K_CPU * cpu_cycles * (computing_capacity_per_request**2) # 能耗 = κ × 计算量 × 算力²
    return latency, energy


def _try_add_file_to_cache(uav: UAV, file_id: int) -> None:
    """Try to add a file to UAV cache if there's enough space."""
    if uav._working_cache[file_id]:
        return  # Already in cache
    used_space: int = np.sum(uav._working_cache * config.FILE_SIZES) # 已用空间
    if used_space + config.FILE_SIZES[file_id] <= uav.storage_capacity:
        uav._working_cache[file_id] = True # 空间够就加入


class UAV:
    # 初始化无人机属性
    def __init__(self, uav_id: int) -> None:
        self.id: int = uav_id #唯一标识
        #获取类型
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
                                         np.random.uniform(0, config.AREA_HEIGHT), # y坐标 0-700米随机
                                         config.UAV_ALTITUDE]) # 固定高度100米

        self._dist_moved: float = 0.0  # Distance moved in the current time slot 当前时隙移动的距离
        self._current_covered_ues: list[UE] = [] #当前无人机对应的ue队列
        self._neighbors: list[UAV] = [] # 相邻无人机
        self._current_service_request_count: int = 0 # 当前无人机服务请求的数量
        self._energy_current_slot: float = 0.0  # Energy consumed for this time slot 当前时隙无人机的能量
        self.collision_violation: bool = False  # Track if UAV has violated minimum separation 跟踪无人机是否违反了最小碰撞距离
        self.boundary_violation: bool = False  # Track if UAV has gone out of bounds 跟踪无人机是否越界

        # Cache and request tracking 无人机物理存储了哪些文件，可以直接服务用户
        self.cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool) # 物理缓存 哪些文件真的存了
        self._working_cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool) #临时缓存，处理请求时用
        self._freq_counts: np.ndarray = np.zeros(config.NUM_FILES) # 当前时间步各文件被请求次数
        self._ema_scores: np.ndarray = np.zeros(config.NUM_FILES) #指数移动平均，平滑后的文件热度

        self._uav_mbs_rate: float = 0.0 #无人机到基站的传输速率

    @property
    def energy(self) -> float:
        return self._energy_current_slot

    @property
    def current_covered_ues(self) -> list[UE]:
        return self._current_covered_ues

    @property
    def neighbors(self) -> list[UAV]:
        return self._neighbors
    # 清空临时数据，为下一秒做准备
    def reset_for_next_step(self) -> None:
        """Reset UAV state for a new step."""
        self._current_covered_ues = []
        self._neighbors = []
        self._current_service_request_count = 0
        self._freq_counts = np.zeros(config.NUM_FILES)
        self._energy_current_slot = 0.0
        self.collision_violation = False
        self.boundary_violation = False

    def update_position(self, next_pos: np.ndarray) -> None:
        """
        Update the UAV's position to the new location chosen by the MARL agent.
        更新无人机位置
        """
        new_pos: np.ndarray = np.append(next_pos, config.UAV_ALTITUDE) # 新的(x,y) 加上高度H
        self._dist_moved = float(np.linalg.norm(new_pos - self.pos)) # 记录移动距离
        self.pos = new_pos # 跟新位置

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
    # 统计当前覆盖的用户中，有多少服务请求。
    def calculate_initial_load(self) -> None:
        for ue in self._current_covered_ues: #遍历当前覆盖的所有用户
            if ue.current_request[0] == 0:  # Service 若请求类型为Service
                self._current_service_request_count += 1 #请求服务计数器+1
        #current_request是一个元组：type=0 请求服务 type=1 只需要下载的内容 type=2 空请求

    def process_requests(self) -> None:  #真正去处理上面的请求的函数
        """
        Process Requests using Probabilistic Decisions with Optimistic Relief.
        使用 具有乐观补偿机制的概率决策方法 处理请求  核心！！
        """
        self._working_cache = self.cache.copy() #备份当前缓存
        self._uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(self.pos, config.MBS_POS)) #无人机到基站的速度
        #2. 随机打乱用户顺序，避免偏见
        shuffled_indices: np.ndarray = np.random.permutation(len(self._current_covered_ues))
        #3. 逐个处理每个用户
        for idx in shuffled_indices:
            ue: UE = self._current_covered_ues[idx]

            req_type, _, req_id = ue.current_request
            #4. 根据请求类型分流
            if req_type == 2:
                self._process_energy_request(ue)
                continue
            # 5. 计算uav-ue传输速率
            ue_uav_rate: float = comms.calculate_ue_uav_rate(comms.calculate_channel_gain(ue.pos, self.pos), len(self._current_covered_ues))
            # 6. 决策：这个任务去哪处理
            # best_target_idx  决策索引：0,1,2,3 代表不同目标
            #best_target_uav   如果选其他无人机，这里返回那个无人机对象
            best_target_idx, best_target_uav = self._decide_offloading_target(ue.current_request, ue_uav_rate)# (用户的当前请求 (type, size, id)，用户到当前无人机的传输速率)
            # 7.更新统计信息
            self._freq_counts[req_id] += 1  # I got a request for this file 记录这个文件被请求了


            if best_target_idx == 1 and best_target_uav is not None:  # Request also seen by collaborating UAV 请求被其他无人机看到了
                best_target_uav._freq_counts[req_id] += 1
            # 8. 乐观卸载处理（重要！）
            if req_type == 0: #只处理”服务请求“  req_type请求类型
                if best_target_idx != 0: # 如果决定不自己处理（卸载给别人）
                    # OPTIMISTIC RELIEF: I was counted in 'calculate_initial_load', but I am leaving. Decrement so next user sees smaller queue.
                    self._current_service_request_count = max(0, self._current_service_request_count - 1)
                    # 第2步 ：如果卸载给其他无人机
                    if best_target_idx == 1 and best_target_uav is not None:
                        # 那就给那个无人机加1，表示他的负载增加了
                        best_target_uav._current_service_request_count += 1
            # 9. 根据类型处理
            if req_type == 0:
                self._process_service_request(ue, ue_uav_rate, best_target_idx, best_target_uav)
            else:
                self._process_content_request(ue, ue_uav_rate, best_target_idx, best_target_uav)

            assert ue.latency_current_request >= 0.0
    # 返回值 best_target_idx:
    # 0: 自己处理
    # 1: 卸载给其他无人机
    # 2: 卸载给基站
    # 3: 本地计算（用户自己算）

    # 核心决策算法 ！ 最关键的核心决策算法
    def _decide_offloading_target(self, current_req: tuple[int, int, int], ue_uav_rate: float) -> tuple[int, UAV | None]:
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
            assert self._current_service_request_count > 0
            est_comp_latency: float = cpu_cycles / (config.UAV_COMPUTING_CAPACITY[self.id] / self._current_service_request_count)
            exp_local_latency = ue_uav_upload_latency + exp_fetch_latency + est_comp_latency  # Overwrite for service

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

        # Collaborating UAV Expected Latency
        for neighbor in self._neighbors:
            belief_prob: float = _get_belief_probability(req_id, neighbor.id)

            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, neighbor.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(neighbor.pos, config.MBS_POS))
            uav_uav_download_latency: float = file_size / uav_uav_rate
            exp_neighbor_fetch_latency: float = (1.0 - belief_prob) * (file_size / uav_mbs_rate)  # For both
            exp_neighbor_latency: float = exp_neighbor_fetch_latency + uav_uav_download_latency + ue_uav_download_latency  # For content
            if req_type == 0:  # Service
                # Neighbor Load: They broadcasted 'initial_load'. We add +1 because "If I come, I add to the pile."
                neigh_load: int = neighbor._current_service_request_count + 1
                assert neigh_load > 0
                est_comp_latency: float = cpu_cycles / (config.UAV_COMPUTING_CAPACITY[neighbor.id] / neigh_load)
                uav_uav_upload_latency: float = req_size / uav_uav_rate
                exp_neighbor_latency = ue_uav_upload_latency + uav_uav_upload_latency + exp_neighbor_fetch_latency + est_comp_latency  # Overwrite for service

            if exp_neighbor_latency < best_exp_latency:
                best_exp_latency = exp_neighbor_latency
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
        if target_idx == 0:  # Associated UAV
            fetch_latency: float = 0.0
            if not self.cache[req_id]:
                fetch_latency = file_size / self._uav_mbs_rate
                _try_add_file_to_cache(self, req_id)

            comp_latency, comp_energy = _get_computing_latency_and_energy(self, cpu_cycles)
            ue.latency_current_request = ue_uav_upload_latency + fetch_latency + comp_latency
            self._energy_current_slot += comp_energy

        elif target_idx == 1:  # Collaborating UAV
            assert target_uav is not None
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, target_uav.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(target_uav.pos, config.MBS_POS))
            uav_uav_upload_latency: float = req_size / uav_uav_rate

            fetch_latency: float = 0.0
            if not target_uav.cache[req_id]:
                fetch_latency = file_size / uav_mbs_rate
                _try_add_file_to_cache(target_uav, req_id)

            comp_latency, comp_energy = _get_computing_latency_and_energy(target_uav, cpu_cycles)
            ue.latency_current_request = ue_uav_upload_latency + uav_uav_upload_latency + fetch_latency + comp_latency
            target_uav._energy_current_slot += comp_energy
            _try_add_file_to_cache(self, req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

        else:  # MBS
            uav_mbs_upload_latency: float = req_size / self._uav_mbs_rate
            ue.latency_current_request = ue_uav_upload_latency + uav_mbs_upload_latency
            _try_add_file_to_cache(self, req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

    def _process_content_request(self, ue: UE, ue_uav_rate: float, target_idx: int, target_uav: UAV | None) -> None:
        req_id: int = ue.current_request[2]
        assert req_id >= config.NUM_SERVICES
        file_size: int = config.FILE_SIZES[req_id]

        ue_uav_download_latency: float = file_size / ue_uav_rate
        ue.update_battery(0.0, 0.0)
        if target_idx == 0:  # Associated UAV
            fetch_latency: float = 0.0
            if not self.cache[req_id]:
                fetch_latency = file_size / self._uav_mbs_rate
                _try_add_file_to_cache(self, req_id)

            ue.latency_current_request = fetch_latency + ue_uav_download_latency

        elif target_idx == 1:  # Collaborating UAV
            assert target_uav is not None
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, target_uav.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(target_uav.pos, config.MBS_POS))
            uav_uav_download_latency: float = file_size / uav_uav_rate

            fetch_latency: float = 0.0
            if not target_uav.cache[req_id]:
                fetch_latency = file_size / uav_mbs_rate
                _try_add_file_to_cache(target_uav, req_id)

            ue.latency_current_request = fetch_latency + uav_uav_download_latency + ue_uav_download_latency
            _try_add_file_to_cache(self, req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

        else:  # MBS
            uav_mbs_download_latency: float = file_size / self._uav_mbs_rate
            ue.latency_current_request = uav_mbs_download_latency + ue_uav_download_latency
            _try_add_file_to_cache(self, req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

    def _process_energy_request(self, ue: UE) -> None:
        """Process an emergency energy request from a UE."""
        channel_gain: float = comms.calculate_channel_gain(self.pos, ue.pos)
        harv_energy: float = config.WPT_EFFICIENCY * config.WPT_TRANSMIT_POWER * channel_gain * config.TIME_SLOT_DURATION
        ue.update_battery(harv_energy, 0.0)
        ue.latency_current_request = 0.0  # No latency deadline for energy requests
    #无人机学习用户偏好的核心函数
    def update_ema_and_cache(self) -> None:
        """Update EMA scores and cache reactively."""
        # 新EMA = α × 当前请求数 + (1 - α) × 旧EMA
        # 其中 α = GDSF_SMOOTHING_FACTOR = 0.75
        self._ema_scores = config.GDSF_SMOOTHING_FACTOR * self._freq_counts + (1 - config.GDSF_SMOOTHING_FACTOR) * self._ema_scores
        self.cache = self._working_cache.copy()  # Update cache after processing all requests of all UAVs 更新缓存：把处理过程中用的临时缓存变成正式缓存

    def gdsf_cache_update(self) -> None: #缓存更新函数
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

    #无人机能量会计的核心函数
    def update_energy_consumption(self) -> None:
        """Update UAV energy consumption for the current time slot."""
        time_moving = self._dist_moved / config.UAV_SPEED
        time_hovering = config.TIME_SLOT_DURATION - time_moving
        fly_energy = config.POWER_MOVE * time_moving + config.POWER_HOVER * time_hovering
        self._energy_current_slot += fly_energy
        has_energy_request = any(ue.current_request[0] == 2 for ue in self._current_covered_ues)
        if has_energy_request:
            self._energy_current_slot += config.WPT_TRANSMIT_POWER * config.TIME_SLOT_DURATION
