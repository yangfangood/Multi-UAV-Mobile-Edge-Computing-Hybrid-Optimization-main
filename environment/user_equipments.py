import config
import numpy as np


class UE:
    all_ids: np.ndarray # 所有文件ID
    global_ranks: np.ndarray # 每个文件的流行度排名[23, 5, 67, ...]
    id_to_rank_map: dict[int, int] # {文件ID: 排名}

    @classmethod
    def initialize_ue_class(cls) -> None:
        # 1. 创建文件ID列表 (0-74)
        cls.all_ids = np.arange(config.NUM_FILES)  # Assume IDs 0 to NUM_SERVICES-1 are Services, rest are Contents
        # 2. 创建随机排名
        cls.global_ranks = np.arange(1, config.NUM_FILES + 1)
        np.random.shuffle(cls.global_ranks)  # Currently random ranks assigned   # 随机打乱 → [23,5,67,...]
        # 3. 建立ID到排名的映射
        cls.id_to_rank_map = dict(zip(cls.all_ids, cls.global_ranks))  # Mapping from ID to rank
        # 4. 计算Zipf分布概率 Zipf定律：**排名越靠前的文件，被请求的概率越高**。
        zipf_denom: float = np.sum(1 / cls.global_ranks**config.ZIPF_BETA)
        cls.global_probs: np.ndarray = (1 / cls.global_ranks**config.ZIPF_BETA) / zipf_denom

    def __init__(self, ue_id: int) -> None:
        self.id: int = ue_id
        # 1. 位置：随机分布在700×700米区域
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), 0.0])
        self.battery_level: float = np.random.uniform(0.6, 1.0) * config.UE_BATTERY_CAPACITY  # Start at capacity between 60% to 100%
        # 3. 当前请求状态
        self.current_request: tuple[int, int, int] = (0, 0, 0)  # Request : (req_type, req_size, req_id)
        self.latency_current_request: float = 0.0  # Latency for the current request 当前请求的延迟时间
        self.assigned: bool = False # 是否有无人机服务

        # Random Waypoint Model 移动模型
        self._waypoint: np.ndarray    # 目标点
        self._wait_time: int   # 等待时间
        self._set_new_waypoint()  # Initialize first waypoint 设置第一个目标点

        # Fairness Tracking 5. 公平性跟踪
        self._successful_requests: int = 0
        self.service_coverage: float = 0.0
    #移动模型  用户采用随机路点模型移动
    def update_position(self) -> None:
        """Updates the UE's position for one time slot as per the Random Waypoint model.
           每个时间步更新用户位置"""
        #如果还在等待，就原地不动
        if self._wait_time > 0:
            self._wait_time -= 1
            return

        # 计算到目标点的方向
        direction_vec: np.ndarray = self._waypoint - self.pos[:2]
        # 计算到目标点的距离
        distance_to_waypoint: float = float(np.linalg.norm(direction_vec))
        # 如果已经到了目标点
        if config.UE_MAX_DIST >= distance_to_waypoint:  # Reached the waypoint
            self.pos[:2] = self._waypoint # 到达
            self._set_new_waypoint()      # 设置新的目标点
        else:  # Move towards the waypoint
            # 向目标点移动一步（最多15米）
            move_vector = (direction_vec / distance_to_waypoint) * config.UE_MAX_DIST
            self.pos[:2] += move_vector

    def generate_request(self) -> None:
        """Generates a new request tuple for the current time slot.
           生成当前时间步的任务请求
        """

        # Check for Emergency Energy Request  1. 紧急能量请求--电量低于30%
        if self.battery_level < config.UE_CRITICAL_THRESHOLD:
            self.current_request = (2, 0, 0) # 类型2：能量请求
            self.latency_current_request = 0.0 # 请求时延设置为0
            self.assigned = False
            return

        # 正常任务请求
        req_id: int = np.random.choice(UE.all_ids, p=UE.global_probs) # 按Zipf选文件
        req_type: int = 0 if req_id < config.NUM_SERVICES else 1 # 类型0=服务 ， 类型1=内容
        req_size: int = np.random.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE) if req_type == 0 else 0 #只有服务请求有大小
        self.current_request = (req_type, req_size, req_id)
        self.latency_current_request = 0.0
        self.assigned = False

    #计算用户被成功服务的比例，也就是"服务质量覆盖率"。这是衡量系统公平性的关键指标。
    def update_service_coverage(self, current_time_step_t: int) -> None:
        """Updates the fairness metric based on service outcome in the current slot.
           更新服务覆盖率，用于公平性计算、
           两个条件同时满足才算一次 成功服务：（不知要被服务，还要服务好）
           1. 有无人机服务   (`assigned=True`)
           2. 延迟在1秒内    (`latency ≤ 1s`)
        """
        # 如果这次被服务 并且 时延<=1秒，算一次成功
        if self.assigned and self.latency_current_request <= config.TIME_SLOT_DURATION:
            self._successful_requests += 1
        # 断言当前时间步大于0
        assert current_time_step_t > 0
        self.service_coverage = self._successful_requests / current_time_step_t #服务覆盖率 = 成功服务次数 / 已过去的时间步

    # 电池管理
    def _set_new_waypoint(self):
        """Set a new destination, speed, and wait time as per the Random Waypoint model.
        设置新的随机目标点和等待时间
           """

        self._waypoint = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)])

        self._wait_time = np.random.randint(0, config.UE_MAX_WAIT_TIME + 1)
    #电池更新逻辑 harv_energy：从无人机无线充电获得的能量（焦耳） ue_transmit_time：用户传输任务的时间（秒）
    # （上限100J，下限0J）
    def update_battery(self, harv_energy: float, ue_transmit_time: float) -> None:
        """Updates battery level based on consumption and harvesting.更新电量 ==消耗+获取"""
        # 计算消耗的能量
        consumed_energy: float = config.UE_STATIC_POWER * config.TIME_SLOT_DURATION
        # 更新电量
        consumed_energy += config.TRANSMIT_POWER * ue_transmit_time
        self.battery_level = min(config.UE_BATTERY_CAPACITY, self.battery_level - consumed_energy + harv_energy) #上限100J ，当前电量 - 消耗 + 获得
        self.battery_level = max(0.0, self.battery_level) # 下限0J，不能为负
