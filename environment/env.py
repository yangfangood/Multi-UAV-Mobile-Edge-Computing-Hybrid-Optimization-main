from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np
from environment.obstacle import StaticObstacle, DynamicObstacle  # 新增导入


class Env:
    def __init__(self) -> None: #构造函数：初始化环境
        self._mbs_pos: np.ndarray = config.MBS_POS #基站位置
        UE.initialize_ue_class() #初始化用户类
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)] #创建100个用户---UE(i)创建了一个实例
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)] #创建5个无人机
        self._time_step: int = 0 #时间步计数器 初始为0，表示仿真刚开始
        #每个step()被调用一次，_time_step就加1

        # ========== 新增：初始化障碍物 ==========
        self._static_obstacles: list = []  # 静态障碍物列表
        self._dynamic_obstacles: list = []  # 动态障碍物列表
        self._init_obstacles()  # 创建障碍物

        # 避障统计（可选，用于日志）
        self._collision_count: int = 0  # 累计碰撞次数
        self._avoidance_count: int = 0  # 累计避障次数

    def _init_obstacles(self) -> None:
        """
        初始化静态和动态障碍物
        位置随机生成，保证不重叠
        """
        np.random.seed(config.SEED)  # 可复现

        # ----- 1. 创建静态障碍物（建筑物）-----
        for i in range(config.NUM_STATIC_OBSTACLES):
            # 随机位置，避免太靠近边界（留50米缓冲区）
            pos_x = np.random.uniform(50, config.AREA_WIDTH - 50)
            pos_y = np.random.uniform(50, config.AREA_HEIGHT - 50)
            pos = np.array([pos_x, pos_y])

            # 检查是否与已有障碍物重叠（简单避免）
            overlap = False
            for obs in self._static_obstacles:
                if np.linalg.norm(obs.pos - pos) < (obs.radius + config.STATIC_OBSTACLE_RADIUS + 20):
                    overlap = True
                    break

            if not overlap:
                obs = StaticObstacle(i, pos, config.STATIC_OBSTACLE_RADIUS)
                self._static_obstacles.append(obs)

        # ----- 2. 创建动态障碍物（移动物体）-----
        for i in range(config.NUM_DYNAMIC_OBSTACLES):
            # 随机位置
            pos_x = np.random.uniform(0, config.AREA_WIDTH)
            pos_y = np.random.uniform(0, config.AREA_HEIGHT)
            pos = np.array([pos_x, pos_y])

            # 随机速度
            vx = np.random.uniform(*config.DYNAMIC_OBSTACLE_SPEED_RANGE)
            vy = np.random.uniform(*config.DYNAMIC_OBSTACLE_SPEED_RANGE)
            velocity = np.array([vx, vy])

            obs = DynamicObstacle(
                i + config.NUM_STATIC_OBSTACLES,
                pos,
                config.DYNAMIC_OBSTACLE_RADIUS,
                velocity,
                (config.AREA_WIDTH, config.AREA_HEIGHT)
            )
            self._dynamic_obstacles.append(obs)

        # 打印障碍物信息
        print(f"🏗️ 初始化完成: {len(self._static_obstacles)}个静态障碍物, "
              f"{len(self._dynamic_obstacles)}个动态障碍物")

    @property
    def obstacles(self):
        """返回所有障碍物列表"""
        return self._static_obstacles + self._dynamic_obstacles

    def _check_obstacle_collisions(self) -> None:
        """
        检测无人机与障碍物的碰撞
        如果碰撞：
            1. 标记违规（用于惩罚）
            2. 推开无人机（物理反应）
        """
        for uav in self._uavs:
            for obs in self.obstacles:
                # 检测碰撞
                if obs.check_collision(uav.pos, config.UAV_COVERAGE_RADIUS):
                    # 标记碰撞违规
                    uav.collision_violation = True
                    self._collision_count += 1

                    # 计算推开向量
                    push = obs.get_push_vector(uav.pos, config.UAV_COVERAGE_RADIUS)

                    # 计算新位置
                    new_pos = uav.pos[:2] + push

                    # 确保在边界内
                    new_pos = np.clip(new_pos, 0,
                                      [config.AREA_WIDTH, config.AREA_HEIGHT])

                    # 更新无人机位置
                    uav.update_position(new_pos)
                    self._avoidance_count += 1



    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    def reset(self) -> list[np.ndarray]: #重置环境，开始新的回合，相当于游戏重新开始
        """Resets the environment to an initial state and returns the initial observations."""
        self._ues = [UE(i) for i in range(config.NUM_UES)]
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step = 0

        # 注意：障碍物位置不变（可选：想重置就取消注释）
        # self._init_obstacles()

        # 重置统计
        self._collision_count = 0
        self._avoidance_count = 0

        return self._get_obs() #返回 初始观测

    #动作执行，返回新状态和奖励！核心函数
    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], list[float], tuple[float, float, float, float]]:#actions是动作，不是随机生成的，来自神经网络
        """Execute one time step of the simulation."""
        self._time_step += 1
        #阶段1：无人机处理任务--计算初始负载
        for uav in self._uavs:
            uav.calculate_initial_load() #计算当前覆盖的用户中有多少服务请求

        for uav in self._uavs:
            uav.process_requests()  #处理任务请求
        #阶段2：用户更新状态
        for ue in self._ues:
            if not ue.assigned:#没被服务的用户
                ue.update_battery(0.0, 0.0)#只消耗电量（没有能量补充）
            ue.update_service_coverage(self._time_step) #更新服务覆盖记录，计算用户被成功服务的比例，也就是"服务质量覆盖率"。这是衡量系统公平性的关键指标。
        #阶段3：无人机更新
        for uav in self._uavs:
            uav.update_ema_and_cache() #更新缓存策略 ema：指数移动平均
            uav.update_energy_consumption() #更新能耗
        #阶段4：计算奖励
        rewards, metrics = self._get_rewards_and_metrics()
        #阶段5：定期更新缓存策略
        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0: #每50个时间步（50秒）执行一次缓存更新
            for uav in self._uavs:
                uav.gdsf_cache_update()#GDSF算法更新缓存，考虑了无人机之间的协作

        # 阶段6：For next time step 为下一步做准备
        for ue in self._ues:
            ue.update_position() #用户移动


        for uav in self._uavs:
            uav.reset_for_next_step()  #重置临时状态
        #阶段7：执行无人机动作（移动）
        self._apply_actions_to_env(actions) #神经网络给出的动作

        # ========== 新增：更新动态障碍物位置 ==========
        for obs in self._dynamic_obstacles:
            obs.update()

        # ========== 新增：障碍物碰撞检测 ==========
        self._check_obstacle_collisions()


        #阶段8：获取下一步的观测并返回
        next_obs: list[np.ndarray] = self._get_obs()
        return next_obs, rewards, metrics

    def _get_obs(self) -> list[np.ndarray]: #获取每个无人机的观测，二维
        """Gets the local observation for each UAV agent.
           返回5个265维的观测向量，每个无人机一个
           ue.generate_request()：每个用户可能生成三种请求：服务请求(type=0)、内容请求(type=1)、能量请求(type=2)
           _associate_ues_to_uavs()： 每个用户只能被一个无人机服务，结果：每个无人机的 current_covered_ues 列表被填充
           set_neighbors()：无人机找邻居
        """
        # For new time step
        for ue in self._ues:
            ue.generate_request() #每个用户随机生成任务请求
        self._associate_ues_to_uavs() #根据覆盖范围，将用户分配给最近的无人机
        for uav in self._uavs:
            uav.set_neighbors(self._uavs) #找出在感知范围内的其他无人机

        all_obs: list[np.ndarray] = []
        for uav in self._uavs:
            # Part 1: Own state (position and cache status)
            own_pos: np.ndarray = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT]) #位置归一化：把700×700区域映射到[0,1]区间  例如位置(350,350) → (0.5, 0.5)
            own_cache: np.ndarray = uav.cache.astype(np.float32) # 缓存状态：75个bool值（True/False），表示是否缓存了每个文件
            own_state: np.ndarray = np.concatenate([own_pos, own_cache]) # 拼接后：2 + 75 = 77维

            # Part 2: Neighbor positions
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, config.NEIGHBOR_OBS_DIM)) #初始化4行*2列的 零矩阵
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[: config.MAX_UAV_NEIGHBORS] # 找出最近的4个邻居（按距离排序）
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE
                # 相对位置归一化：除以感知半径300米，  范围在[-1, 1]之间
                neighbor_states[i, :] = relative_pos # 第i行存这个邻居的相对位置(x,y)

            # Part 3: State of associated UEs 关联用户状态
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, config.UE_OBS_DIM)) # 初始化：30行6列的零矩阵
            # 找出最近的30个用户
            ues: list[UE] = sorted(uav.current_covered_ues, key=lambda u: float(np.linalg.norm(uav.pos[:2] - u.pos[:2])))[: config.MAX_ASSOCIATED_UES]
            for i, ue in enumerate(ues):
                # 相对位置（2维）
                delta_pos: np.ndarray = (ue.pos[:2] - uav.pos[:2]) / config.AREA_WIDTH # 除以区域宽度700米，归一化到[-1,1]
                req_type, req_size, req_id = ue.current_request  # 用户请求信息（4维）
                norm_type: float = float(req_type) / 2.0  # assuming 3 types: 0,1,2
                norm_id: float = float(req_id) / float(config.NUM_FILES)
                norm_size: float = float(req_size) / float(config.MAX_INPUT_SIZE) # 归一化文件大小
                norm_battery: float = ue.battery_level / config.UE_BATTERY_CAPACITY # 电量比例0-1
                request_info: np.ndarray = np.array([norm_type, norm_size, norm_id, norm_battery], dtype=np.float32) # 拼接相对位置(2维) + 请求信息(4维) = 6维
                ue_states[i, :] = np.concatenate([delta_pos, request_info])

            # Part 4: Combine all parts into a single, flat observation vector
            obs: np.ndarray = np.concatenate([own_state, neighbor_states.flatten(), ue_states.flatten()])
            all_obs.append(obs)

        return all_obs

    #执行动作，移动无人机 ---需要考虑无人机的碰撞
    def _apply_actions_to_env(self, actions: np.ndarray) -> None:
        """Calculates next positions and resolves potential collisions iteratively.
           actions: (5, 2) 的数组，5个无人机，每个[角度比例, 距离比例]
           ---将神经网络输出的动作转换成实际的无人机移动
        """
        current_positions: np.ndarray = np.array([uav.pos[:2] for uav in self._uavs])
        max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION

        # Interpret actions as a direct (x, y) vector-将各种动作视为一个x，y来表示
        delta_vec_raw: np.ndarray = np.array(actions, dtype=np.float32)

        # Calculate the magnitude (distance) of this raw vector---raw_magnitude 原始向量长度
        # 动作 [0.5, 0.8] 的 raw_magnitude = √(0.5² + 0.8²) = 0.94
        # 这个值代表“想飞多远”的比例
        raw_magnitude: np.ndarray = np.linalg.norm(delta_vec_raw, axis=1, keepdims=True)

        # Clip the magnitude to be at most 1.0 --剪裁后的长度
        clipped_magnitude: np.ndarray = np.minimum(raw_magnitude, 1.0)
        distances: np.ndarray = clipped_magnitude * max_dist #实际移动距离
        # 计算单位方向向量（避免除0）
        denom: np.ndarray = raw_magnitude + float(config.EPSILON)
        directions: np.ndarray = delta_vec_raw / denom  # 归一化得到方向
        delta_pos: np.ndarray = directions * distances  # 方向 × 距离
        #计算新位置
        proposed_positions: np.ndarray = current_positions + delta_pos # 新位置 = 当前位置 + 移动向量

        #边界检查
        min_boundary_gap: float = config.UAV_COVERAGE_RADIUS / 2.0
        for i, uav in enumerate(self._uavs): #enumerate优雅的获取索引和对应的值---无需自己找索引
            # 检查是否要飞出边界（考虑无人机半径，不能贴边）
            if not (min_boundary_gap <= proposed_positions[i, 0] <= config.AREA_WIDTH - min_boundary_gap and min_boundary_gap <= proposed_positions[i, 1] <= config.AREA_HEIGHT - min_boundary_gap):
                uav.boundary_violation = True #标记越界违规
        # 强行拉回边界内
        # 位置被限制在 [50, 650] 范围内
        next_positions: np.ndarray = np.clip(proposed_positions, [min_boundary_gap, min_boundary_gap], [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap])

        #避免碰撞的核心算法
        min_sep_sq: float = config.MIN_UAV_SEPARATION**2 # 200² = 40000
        for _ in range(config.COLLISION_AVOIDANCE_ITERATIONS + 1):
            collision_detected_in_iter: bool = False #现在的碰撞距离为false--理解为哨兵
            for i in range(config.NUM_UAVS):
                for j in range(i + 1, config.NUM_UAVS):
                    pos_i: np.ndarray = next_positions[i]
                    pos_j: np.ndarray = next_positions[j]
                    dist_sq: float = np.sum((pos_i - pos_j) ** 2) #计算距离平方
                    if dist_sq < min_sep_sq:
                        # 标记违规
                        self._uavs[i].collision_violation = True
                        self._uavs[j].collision_violation = True
                        collision_detected_in_iter = True
                        # 计算推开的方向和距离
                        # 如果距离平方大于0，就开方得到实际距离；否则就用一个极小值代替。
                        dist: float = np.sqrt(dist_sq) if dist_sq > 0 else config.EPSILON
                        overlap: float = config.MIN_UAV_SEPARATION - dist # 重叠了多少
                        direction: np.ndarray = (pos_i - pos_j) / dist # 推开方向
                        # 两个无人机个推开一半
                        next_positions[i] += direction * overlap * 0.5
                        next_positions[j] -= direction * overlap * 0.5
            #如果本轮未检测到碰撞--提前结束
            if not collision_detected_in_iter:
                break
        # 再次确保在边界内（推开后可能又出界）
        final_positions: np.ndarray = np.clip(next_positions, [min_boundary_gap, min_boundary_gap], [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap])
        # 更新无人机位置
        for i, uav in enumerate(self._uavs):
            uav.update_position(final_positions[i])
    #将用户分配给无人机
    def _associate_ues_to_uavs(self) -> None:
        """Assigns each UE to at most one UAV, resolving overlaps by choosing the closest UAV.
           # 无返回值，直接修改 ue.assigned 和 uav.current_covered_ues
        """
        for ue in self._ues: # 遍历所有ue 100个
            covering_uavs: list[tuple[UAV, float]] = [] # 存储能覆盖这个用户的无人机及距离
            for uav in self._uavs:
                distance: float = float(np.linalg.norm(uav.pos[:2] - ue.pos[:2])) # 计算uav和ue之间的平面距离，忽略高度
                if distance <= config.UAV_COVERAGE_RADIUS: #如果在uav覆盖的钣金范围内
                    covering_uavs.append((uav, distance)) #存 uav和对应距离

            if not covering_uavs:
                continue #无果没有uav覆盖这个ue，则跳过这个用户（没人服务他）
            best_uav, _ = min(covering_uavs, key=lambda x: x[1]) # 按距离排序，取最小的
            best_uav.current_covered_ues.append(ue)  # 把这个用户加入无人机服务列表
            ue.assigned = True

    #计算奖励和指标
    def _get_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float, float]]:
        """Returns the reward and other metrics. 返回: (每个无人机的奖励列表, (总延迟, 总能耗, 公平性, 离线率))
           这是整个强化学习系统的**心脏**——它定义了无人机**为什么学习**以及**学什么**！
           ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues
           # 语法：值1 if 条件 else 值2

          if ue.assigned:  # 如果用户被服务
                return ue.latency_current_request  # 返回他的实际延迟
          else:  # 如果用户没被服务
                return config.NON_SERVED_LATENCY_PENALTY  # 返回惩罚值（20.0）
        """
        # 计算总延迟
        total_latency: float = sum(ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues)
        # 所有无人机的能耗之和（飞行 + 悬停 + 计算 + WPT）
        total_energy: float = sum(uav.energy for uav in self._uavs)
        # 每个用户的服务覆盖率（被成功服务的次数/总时间步）
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues])
        # Jain's Fairness Index 公式： (Σx)² / (n × Σx²)
        # 范围 [0, 1]，越接近1越公平
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))
        # 电量低于30J的用户数量
        offline_count: int = sum(1 for ue in self._ues if ue.battery_level < config.UE_CRITICAL_THRESHOLD)
        # 离线用户比例
        offline_rate: float = offline_count / config.NUM_UES
        # 计算奖励  奖励公式：reward = α₃×log(公平性) - α₁×log(延迟) - α₂×log(能耗) - α₄×log(1+离线率)

        # ========== 新增：计算避障惩罚 ==========
        # 统计本步有多少无人机碰撞了障碍物
        obstacle_penalty = 0.0
        for uav in self._uavs:
            if uav.collision_violation:
                # 如果无人机碰撞了，添加惩罚
                obstacle_penalty += config.OBSTACLE_COLLISION_PENALTY

        r_fairness: float = config.ALPHA_3 * np.log(jfi + config.EPSILON) # 正项：鼓励公平
        r_latency: float = config.ALPHA_1 * np.log(total_latency + config.EPSILON) # 负项：惩罚延迟
        r_energy: float = config.ALPHA_2 * np.log(total_energy + config.EPSILON) # 负项：惩罚能耗
        r_offline: float = config.ALPHA_4 * np.log(1.0 + offline_rate)  # 负项：惩罚离线率


        # ========== 新奖励公式（加入避障惩罚）==========
        reward = r_fairness - r_latency - r_energy - r_offline - obstacle_penalty
        # reward: float = r_fairness - r_latency - r_energy - r_offline  # 最终奖励 = 公平性奖励 - 各项惩罚
        rewards: list[float] = [reward] * config.NUM_UAVS  # 所有无人机得到相同基础奖励

        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY # 碰撞惩罚 10.0
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY # 越界惩罚 10.0
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards] # 乘以0.01防止值过大 对奖励进行缩放
        return rewards, (total_latency, total_energy, jfi, offline_rate)

