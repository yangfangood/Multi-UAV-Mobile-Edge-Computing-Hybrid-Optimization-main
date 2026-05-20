from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np


class Env:
    def __init__(self) -> None: #构造函数：初始化环境
        self._mbs_pos: np.ndarray = config.MBS_POS #基站位置
        UE.initialize_ue_class() #初始化用户类
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)] #创建100个用户---UE(i)创建了一个实例
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)] #创建5个无人机
        self._time_step: int = 0 #时间步计数器 初始为0，表示仿真刚开始
        #每个step()被调用一次，_time_step就加1


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
        return self._get_obs() #返回 初始观测

    #动作执行，返回新状态和奖励！核心函数
    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], list[float], tuple[float, float, float, float]]:#actions是动作，不是随机生成的，来自神经网络
        """Execute one time step of the simulation."""
        self._time_step += 1
        # 信道模型G2A 测试
        if self._time_step == 1:
            from environment import comm_model as comms
            sample_gain = comms.calculate_channel_gain(self._uavs[0].pos, self._ues[0].pos)
            print(f"Sample channel gain: {sample_gain}")
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
        #阶段8：获取下一步的观测并返回
        next_obs: list[np.ndarray] = self._get_obs()
        return next_obs, rewards, metrics

    def _get_obs(self) -> list[np.ndarray]:
        #print("!!! Entering _get_obs !!!")
        # For new time step
        for ue in self._ues:
            ue.generate_request()
        self._associate_ues_to_uavs()
        for uav in self._uavs:
            uav.set_neighbors(self._uavs)

        all_obs: list[np.ndarray] = []
        for uav in self._uavs:
            # Part 1: Own state (position, cache status, and UAV type)
            own_pos: np.ndarray = (uav.pos[:2] / [config.AREA_WIDTH, config.AREA_HEIGHT]).astype(np.float32)
            own_cache: np.ndarray = uav.cache.astype(np.float32)

            # ========== 新增：无人机类型 one-hot 编码 (3维) ==========
            type_onehot = np.zeros(3, dtype=np.float32)
            type_onehot[uav.type] = 1.0
            # =======================新增能力数值===============================
            # 假设最大计算能力为 config.UAV_COMPUTING_CAPACITY 中的最大值（或已知常量）
            norm_compute = np.clip(uav.computing_capacity / config.MAX_UAV_COMPUTING, 0.0, 1.0)
            norm_storage = np.clip(uav.storage_capacity / config.MAX_UAV_STORAGE, 0.0, 1.0)
            norm_load = np.clip(uav._current_service_request_count / config.MAX_LOAD, 0.0, 1.0)
            # 原代码：own_state = np.concatenate([own_pos, own_cache])   # 77维
            # 修改后：加入 type_onehot，变为 80 维
            # own_state: np.ndarray = np.concatenate([own_pos, own_cache, type_onehot,[norm_compute, norm_storage],[norm_load] ])  # 2+75+3 + 2 +1=83  增加 2 维
            capability = np.array([norm_compute, norm_storage], dtype=np.float32)
            load = np.array([norm_load], dtype=np.float32)
            own_state = np.concatenate([
                own_pos,
                own_cache,
                type_onehot,
                capability,
                load
            ])
            # Part 2: Neighbor positions（不变）
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, config.NEIGHBOR_OBS_DIM))
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[
                                   : config.MAX_UAV_NEIGHBORS]
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE
                neighbor_states[i, :] = relative_pos

            # Part 3: State of associated UEs（不变）
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, config.UE_OBS_DIM))
            ues: list[UE] = sorted(uav.current_covered_ues,
                                   key=lambda u: float(np.linalg.norm(uav.pos[:2] - u.pos[:2])))[
                            : config.MAX_ASSOCIATED_UES]
            for i, ue in enumerate(ues):
                delta_pos: np.ndarray = (ue.pos[:2] - uav.pos[:2]) / config.AREA_WIDTH
                req_type, req_size, req_id = ue.current_request
                norm_type: float = float(req_type) / 2.0
                norm_id: float = float(req_id) / float(config.NUM_FILES)
                norm_size: float = float(req_size) / float(config.MAX_INPUT_SIZE)
                norm_battery: float = ue.battery_level / config.UE_BATTERY_CAPACITY
                request_info: np.ndarray = np.array([norm_type, norm_size, norm_id, norm_battery], dtype=np.float32)
                ue_states[i, :] = np.concatenate([delta_pos, request_info])

            # Part 4: Combine all parts
            obs: np.ndarray = np.concatenate([own_state, neighbor_states.flatten(), ue_states.flatten()])
            all_obs.append(obs)

        # 可选：打印第一个观测的维度以验证
        #print(f"DEBUG env._get_obs: obs shape = {all_obs[0].shape}")
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
        # print("USING MODIFIED REWARD FUNCTION (LINEAR NORMALIZED)")
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
        # ========== 归一化处理 ==========


        norm_latency = total_latency / config.MAX_LATENCY
        norm_energy = total_energy / config.MAX_ENERGY

        # 可选：clip 防止极端值
        norm_latency = np.clip(norm_latency, 0.0, 1.0)
        norm_energy = np.clip(norm_energy, 0.0, 1.0)

        # ========== 奖励权重 ==========
        w_fair = config.ALPHA_3  # 公平性增益
        w_lat = config.ALPHA_1  # 延迟惩罚
        w_eng = config.ALPHA_2  # 能耗惩罚
        w_off = config.ALPHA_4  # 离线率惩罚（保持高权重，鼓励充电）

        # 奖励公式（所有项均为线性，范围相近）
        reward = (w_fair * jfi
                  - w_lat * norm_latency
                  - w_eng * norm_energy
                  - w_off * offline_rate)
        #确认数值量级
        if self._time_step % 100 == 0:  # 每100步打印一次
            print(f"[DEBUG] total_latency={total_latency:.2e}, norm_lat={norm_latency:.3f}")
            print(f"[DEBUG] total_energy={total_energy:.2e}, norm_eng={norm_energy:.3f}")
            print(f"[DEBUG] jfi={jfi:.3f}, offline_rate={offline_rate:.3f}")
            print(f"[DEBUG] raw reward (before scaling) = {reward:.4f}")
            print(f"[DEBUG] final reward (after scaling) = {reward * config.REWARD_SCALING_FACTOR:.4f}")

        # 每个无人机获得相同的基础奖励
        rewards = [reward] * config.NUM_UAVS

        # 添加碰撞/越界惩罚（保持原样）
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY

        # 最终缩放（可选，保留原缩放因子）
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]

        return rewards, (total_latency, total_energy, jfi, offline_rate)
"""
r_fairness: float = config.ALPHA_3 * np.log(jfi + config.EPSILON) # 正项：鼓励公平
        r_latency: float = config.ALPHA_1 * np.log(total_latency + config.EPSILON) # 负项：惩罚延迟
        r_energy: float = config.ALPHA_2 * np.log(total_energy + config.EPSILON) # 负项：惩罚能耗
        r_offline: float = config.ALPHA_4 * np.log(1.0 + offline_rate)  # 负项：惩罚离线率
        reward: float = r_fairness - r_latency - r_energy - r_offline  # 最终奖励 = 公平性奖励 - 各项惩罚
        rewards: list[float] = [reward] * config.NUM_UAVS # 所有无人机得到相同基础奖励

        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY # 碰撞惩罚 10.0
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY # 越界惩罚 10.0
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards] # 乘以0.01防止值过大 对奖励进行缩放
        return rewards, (total_latency, total_energy, jfi, offline_rate)  
"""



