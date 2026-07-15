from environment.user_equipments import UE
from environment.uavs import UAV, RequestPlan
from environment.uavs import RouteEstimate, estimate_route  # v1.5
import config
import numpy as np
import time


# ============================================================
# v1.5 模块级辅助函数
# ============================================================

def _offload_cost(est: RouteEstimate) -> float:
    """代价函数由 OFFLOAD_PIPELINE_MODE 唯一决定。"""
    latency = est.total_latency
    energy = est.controllable_energy

    if not np.isfinite(latency) or latency < 0.0:
        raise ValueError(f"Invalid latency in estimate: {latency}")
    if not np.isfinite(energy) or energy < 0.0:
        raise ValueError(f"Invalid energy in estimate: {energy}")

    mode = config.OFFLOAD_PIPELINE_MODE

    if mode in {"unified_fixed_targets", "iterative_latency"}:
        return latency

    if mode == "iterative_joint":
        norm_lat = latency / config.OFFLOAD_LATENCY_REF
        norm_eng = energy / config.OFFLOAD_ENERGY_REF
        return (config.OFFLOAD_LATENCY_WEIGHT * norm_lat
                + config.OFFLOAD_ENERGY_WEIGHT * norm_eng)

    if mode == "legacy":
        raise RuntimeError("Legacy mode must not use unified offload cost.")

    raise ValueError(f"Unknown pipeline mode: {mode}")


def _is_better_cost(new_cost: float, old_cost: float) -> bool:
    return new_cost < old_cost - config.OFFLOAD_COST_TOLERANCE


class Env:
    def __init__(self) -> None: #构造函数：初始化环境
        self._mbs_pos: np.ndarray = config.MBS_POS #基站位置
        UE.initialize_ue_class() #初始化用户类
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)] #创建100个用户---UE(i)创建了一个实例
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)] #创建5个无人机
        self._initialize_safe_uav_positions()  # 无碰撞初始位置
        self._time_step: int = 0 #时间步计数器 初始为0，表示仿真刚开始
        #每个step()被调用一次，_time_step就加1
        self.cumulative_served = [0] * config.NUM_UAVS
        self.cumulative_energy = [0.0] * config.NUM_UAVS
        self.cumulative_latency = [0.0] * config.NUM_UAVS
        self._diag_file: str | None = None  # 诊断日志文件路径
        self._request_generation_round: int = 0  # 验证：请求生成轮数
        self._last_system_energy: float = 0.0

        # === 卸载规划诊断（v1.5） ===
        self._last_offload_iterations: int = 0
        self._last_offload_converged: bool = True
        self._last_offload_planning_time: float = 0.0

        self._offload_planning_times: list[float] = []
        self._offload_iteration_history: list[int] = []
        self._offload_convergence_history: list[bool] = []

        self._j5_verified_service_steps: int = 0
        self._mbs_service_ratio_history: list[float] = []

        self._route_latencies: list[float] = []
        self._route_energies: list[float] = []

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    @property
    def last_system_energy(self) -> float:
        return self._last_system_energy

    def _initialize_safe_uav_positions(self) -> None:
        """整组重试生成无碰撞 UAV 初始位置"""
        min_gap: float = config.UAV_COVERAGE_RADIUS / 2.0
        min_sep: float = config.MIN_UAV_SEPARATION
        x_min: float = min_gap
        x_max: float = config.AREA_WIDTH - min_gap
        y_min: float = min_gap
        y_max: float = config.AREA_HEIGHT - min_gap

        for _ in range(100):
            layout: list[np.ndarray] = []
            for _uav_id in range(config.NUM_UAVS):
                placed: bool = False
                for _ in range(1000):
                    candidate = np.array(
                        [np.random.uniform(x_min, x_max),
                         np.random.uniform(y_min, y_max)],
                        dtype=np.float64,
                    )
                    if all(np.linalg.norm(candidate - p) >= min_sep for p in layout):
                        layout.append(candidate)
                        placed = True
                        break
                if not placed:
                    break
            if len(layout) == config.NUM_UAVS:
                for uav, pos in zip(self._uavs, layout):
                    uav.pos[:2] = pos
                if config.VERIFY_TIMING:
                    for i in range(config.NUM_UAVS):
                        for j in range(i + 1, config.NUM_UAVS):
                            d = float(np.linalg.norm(self._uavs[i].pos[:2] - self._uavs[j].pos[:2]))
                            if d < min_sep - 1e-6:
                                raise AssertionError(
                                    f"Unsafe initial UAV layout: UAV{i}-UAV{j}, distance={d:.6f}"
                                )
                return
        raise RuntimeError(
            "Unable to generate collision-free UAV layout after 100 full-group attempts."
        )

    def set_diag_log(self, filepath: str) -> None:
        """设置诊断日志文件路径"""
        self._diag_file = filepath

    def _diag_print(self, *args, **kwargs) -> None:
        """写到诊断日志文件，未设置则输出到终端"""
        if self._diag_file:
            with open(self._diag_file, "a", encoding="utf-8") as f:
                print(*args, file=f, **kwargs)
        else:
            print(*args, **kwargs)

    def reset(self) -> list[np.ndarray]: #重置环境，开始新的回合，相当于游戏重新开始
        """Resets the environment to an initial state and returns the initial observations."""
        self._ues = [UE(i) for i in range(config.NUM_UES)]
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]
        self._initialize_safe_uav_positions()
        self._time_step = 0
        self._request_generation_round = 0
        # ========== 一次性绘图：检查用户分布 ==========
        if not hasattr(self, '_plotted'):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))

            # 用户位置
            ue_x = [ue.pos[0] for ue in self._ues]
            ue_y = [ue.pos[1] for ue in self._ues]
            ax.scatter(ue_x, ue_y, c='blue', s=10, alpha=0.5, label='UEs')

            # 无人机初始位置
            uav_x = [uav.pos[0] for uav in self._uavs]
            uav_y = [uav.pos[1] for uav in self._uavs]
            ax.scatter(uav_x, uav_y, c='red', s=100, marker='^', label='UAVs')

            # 覆盖圆
            for uav in self._uavs:
                circle = plt.Circle((uav.pos[0], uav.pos[1]), config.UAV_COVERAGE_RADIUS,
                                    color='red', fill=False, alpha=0.3)
                ax.add_patch(circle)

            ax.set_xlim(0, config.AREA_WIDTH)
            ax.set_ylim(0, config.AREA_HEIGHT)
            ax.set_aspect('equal')
            ax.legend()
            ax.set_title(f'Initial Distribution: {config.NUM_UES} UEs, {config.NUM_UAVS} UAVs')
            plt.savefig('initial_distribution.png', dpi=150)
            plt.close()
            self._plotted = True
            print("[PLOT] Saved initial_distribution.png")
        # ============================================
        # 为初始状态 s_0 生成请求并建立关联
        self._generate_requests()
        self._refresh_relations()

        if config.VERIFY_TIMING and self._request_generation_round != 1:
            raise AssertionError(
                f"[VERIFY] reset: request gen round={self._request_generation_round}, expected=1"
            )

        return self._build_observations()


    #动作执行，返回新状态和奖励！核心函数  s_t + a_t → s_{t+1}, r_t
    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], list[float], tuple[float, float, float, float]]:
        """Execute one time step of the simulation.
        新时序：先执行动作→重新关联→处理 Actor 看到的请求→计算奖励→生成下一请求→构建观测
        """
        self._time_step += 1

        # ====== 一次性配置检查 ======
        if self._time_step == 1:
            for uav in self._uavs:
                self._diag_print(f"[CONFIG_CHECK] UAV {uav.id}: computing_capacity = {uav.computing_capacity:.2e} Hz")
            self._diag_print(f"[CONFIG_CHECK] UAV_COMPUTING_CAPACITY (config) = {config.UAV_COMPUTING_CAPACITY}")
            from environment import comm_model as comms
            sample_gain = comms.calculate_channel_gain(self._uavs[0].pos, self._ues[0].pos)
            self._diag_print(f"Sample channel gain: {sample_gain}")

        # === 验证 A：保存动作前的请求快照 ===
        if config.VERIFY_TIMING:
            requests_before = [ue.current_request for ue in self._ues]

        # 1. 执行 a_t：UAV 移动  s_t + a_t → 新位置
        self._apply_actions_to_env(actions)

        # 2. 移动后重新关联（处理 s_t 中已有的请求 q_t，不重新生成）
        self._refresh_relations()

        # === 验证 A：请求在动作前后必须一致 ===
        if config.VERIFY_TIMING:
            requests_after = [ue.current_request for ue in self._ues]
            if requests_before != requests_after:
                self._diag_print("[VERIFY_ERROR] Requests changed between action and processing")
                for i, (b, a) in enumerate(zip(requests_before, requests_after)):
                    if b != a:
                        self._diag_print(f"  UE{i}: before={b} after={a}")
                raise AssertionError("VERIFY: requests changed between action selection and processing")

        # 3. 统一初始化临时缓存和回传速率
        for uav in self._uavs:
            uav.prepare_request_slot()

        # 4. 管道模式调度（v1.5 四模式）
        mode = config.OFFLOAD_PIPELINE_MODE

        if mode == "legacy":
            all_plans = self._build_legacy_plans()

            # 验证E：计划完整性
            if config.VERIFY_TIMING:
                associated_ids = sorted(
                    ue.id
                    for uav in self._uavs
                    for ue in uav.current_covered_ues
                )
                planned_ids = sorted(plan.ue.id for plan in all_plans)
                if planned_ids != associated_ids:
                    raise AssertionError(
                        f"Request plan coverage mismatch: "
                        f"associated={associated_ids}, planned={planned_ids}"
                    )

            self._count_execution_loads(all_plans)
            for plan in all_plans:
                plan.source_uav.execute_legacy_request_plan(plan)

        elif mode == "unified_fixed_targets":
            all_plans = self._build_legacy_plans()
            self._count_execution_loads(all_plans)
            all_plans = self._freeze_plan_estimates(all_plans)
            for plan in all_plans:
                plan.source_uav.execute_request_plan(plan)

        elif mode in {"iterative_latency", "iterative_joint"}:
            all_plans = self._plan_all_requests()
            self._count_execution_loads(all_plans)
            for plan in all_plans:
                plan.source_uav.execute_request_plan(plan)

        else:
            raise ValueError(f"Unknown OFFLOAD_PIPELINE_MODE: {mode}")

        # 验证G：缓存频率守恒
        if config.VERIFY_TIMING:
            expected_freq = sum(
                1 for p in all_plans if p.request_type in (0, 1)
            )
            expected_freq += sum(
                1 for p in all_plans
                if p.request_type in (0, 1) and p.target_idx == 1
            )
            actual_freq = int(
                sum(np.sum(uav._freq_counts) for uav in self._uavs)
            )
            if actual_freq != expected_freq:
                raise AssertionError(
                    f"Frequency count mismatch: "
                    f"actual={actual_freq}, expected={expected_freq}"
                )

        # 5. 更新 UE 状态
        for ue in self._ues:
            if not ue.assigned:
                ue.update_battery(0.0, 0.0)
            ue.update_service_coverage(self._time_step)

        # 6. 更新 UAV 缓存统计和能耗
        for uav in self._uavs:
            uav.update_ema_and_cache()
            uav.update_energy_consumption()

        # === 验证 C：指令位移 + 实际位移 + 飞行能耗 三重验证 ===
        if config.VERIFY_TIMING:
            max_distance = config.UAV_SPEED * config.TIME_SLOT_DURATION

            for i, uav in enumerate(self._uavs):
                # C1. 指令位移：验证动作缩放正确
                commanded_dist = float(np.linalg.norm(self._commanded_delta[i]))
                if commanded_dist > max_distance + 1e-6:
                    raise AssertionError(
                        f"UAV{i} commanded movement exceeds limit: "
                        f"actual={commanded_dist:.8f}, max={max_distance:.8f}"
                    )

                # C2. 实际位移：验证碰撞拒绝/边界裁剪后不超速
                actual_dist = float(uav._dist_moved)
                if actual_dist > max_distance + 1e-6:
                    raise AssertionError(
                        f"UAV{i} actual movement exceeds limit: "
                        f"actual={actual_dist:.8f}, max={max_distance:.8f}"
                    )

                # C3. 飞行能耗 = 移动能耗 + 悬停能耗
                move_time = actual_dist / config.UAV_SPEED
                hover_time = config.TIME_SLOT_DURATION - move_time
                if hover_time < -1e-6:
                    raise AssertionError(
                        f"UAV{i} has negative hovering time: {hover_time:.8f}"
                    )
                hover_time = max(0.0, hover_time)
                expected_fly_energy = (
                    config.POWER_MOVE * move_time
                    + config.POWER_HOVER * hover_time
                )
                if not np.isclose(
                    uav._fly_energy_slot,
                    expected_fly_energy,
                    rtol=1e-6,
                    atol=1e-6,
                ):
                    raise AssertionError(
                        f"UAV{i} flight energy mismatch: "
                        f"actual={uav._fly_energy_slot:.8f}, "
                        f"expected={expected_fly_energy:.8f}, "
                        f"distance={actual_dist:.8f}"
                    )

        # 7. 计算奖励 r_t（当前奖励由 s_t、a_t 及动作执行后的服务结果共同决定）
        rewards, metrics = self._get_rewards_and_metrics()

        # 8. 周期性缓存更新
        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        # 9. UE 移动到下一时隙位置
        for ue in self._ues:
            ue.update_position()
            ue.reset_for_next_step()

        # 10. 清理当前时隙临时状态
        for uav in self._uavs:
            uav.reset_for_next_step()

        # 11. 生成下一时隙请求 q_{t+1}
        self._generate_requests()

        # === 验证 B：请求生成次数 = time_step + 1 ===
        if config.VERIFY_TIMING:
            if self._request_generation_round != self._time_step + 1:
                self._diag_print(f"[VERIFY_ERROR] request gen round={self._request_generation_round} expected={self._time_step + 1}")
                raise AssertionError(f"Request generation count mismatch")

        # 12. 为下一观测建立关联和邻居
        self._refresh_relations()

        # 13. 构建 s_{t+1}
        next_obs: list[np.ndarray] = self._build_observations()

        # === 验证 D：观测完整有效 ===
        if config.VERIFY_TIMING:
            if len(next_obs) != config.NUM_UAVS:
                raise AssertionError(f"[VERIFY] next_obs length={len(next_obs)}, expected={config.NUM_UAVS}")
            load_idx = 2 + config.NUM_FILES + config.UAV_TYPE_OBS_DIM + config.UAV_CAPABILITY_OBS_DIM
            ue_block_start = config.SELF_OBS_DIM + config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM

            for i, obs in enumerate(next_obs):
                if obs.shape != (config.OBS_DIM_SINGLE,):
                    self._diag_print(f"[VERIFY_ERROR] UAV{i} obs shape={obs.shape} expected=({config.OBS_DIM_SINGLE},)")
                    raise AssertionError(f"UAV{i} observation shape error")
                if not np.all(np.isfinite(obs)):
                    self._diag_print(f"[VERIFY_ERROR] UAV{i} obs contains NaN or Inf")
                    raise AssertionError(f"UAV{i} observation contains NaN/Inf")

                # 负载一致
                expected_load = sum(1 for ue in self._uavs[i].current_covered_ues if ue.current_request[0] == 0)
                expected_norm = np.clip(expected_load / config.MAX_LOAD, 0.0, 1.0)
                actual_norm = obs[load_idx]
                if not np.isclose(actual_norm, expected_norm, atol=1e-6):
                    self._diag_print(f"[VERIFY_ERROR] UAV{i} load mismatch: obs={actual_norm:.4f} expected={expected_norm:.4f}")
                    raise AssertionError(f"UAV{i} load observation mismatch")

                # 用户块非空
                if len(self._uavs[i].current_covered_ues) > 0:
                    if not np.any(np.abs(obs[ue_block_start:]) > 1e-8):
                        self._diag_print(f"[VERIFY_ERROR] UAV{i} has UEs but obs block all zero")
                        raise AssertionError(f"UAV{i} UE observation block is all zero")

        return next_obs, rewards, metrics

    def _generate_requests(self) -> None:
        """每个新时隙只生成一次请求"""
        for ue in self._ues:
            ue.generate_request()
        self._request_generation_round += 1

    def _refresh_relations(self) -> None:
        """根据当前 UE 和 UAV 位置重新计算关联与邻居，不生成请求"""
        self._associate_ues_to_uavs()
        for uav in self._uavs:
            uav.set_neighbors(self._uavs)
            uav.update_associated_service_load()

    def _build_observations(self) -> list[np.ndarray]:
        """纯读取当前状态构建观测数组，无任何副作用"""
        all_obs: list[np.ndarray] = []
        for uav in self._uavs:
            own_pos: np.ndarray = (uav.pos[:2] / [config.AREA_WIDTH, config.AREA_HEIGHT]).astype(np.float32)
            own_cache: np.ndarray = uav.cache.astype(np.float32)

            obs_parts = [own_pos, own_cache]

            if config.UAV_TYPE_OBS_DIM > 0:
                type_onehot = np.zeros(config.UAV_TYPE_OBS_DIM, dtype=np.float32)
                type_onehot[uav.type] = 1.0
                obs_parts.append(type_onehot)

            if config.UAV_CAPABILITY_OBS_DIM > 0:
                norm_compute = np.clip(uav.computing_capacity / config.MAX_UAV_COMPUTING, 0.0, 1.0)
                norm_storage = np.clip(uav.storage_capacity / config.MAX_UAV_STORAGE, 0.0, 1.0)
                capability = np.array([norm_compute, norm_storage], dtype=np.float32)
                obs_parts.append(capability)

            if config.UAV_LOAD_OBS_DIM > 0:
                associated_load = uav.associated_service_load
                norm_load = np.clip(associated_load / config.MAX_LOAD, 0.0, 1.0)
                load = np.array([norm_load], dtype=np.float32)
                obs_parts.append(load)

            own_state = np.concatenate(obs_parts)

            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, config.NEIGHBOR_OBS_DIM))
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[
                                   : config.MAX_UAV_NEIGHBORS]
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE
                neighbor_states[i, :] = relative_pos

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

            obs: np.ndarray = np.concatenate([own_state, neighbor_states.flatten(), ue_states.flatten()])
            all_obs.append(obs)

        return all_obs

    def _get_obs(self) -> list[np.ndarray]:
        """兼容旧接口：仅读取状态，不生成请求、不更新关联。"""
        return self._build_observations()

    #执行动作，移动无人机 ---需要考虑无人机的碰撞
    def _apply_actions_to_env(self, actions: np.ndarray) -> None:
        """执行 UAV 移动：输入验证 → 模长裁剪 → 边界裁剪 → 迭代碰撞拒绝 → 更新位置"""
        # ---- 0. 输入验证 ----
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (config.NUM_UAVS, config.ACTION_DIM):
            raise ValueError(f"Invalid action shape: received={actions.shape}, "
                             f"expected=({config.NUM_UAVS}, {config.ACTION_DIM})")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Actions contain NaN or Inf.")

        current_positions: np.ndarray = np.array([uav.pos[:2] for uav in self._uavs])
        max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION

        # ---- 1. 模长裁剪：保证 ||commanded_delta|| ≤ max_dist ----
        raw_delta: np.ndarray = actions * max_dist
        raw_norms: np.ndarray = np.linalg.norm(raw_delta, axis=1, keepdims=True)
        scales: np.ndarray = np.minimum(1.0, max_dist / (raw_norms + float(config.EPSILON)))
        commanded_delta: np.ndarray = raw_delta * scales
        self._commanded_delta: np.ndarray = commanded_delta.copy()

        # ---- 2. 诊断打印（每 500 步） ----
        if self._time_step % 500 == 0:
            self._diag_print(f"\n[Step {self._time_step}] Action -> Movement:")
            for i in range(config.NUM_UAVS):
                move_dist = np.linalg.norm(commanded_delta[i])
                self._diag_print(f"  UAV{i}: action=({actions[i, 0]:.4f}, {actions[i, 1]:.4f}), "
                                 f"move_dist={move_dist:.2f}m, max_dist={max_dist:.2f}m")

        # ---- 3. 边界检查 + 裁剪 ----
        min_boundary_gap: float = config.UAV_COVERAGE_RADIUS / 2.0
        proposed_positions: np.ndarray = current_positions + commanded_delta
        for i, uav in enumerate(self._uavs):
            if not (min_boundary_gap <= proposed_positions[i, 0] <= config.AREA_WIDTH - min_boundary_gap and
                    min_boundary_gap <= proposed_positions[i, 1] <= config.AREA_HEIGHT - min_boundary_gap):
                uav.boundary_violation = True
        bounded_positions: np.ndarray = np.clip(proposed_positions,
                                                 [min_boundary_gap, min_boundary_gap],
                                                 [config.AREA_WIDTH - min_boundary_gap,
                                                  config.AREA_HEIGHT - min_boundary_gap])

        # ---- 4. 迭代碰撞拒绝 ----
        min_sep: float = config.MIN_UAV_SEPARATION
        rejected_agents: set[int] = set()

        for _ in range(config.NUM_UAVS + 1):
            final_positions: np.ndarray = bounded_positions.copy()
            for i in rejected_agents:
                final_positions[i] = current_positions[i]

            new_conflicts: set[int] = set()
            for i in range(config.NUM_UAVS):
                for j in range(i + 1, config.NUM_UAVS):
                    if float(np.linalg.norm(final_positions[i] - final_positions[j])) < min_sep - 1e-6:
                        new_conflicts.add(i)
                        new_conflicts.add(j)

            if not new_conflicts:
                break

            prev_count: int = len(rejected_agents)
            rejected_agents.update(new_conflicts)
            if len(rejected_agents) == prev_count:
                raise RuntimeError(
                    "Collision rejection cannot converge. Current UAV positions may already be unsafe."
                )
        else:
            raise RuntimeError("Collision rejection exceeded maximum iterations.")

        for i in rejected_agents:
            self._uavs[i].collision_violation = True

        # ---- 5. 最终间距验证 ----
        if config.VERIFY_TIMING:
            for i in range(config.NUM_UAVS):
                for j in range(i + 1, config.NUM_UAVS):
                    final_sep = float(np.linalg.norm(final_positions[i] - final_positions[j]))
                    if final_sep < min_sep - 1e-6:
                        raise AssertionError(
                            f"Final UAV separation violation: UAV{i}-UAV{j}, "
                            f"distance={final_sep:.6f}, required={min_sep:.6f}"
                        )

        # ---- 6. 更新位置 ----
        for i, uav in enumerate(self._uavs):
            uav.update_position(final_positions[i])
    # 将用户分配给无人机
    def _associate_ues_to_uavs(self) -> None:
        """Assigns each UE to at most one UAV, using nearest association."""
        # ====== 清空上一轮的残留数据 ======
        for uav in self._uavs:
            uav._current_covered_ues = []
        for ue in self._ues:
            ue.assigned = False

        # ====== 关联分配 ======
        for ue in self._ues:
            covering_uavs: list[tuple[UAV, float]] = []
            for uav in self._uavs:
                distance: float = float(np.linalg.norm(uav.pos[:2] - ue.pos[:2]))
                if distance <= config.UAV_COVERAGE_RADIUS:
                    covering_uavs.append((uav, distance))

            if not covering_uavs:
                continue

            best_uav, _ = min(covering_uavs, key=lambda x: x[1])

            if len(best_uav._current_covered_ues) < config.MAX_ASSOCIATED_UES:
                best_uav._current_covered_ues.append(ue)
                ue.assigned = True

        # ====== 诊断统计（每 500 步） ======
        if self._time_step % 500 == 0:
            covered_users = sum(
                1 for ue in self._ues
                if any(np.linalg.norm(uav.pos[:2] - ue.pos[:2]) <= config.UAV_COVERAGE_RADIUS
                       for uav in self._uavs)
            )
            assigned_users = sum(1 for ue in self._ues if ue.assigned)
            self._diag_print(f"[Step {self._time_step}] Coverage: covered={covered_users}/{config.NUM_UES} assigned={assigned_users}/{config.NUM_UES}")
    #计算奖励和指标
    """Returns the reward and other metrics. 返回: (每个无人机的奖励列表, (总延迟, 总能耗, 公平性, 离线率))
               这是整个强化学习系统的**心脏**——它定义了无人机**为什么学习**以及**学什么**！
               ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues
               # 语法：值1 if 条件 else 值2

              if ue.assigned:  # 如果用户被服务
                    return ue.latency_current_request  # 返回他的实际延迟
              else:  # 如果用户没被服务
                    return config.NON_SERVED_LATENCY_PENALTY  # 返回惩罚值（20.0）
            """

    def _count_execution_loads(self, plans: list[RequestPlan]) -> None:
        expected_loads = np.zeros(config.NUM_UAVS, dtype=np.int32)

        for plan in plans:
            if plan.request_type != 0:
                continue
            if plan.target_idx == 0:
                expected_loads[plan.source_uav.id] += 1
            elif plan.target_idx == 1:
                if plan.target_uav is None:
                    raise RuntimeError("Neighbor plan has no target UAV.")
                expected_loads[plan.target_uav.id] += 1
            elif plan.target_idx == 2:
                continue
            else:
                raise ValueError(f"Invalid service target: {plan.target_idx}")

        for uav in self._uavs:
            uav.set_execution_service_load(int(expected_loads[uav.id]))

        if config.VERIFY_TIMING:
            planned_uav_services = sum(
                1 for p in plans
                if p.request_type == 0 and p.target_idx in (0, 1)
            )
            if int(expected_loads.sum()) != planned_uav_services:
                raise AssertionError(
                    f"Execution load total mismatch: "
                    f"loads={expected_loads.tolist()}, planned={planned_uav_services}"
                )
            actual_loads = np.array(
                [uav.execution_service_load for uav in self._uavs], dtype=np.int32
            )
            if not np.array_equal(actual_loads, expected_loads):
                raise AssertionError(
                    f"Execution load per-UAV mismatch: "
                    f"actual={actual_loads.tolist()}, expected={expected_loads.tolist()}"
                )

    # ============================================================
    # v1.5 联合卸载优化：Env 层规划辅助方法（阶段 1A）
    # ============================================================

    def _build_legacy_plans(self) -> list[RequestPlan]:
        """用当前 plan_requests() 生成 Plan 列表，保留原随机顺序。"""
        from dataclasses import replace

        plans: list[RequestPlan] = []
        order_index = 0

        for uav in self._uavs:
            for plan in uav.plan_requests():
                plans.append(replace(plan, order_index=order_index))
                order_index += 1

        return plans

    def _get_ue_uav_rate(self, source_uav: UAV, ue: UE) -> float:
        """计算 UE-UAV 速率（含输入验证）。"""
        num_associated = len(source_uav.current_covered_ues)
        if num_associated <= 0:
            raise RuntimeError(f"UAV{source_uav.id} has no associated UEs.")
        from environment import comm_model as comms
        channel_gain = comms.calculate_channel_gain(ue.pos, source_uav.pos)
        rate = comms.calculate_ue_uav_rate(channel_gain, num_associated)
        if not np.isfinite(rate) or rate <= 0.0:
            raise RuntimeError(
                f"Invalid UE-UAV rate: UAV={source_uav.id}, UE={ue.id}, rate={rate}"
            )
        return rate

    def _freeze_plan_estimates(self, plans: list[RequestPlan]) -> list[RequestPlan]:
        """用最终执行负载补 RouteEstimate。仅 unified_fixed_targets 使用。"""
        from dataclasses import replace

        # MBS 服务负载：循环外统计一次
        mbs_service_load = sum(
            1 for p in plans
            if p.request_type == 0 and p.target_idx == 2
        )

        frozen: list[RequestPlan] = []
        for plan in plans:
            if plan.request_type == 2:
                frozen.append(plan)
                continue

            if plan.ue_uav_rate is None:
                raise RuntimeError("Fixed-target plan has no UE-UAV rate.")
            ue_uav_rate = plan.ue_uav_rate

            source_uav = plan.source_uav
            source_mbs_rate = source_uav._uav_mbs_rate

            if plan.target_idx == 0:
                if plan.request_type == 0:
                    exec_load = source_uav.execution_service_load
                else:
                    exec_load = 0
                tgt_mbs = None
                mbs_for_est = 0

            elif plan.target_idx == 1:
                if plan.target_uav is None:
                    raise ValueError("Neighbor plan missing target_uav.")
                if plan.request_type == 0:
                    exec_load = plan.target_uav.execution_service_load
                else:
                    exec_load = 0
                tgt_mbs = plan.target_uav._uav_mbs_rate
                mbs_for_est = 0

            elif plan.target_idx == 2:
                exec_load = 0
                tgt_mbs = None
                mbs_for_est = mbs_service_load

            else:
                raise ValueError(f"Invalid target_idx: {plan.target_idx}")

            # 服务请求必须有正执行负载
            if plan.request_type == 0:
                if plan.target_idx in (0, 1) and exec_load <= 0:
                    raise ValueError(
                        f"Service plan has invalid execution load: "
                        f"UE={plan.ue.id}, target={plan.target_idx}, "
                        f"UAV={source_uav.id}, load={exec_load}"
                    )
                if plan.target_idx == 2 and mbs_for_est <= 0:
                    raise ValueError(
                        f"Service plan offloaded to MBS but mbs_service_load=0. "
                        f"UE={plan.ue.id}, UAV={source_uav.id}"
                    )

            from environment.uavs import estimate_route
            est = estimate_route(
                source_uav, plan.ue, plan.target_idx, plan.target_uav,
                exec_load, ue_uav_rate, source_mbs_rate,
                target_mbs_rate=tgt_mbs,
                mbs_execution_load=mbs_for_est,
            )

            frozen.append(replace(plan, estimate=est))

        return frozen

    # ============================================================
    # v1.5 联合卸载优化：阶段 1B — 坐标下降规划
    # ============================================================

    def _plan_all_requests(self) -> list[RequestPlan]:
        """iterative_latency / iterative_joint 模式的统一规划入口。"""
        from dataclasses import replace
        from environment import comm_model as comms

        t_start = time.perf_counter()

        # 1. 按原 per-UAV 随机顺序生成带索引上下文
        ordered_contexts: list[tuple[int, UAV, UE]] = []
        order_index = 0

        for uav in self._uavs:
            shuffled_indices = np.random.permutation(len(uav.current_covered_ues))
            for idx in shuffled_indices:
                ue = uav.current_covered_ues[idx]
                ordered_contexts.append((order_index, uav, ue))
                order_index += 1

        # 2. 按类型分类
        service_reqs: list[tuple[int, UAV, UE]] = []
        content_reqs: list[tuple[int, UAV, UE]] = []
        energy_reqs: list[tuple[int, UAV, UE]] = []

        for oi, uav, ue in ordered_contexts:
            req_type = ue.current_request[0]
            if req_type == 0:
                service_reqs.append((oi, uav, ue))
            elif req_type == 1:
                content_reqs.append((oi, uav, ue))
            elif req_type == 2:
                energy_reqs.append((oi, uav, ue))

        plans: list[RequestPlan] = []

        # 3. 内容：单轮枚举
        for oi, uav, ue in content_reqs:
            ue_uav_rate = self._get_ue_uav_rate(uav, ue)
            target_idx, target_uav, est = self._select_content_target(
                uav, ue, ue_uav_rate
            )
            plans.append(RequestPlan(
                source_uav=uav, ue=ue, request_type=1,
                ue_uav_rate=ue_uav_rate,
                target_idx=target_idx, target_uav=target_uav,
                estimate=est, order_index=oi,
            ))

        # 4. 能量：直接生成
        for oi, uav, ue in energy_reqs:
            plans.append(RequestPlan(
                source_uav=uav, ue=ue, request_type=2,
                order_index=oi,
            ))

        # 5. 服务：坐标下降
        if service_reqs:
            service_plans = self._coordinate_descent_service_planning(service_reqs)
            plans.extend(service_plans)
        else:
            service_plans = []
            self._last_offload_iterations = 0
            self._last_offload_converged = True

        # 6. 按 order_index 恢复原执行顺序
        plans.sort(key=lambda p: p.order_index)

        # ── J8 累计统计 ──
        self._last_offload_planning_time = time.perf_counter() - t_start
        self._offload_planning_times.append(self._last_offload_planning_time)
        self._offload_iteration_history.append(self._last_offload_iterations)
        self._offload_convergence_history.append(self._last_offload_converged)

        # ── MBS 服务卸载比率 ──
        if service_reqs:
            _targets_for_mbs = [p.target_idx for p in service_plans] if service_plans else []
            mbs_count = sum(1 for t in _targets_for_mbs if t == 2)
            self._mbs_service_ratio_history.append(mbs_count / len(service_reqs))
        else:
            self._mbs_service_ratio_history.append(0.0)

        # ── 归一化数据收集 ──
        if (config.COLLECT_ROUTE_NORMALIZATION_STATS
                and config.OFFLOAD_PIPELINE_MODE == "iterative_latency"):
            for plan in plans:
                if plan.estimate is not None:
                    self._route_latencies.append(plan.estimate.total_latency)
                    self._route_energies.append(plan.estimate.controllable_energy)

        return plans

    def _select_content_target(
        self, source_uav: UAV, ue: UE, ue_uav_rate: float
    ) -> tuple[int, UAV | None, RouteEstimate]:
        """内容请求：枚举本地/邻居/MBS，选代价最小的目标。"""
        source_mbs_rate = source_uav._uav_mbs_rate
        candidates = []

        # 本地
        est_local = estimate_route(source_uav, ue, 0, None, 0,
                                   ue_uav_rate, source_mbs_rate)
        candidates.append((_offload_cost(est_local), 0, None, est_local))

        # 邻居
        for neighbor in source_uav.neighbors:
            target_mbs_rate = neighbor._uav_mbs_rate
            est_neighbor = estimate_route(source_uav, ue, 1, neighbor, 0,
                                          ue_uav_rate, source_mbs_rate, target_mbs_rate)
            candidates.append((_offload_cost(est_neighbor), 1, neighbor, est_neighbor))

        # MBS
        est_mbs = estimate_route(source_uav, ue, 2, None, 0,
                                 ue_uav_rate, source_mbs_rate)
        candidates.append((_offload_cost(est_mbs), 2, None, est_mbs))

        best = min(candidates, key=lambda c: c[0])
        return best[1], best[2], best[3]

    def _coordinate_descent_service_planning(
        self, requests: list[tuple[int, UAV, UE]]
    ) -> list[RequestPlan]:
        """坐标下降：每次切换必须降低系统总代价。收敛至单请求邻域局部最优。"""
        from dataclasses import replace

        n = len(requests)
        if n == 0:
            self._last_offload_iterations = 0
            self._last_offload_converged = True
            return []

        # 初始分配：全本地
        targets = [0] * n
        target_uavs: list[UAV | None] = [None] * n

        for iteration in range(config.OFFLOAD_MAX_ITERATIONS):
            changed = False

            for i, (oi, source_uav, ue) in enumerate(requests):
                # 收集可行候选
                candidates: list[tuple[int, UAV | None]] = [(0, None)]
                for neighbor in source_uav.neighbors:
                    candidates.append((1, neighbor))
                candidates.append((2, None))

                best_target = targets[i]
                best_target_uav = target_uavs[i]

                # 当前系统总代价
                current_total, _, _, _ = self._evaluate_service_assignment(
                    requests, targets, target_uavs
                )
                best_total = current_total

                for cand_idx, cand_uav in candidates:
                    if cand_idx == targets[i] and cand_uav is target_uavs[i]:
                        continue

                    trial_targets = targets.copy()
                    trial_target_uavs = target_uavs.copy()
                    trial_targets[i] = cand_idx
                    trial_target_uavs[i] = cand_uav

                    trial_total, _, _, _ = self._evaluate_service_assignment(
                        requests, trial_targets, trial_target_uavs
                    )

                    if _is_better_cost(trial_total, best_total):
                        best_total = trial_total
                        best_target = cand_idx
                        best_target_uav = cand_uav

                if best_target != targets[i] or best_target_uav is not target_uavs[i]:
                    targets[i] = best_target
                    target_uavs[i] = best_target_uav
                    changed = True

            if not changed:
                break

        converged = not changed
        self._last_offload_iterations = iteration + 1
        self._last_offload_converged = converged

        if not converged:
            message = (
                f"Coordinate descent did not converge within "
                f"{config.OFFLOAD_MAX_ITERATIONS} iterations."
            )
            if config.VERIFY_TIMING:
                raise AssertionError(message)
            self._diag_print(f"[OFFLOAD_WARN] {message}")

        # J5 验证
        if config.VERIFY_TIMING:
            self._verify_j5_convergence(requests, targets, target_uavs)
            self._j5_verified_service_steps += 1

        # 用最终负载冻结 RouteEstimate
        uav_loads, mbs_load = self._count_assignment_loads(
            requests, targets, target_uavs
        )

        final_plans: list[RequestPlan] = []
        for i, (oi, source_uav, ue) in enumerate(requests):
            ue_uav_rate = self._get_ue_uav_rate(source_uav, ue)
            source_mbs_rate = source_uav._uav_mbs_rate

            tidx = targets[i]
            tuav = target_uavs[i]

            if tidx == 0:
                load = int(uav_loads[source_uav.id])
                tgt_mbs = None
                mbs_for_est = 0
            elif tidx == 1:
                load = int(uav_loads[tuav.id])
                tgt_mbs = tuav._uav_mbs_rate
                mbs_for_est = 0
            elif tidx == 2:
                load = 0
                tgt_mbs = None
                mbs_for_est = mbs_load
            else:
                raise ValueError(f"Invalid final service target: {tidx}")

            est = estimate_route(source_uav, ue, tidx, tuav, load,
                                 ue_uav_rate, source_mbs_rate, tgt_mbs,
                                 mbs_execution_load=mbs_for_est)

            final_plans.append(RequestPlan(
                source_uav=source_uav, ue=ue, request_type=0,
                ue_uav_rate=ue_uav_rate,
                target_idx=tidx, target_uav=tuav,
                estimate=est, order_index=oi,
            ))

        return final_plans

    def _count_assignment_loads(
        self,
        requests: list[tuple[int, UAV, UE]],
        targets: list[int],
        target_uavs: list[UAV | None],
    ) -> tuple[np.ndarray, int]:
        """统计给定分配下的 UAV 和 MBS 服务负载。"""
        uav_loads = np.zeros(config.NUM_UAVS, dtype=np.int32)
        mbs_load = 0
        for i in range(len(requests)):
            if targets[i] == 0:
                uav_loads[requests[i][1].id] += 1
            elif targets[i] == 1:
                if target_uavs[i] is None:
                    raise ValueError(f"Neighbor assignment has no target UAV at index {i}.")
                uav_loads[target_uavs[i].id] += 1
            elif targets[i] == 2:
                mbs_load += 1
            else:
                raise ValueError(f"Invalid assignment target: {targets[i]} at index {i}")
        return uav_loads, mbs_load

    def _evaluate_service_assignment(
        self,
        requests: list[tuple[int, UAV, UE]],
        targets: list[int],
        target_uavs: list[UAV | None],
    ) -> tuple[float, np.ndarray, int, list[RouteEstimate]]:
        """用统一最终负载重算全体服务请求的一致系统总代价。"""
        uav_loads, mbs_load = self._count_assignment_loads(
            requests, targets, target_uavs
        )
        mbs_effective = max(mbs_load, 1) if mbs_load > 0 else 0

        estimates: list[RouteEstimate] = []
        total_cost = 0.0

        for i, (oi, source_uav, ue) in enumerate(requests):
            ue_uav_rate = self._get_ue_uav_rate(source_uav, ue)
            source_mbs_rate = source_uav._uav_mbs_rate

            tidx = targets[i]
            tuav = target_uavs[i]

            if tidx == 0:
                load = int(uav_loads[source_uav.id])
                tgt_mbs = None
                mbs_for_est = 0
            elif tidx == 1:
                if tuav is None:
                    raise ValueError("Neighbor target missing target_uav.")
                load = int(uav_loads[tuav.id])
                tgt_mbs = tuav._uav_mbs_rate
                mbs_for_est = 0
            elif tidx == 2:
                load = 0
                tgt_mbs = None
                mbs_for_est = mbs_effective
            else:
                raise ValueError(f"Invalid target: {tidx}")

            est = estimate_route(source_uav, ue, tidx, tuav, load,
                                 ue_uav_rate, source_mbs_rate, tgt_mbs,
                                 mbs_execution_load=mbs_for_est)
            estimates.append(est)
            total_cost += _offload_cost(est)

        return total_cost, uav_loads, mbs_load, estimates

    def _verify_j5_convergence(
        self,
        requests: list[tuple[int, UAV, UE]],
        targets: list[int],
        target_uavs: list[UAV | None],
    ) -> None:
        """J5：坐标下降收敛后，任意单请求切换均无法降低系统总代价超过容差。"""
        final_total, _, _, _ = self._evaluate_service_assignment(
            requests, targets, target_uavs
        )

        for i, (oi, source_uav, ue) in enumerate(requests):
            candidates: list[tuple[int, UAV | None]] = [(0, None)]
            for neighbor in source_uav.neighbors:
                candidates.append((1, neighbor))
            candidates.append((2, None))

            for cand_idx, cand_uav in candidates:
                if cand_idx == targets[i] and cand_uav is target_uavs[i]:
                    continue

                trial_targets = targets.copy()
                trial_target_uavs = target_uavs.copy()
                trial_targets[i] = cand_idx
                trial_target_uavs[i] = cand_uav

                trial_total, _, _, _ = self._evaluate_service_assignment(
                    requests, trial_targets, trial_target_uavs
                )

                if _is_better_cost(trial_total, final_total):
                    raise AssertionError(
                        f"J5 FAILED: request {i} (UAV{source_uav.id}, UE{ue.id}) "
                        f"can switch from (idx={targets[i]}, uav={target_uavs[i]}) "
                        f"to (idx={cand_idx}, uav={cand_uav}) "
                        f"and reduce system cost: {trial_total} < {final_total}"
                    )

    def _get_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float, float]]:
        """
        返回: (每个无人机的奖励列表, (总延迟, 优化能耗, 公平性, 离线率))
        """
        # ========== 1. 计算各项指标 ==========
        total_latency: float = sum(
            ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY
            for ue in self._ues
        )

        # 可控能耗
        total_uav_energy = sum(uav.energy for uav in self._uavs)
        total_ue_tx_energy = sum(ue.tx_energy for ue in self._ues)
        optimization_energy = total_uav_energy + total_ue_tx_energy

        # 系统总能耗（保存供后续日志用）
        total_ue_static = sum(ue.static_energy for ue in self._ues)
        self._last_system_energy = optimization_energy + total_ue_static

        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues])
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics ** 2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics ** 2))

        offline_count: int = sum(1 for ue in self._ues if ue.battery_level < config.UE_CRITICAL_THRESHOLD)
        offline_rate: float = offline_count / config.NUM_UES

        # ========== 2. 全局奖励 ==========
        reward = (
                config.ALPHA_3 * np.log(max(jfi, 1e-6))
                - config.ALPHA_1 * np.log(max(total_latency, 1.0))
                - config.ALPHA_2 * np.log(max(optimization_energy, 1.0))
                - config.ALPHA_4 * np.log(1.0 + offline_rate)
        )

        # ========== 3. 混合奖励（修复：局部能耗含关联UE TX） ==========
        if config.USE_LOCAL_REWARD:
            rewards = []
            for uav in self._uavs:
                uav_avg_lat = uav.get_avg_served_latency()

                # 该 UAV 覆盖用户的 TX 能耗
                associated_ue_tx_energy = sum(
                    ue.tx_energy for ue in uav.current_covered_ues
                )
                local_optimization_energy = uav.energy + associated_ue_tx_energy

                norm_uav_lat = np.log(max(uav_avg_lat, 1.0))
                norm_uav_eng = np.log(max(local_optimization_energy, 1.0))

                local_reward = (
                    -config.ALPHA_1 * norm_uav_lat
                    -config.ALPHA_2 * norm_uav_eng
                )

                reward_i = (
                    (1.0 - config.LOCAL_REWARD_WEIGHT) * reward
                    + config.LOCAL_REWARD_WEIGHT * local_reward
                )
                rewards.append(reward_i)
        else:
            rewards = [reward] * config.NUM_UAVS

        # 碰撞/越界惩罚
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY

        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]

        # ========== 诊断日志 ==========
        if self._time_step % 100 == 0:
            lat_term = -config.ALPHA_1 * np.log(max(total_latency, 1.0))
            eng_term = -config.ALPHA_2 * np.log(max(optimization_energy, 1.0))
            fair_term = config.ALPHA_3 * np.log(max(jfi, 1e-6))
            off_term = -config.ALPHA_4 * np.log(1.0 + offline_rate)
            total = lat_term + eng_term + fair_term + off_term
            served_count = sum(1 for ue in self._ues if ue.assigned)

            self._diag_print(
                f"[Step {self._time_step}] "
                f"Fair={fair_term:+.3f} Lat={lat_term:+.3f} Eng={eng_term:+.3f} Off={off_term:+.3f} Total={total:+.3f} | "
                f"latency={total_latency:.2e} opt_energy={optimization_energy:.2e} fairness={jfi:.4f} offline={offline_rate:.3f} served={served_count}/{config.NUM_UES}"
            )

        if self._time_step % 500 == 0:
            total_moved = sum(uav._dist_moved for uav in self._uavs)
            avg_moved = total_moved / config.NUM_UAVS
            self._diag_print(f"[Step {self._time_step}] Movement: total={total_moved:.1f}m avg={avg_moved:.1f}m")
            for uav in self._uavs:
                n = len(uav._served_latencies)
                avg_l = uav.get_avg_served_latency()
                uav_energy = uav.energy
                ue_tx_energy = sum(ue.tx_energy for ue in uav.current_covered_ues)
                self._diag_print(
                    f"  UAV{uav.id}: served={n:3d} avg_lat={avg_l:.2f}s "
                    f"uav_energy={uav_energy:.2f}J ue_tx={ue_tx_energy:.4f}J "
                    f"local_opt={uav_energy + ue_tx_energy:.2f}J"
                )

        # === 验证 H：能耗守恒 ===
        if config.VERIFY_TIMING:
            # H1 & H2：所有能耗分项有限且非负 + UAV 分项求和
            for uav in self._uavs:
                components = np.array([
                    uav._fly_energy_slot,
                    uav._compute_energy_slot,
                    uav._comm_tx_energy_slot,
                    uav._comm_rx_energy_slot,
                    uav._wpt_energy_slot,
                ])
                if not np.all(np.isfinite(components)):
                    raise AssertionError(f"UAV{uav.id} has non-finite energy.")
                if np.any(components < -1e-10):
                    raise AssertionError(f"UAV{uav.id} has negative energy: {components}")

                uav_sum = components.sum()
                uav_prop = uav.energy
                if not np.isclose(uav_sum, uav_prop, rtol=1e-6, atol=1e-8):
                    raise AssertionError(f"UAV{uav.id} energy mismatch: sum={uav_sum} vs prop={uav_prop}")

                # H3：WPT 守恒
                expected_wpt = uav._harvested_energy_slot / config.WPT_EFFICIENCY
                if not np.isclose(uav._wpt_energy_slot, expected_wpt, rtol=1e-6, atol=1e-8):
                    raise AssertionError(
                        f"UAV{uav.id} WPT mismatch: cost={uav._wpt_energy_slot}, "
                        f"harvested={uav._harvested_energy_slot}, expected={expected_wpt}"
                    )

            # H4：UE 每时隙恰好结算一次电池
            for ue in self._ues:
                if ue._battery_updates_slot != 1:
                    raise AssertionError(
                        f"UE{ue.id} battery updated {ue._battery_updates_slot} times in one slot."
                    )

            # H5：优化能耗和系统能耗边界
            total_uav_check = sum(uav.energy for uav in self._uavs)
            total_ue_tx_check = sum(ue.tx_energy for ue in self._ues)
            total_ue_static_check = sum(ue.static_energy for ue in self._ues)
            optimization_check = total_uav_check + total_ue_tx_check
            system_check = optimization_check + total_ue_static_check

            if not np.isclose(optimization_check, optimization_energy, rtol=1e-6, atol=1e-8):
                raise AssertionError(
                    f"Optimization energy mismatch: calculated={optimization_check}, "
                    f"expected={optimization_energy}"
                )
            if not np.isclose(system_check, self._last_system_energy, rtol=1e-6, atol=1e-8):
                raise AssertionError(
                    f"System energy mismatch: calculated={system_check}, "
                    f"expected={self._last_system_energy}"
                )

        return rewards, (total_latency, optimization_energy, jfi, offline_rate)