# environment/obstacle.py
import numpy as np
import config


class Obstacle:
    """障碍物基类"""

    def __init__(self, obs_id, pos, radius):
        self.id = obs_id
        self.pos = np.array(pos, dtype=np.float32)  # (x, y) 位置
        self.radius = radius  # 障碍物半径
        self.active = True

    def check_collision(self, uav_pos, uav_radius):
        """检查是否与无人机碰撞"""
        distance = np.linalg.norm(uav_pos[:2] - self.pos)
        return distance < (self.radius + uav_radius)

    def update(self):
        """更新障碍物状态（基类空实现）"""
        pass

    def get_obs(self):
        """返回障碍物观测（用于无人机感知）"""
        return np.array([self.pos[0], self.pos[1], self.radius])


class StaticObstacle(Obstacle):
    """静态障碍物（建筑物、树木等）"""

    def __init__(self, obs_id, pos, radius):
        super().__init__(obs_id, pos, radius)
        self.type = 'static'


class DynamicObstacle(Obstacle):
    """动态障碍物（其他飞行器、移动物体）"""

    def __init__(self, obs_id, pos, radius, velocity, area_boundary):
        super().__init__(obs_id, pos, radius)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.area_width = area_boundary[0]
        self.area_height = area_boundary[1]
        self.type = 'dynamic'
        self.trajectory = [pos.copy()]  # 记录轨迹用于分析

    def update(self):
        """更新动态障碍物位置（随机移动或预设路径）"""
        # 简单随机游走模型
        self.pos += self.velocity * config.TIME_SLOT_DURATION

        # 边界反弹（像台球一样）
        if self.pos[0] < 0 or self.pos[0] > self.area_width:
            self.velocity[0] *= -1
            self.pos[0] = np.clip(self.pos[0], 0, self.area_width)

        if self.pos[1] < 0 or self.pos[1] > self.area_height:
            self.velocity[1] *= -1
            self.pos[1] = np.clip(self.pos[1], 0, self.area_height)

        # 记录轨迹
        self.trajectory.append(self.pos.copy())

    def set_waypoint_path(self, waypoints):
        """设置预设路径点（可选）"""
        self.waypoints = waypoints
        self.current_target = 0
        # 实现路径跟随逻辑...