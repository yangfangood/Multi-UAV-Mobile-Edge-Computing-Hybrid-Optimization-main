# environment/obstacle.py
import numpy as np

import config


class StaticObstacle:
    """静态障碍物（建筑物、树木等）"""

    def __init__(self, obs_id, pos, radius):
        """
        初始化静态障碍物

        参数:
            obs_id: 障碍物唯一ID
            pos: 位置坐标 (x, y)
            radius: 障碍物半径（米）
        """
        self.id = obs_id
        self.pos = np.array(pos, dtype=np.float32)  # (x, y) 位置
        self.radius = radius  # 障碍物半径
        self.type = 'static'

    def check_collision(self, uav_pos, uav_radius):
        """检查是否与无人机碰撞
        参数:
            uav_pos: 无人机位置 (x, y, z)
            uav_radius: 无人机覆盖半径

        返回:
            True: 碰撞， False: 安全

        """
        # 计算水平距离，忽略高度差
        distance = np.linalg.norm(uav_pos[:2] - self.pos)
        #  # 碰撞条件：距离 < 障碍物半径 + 无人机半径
        return distance < (self.radius + uav_radius)

    def get_push_vector(self, uav_pos: np.ndarray, uav_radius: float) -> np.ndarray:
        """
        计算推开无人机的向量

        参数:
            uav_pos: 无人机位置
            uav_radius: 无人机半径

        返回:
            推开向量 (dx, dy)
        """
        direction = uav_pos[:2] - self.pos
        distance = np.linalg.norm(direction)

        if distance == 0:
            # 完全重合，随机方向推开
            direction = np.array([1, 0])
            distance = 1
        else:
            direction = direction / distance

        # 重叠量 = 障碍物半径 + 无人机半径 - 实际距离
        overlap = self.radius + uav_radius - distance

        # 推开的距离 = 重叠量 + 10米安全距离
        push_distance = max(overlap + 10, 0)

        return direction * push_distance

    def get_info(self):
        """返回障碍物信息（用于日志）"""
        return {
            'id': self.id,
            'pos': self.pos.tolist(),
            'radius': self.radius
        }


class DynamicObstacle(StaticObstacle):

    def __init__(self, obs_id: int, pos: np.ndarray, radius: float,
                 velocity: np.ndarray, area_boundary: tuple):
        """

         初始化动态障碍物

        参数:
            obs_id: 障碍物ID
            pos: 初始位置
            radius: 半径
            velocity: 初始速度 (vx, vy) 米/秒
            area_boundary: 区域边界 (width, height)

        """
        super().__init__(obs_id, pos, radius)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.area_width = area_boundary[0]
        self.area_height = area_boundary[1]
        self.type = 'dynamic'
        # 记录轨迹（用于可视化）
        self.trajectory = [pos.copy()]

    def update(self):
        """
         更新动态障碍物位置（随机游走 + 边界反弹）
        """
        # 按速度移动
        self.pos += self.velocity * config.TIME_SLOT_DURATION
        # 边界反弹（像台球一样）
        if self.pos[0] < 0:
            self.pos[0] = -self.pos[0]  # 弹回
            self.velocity[0] *= -1  # 反向
        elif self.pos[0] > self.area_width:
            self.pos[0] = 2 * self.area_width - self.pos[0]
            self.velocity[0] *= -1

        if self.pos[1] < 0:
            self.pos[1] = -self.pos[1]
            self.velocity[1] *= -1
        elif self.pos[1] > self.area_height:
            self.pos[1] = 2 * self.area_height - self.pos[1]
            self.velocity[1] *= -1

        # 记录轨迹
        self.trajectory.append(self.pos.copy())