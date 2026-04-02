import config
from environment.env import Env
import numpy as np


def check_obstacle():
    """测试障碍物创建和碰撞检测"""

    # 1. 创建环境
    env = Env()

    print("=" * 50)
    print("障碍物测试")
    print("=" * 50)

    # 2. 打印障碍物信息
    print(f"\n📊 障碍物统计:")
    print(f"   静态障碍物: {len(env._static_obstacles)} 个")
    print(f"   动态障碍物: {len(env._dynamic_obstacles)} 个")

    for i, obs in enumerate(env._static_obstacles):
        print(f"   静态障碍物{i}: 位置({obs.pos[0]:.1f}, {obs.pos[1]:.1f}), 半径{obs.radius}m")

    for i, obs in enumerate(env._dynamic_obstacles):
        print(
            f"   动态障碍物{i}: 位置({obs.pos[0]:.1f}, {obs.pos[1]:.1f}), 速度({obs.velocity[0]:.1f}, {obs.velocity[1]:.1f})")

    # 3. 测试碰撞检测
    print("\n🔍 碰撞检测测试:")
    obs = env.reset()

    # 记录初始位置
    initial_positions = [uav.pos[:2].copy() for uav in env._uavs]

    # 运行50步，观察是否有碰撞
    collision_occurred = False
    for step in range(50):
        # 随机动作
        actions = np.random.uniform(-0.5, 0.5, (config.NUM_UAVS, 2))

        next_obs, rewards, metrics = env.step(actions)

        # 检查碰撞
        for uav in env._uavs:
            if uav.collision_violation:
                collision_occurred = True
                print(f"   ⚠️ 第{step}步: 无人机{uav.id}碰撞障碍物!")

        if step % 10 == 0:
            print(f"   第{step}步: 无碰撞")

    # 4. 输出结果
    print("\n" + "=" * 50)
    if collision_occurred:
        print("✅ 障碍物模块正常工作（检测到了碰撞）")
    else:
        print("⚠️ 警告：50步内没有检测到碰撞（可能需要增加障碍物或调整半径）")
    print("=" * 50)


if __name__ == "__main__":
    check_obstacle()