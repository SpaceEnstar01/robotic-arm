#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
from mujoco import viewer

# 设置路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.controllers.second_controller import UR5Controller

def main():
    # 加载控制器
    model_path = os.path.join(project_root, "assets", "UR5_gripper", "UR5gripper_2_finger.xml")
    controller = UR5Controller(model_path=model_path)

    # 启动 viewer
    with viewer.launch_passive(controller.model, controller.data) as v:
        time.sleep(0.5)
        positions = []  # 记录末端轨迹

        # 末端将沿Z方向上下移动形成路径
        init_pos = controller.data.xpos[mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_BODY, controller.ee_body)].copy()

        for i in range(30):
            delta_z = 0.1 * np.sin(i * 0.2)
            target_pos = init_pos + np.array([0, 0, delta_z])
            print(f"移动至: {target_pos}")
            controller.move_ee(target_pos, tolerance=0.05, max_steps=1000, render=False)

            # 记录当前位置
            ee_id = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_BODY, controller.ee_body)
            current_pos = controller.data.xpos[ee_id].copy()
            positions.append(current_pos)

            for _ in range(10):  # 模拟几步以便 viewer 更新
                controller.simulate_once()
                v.sync()

        # 可视化轨迹
        positions = np.array(positions)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', color='blue')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("UR5 末端轨迹")
        plt.show()

if __name__ == "__main__":
    main()

