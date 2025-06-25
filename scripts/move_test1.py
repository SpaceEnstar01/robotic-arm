#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import mujoco
from mujoco import viewer

# 设置路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.controllers.second_controller import UR5Controller

def main():
    # 加载模型路径
    model_path = os.path.join(project_root, "assets", "UR5_gripper", "UR5gripper_2_finger.xml")
    controller = UR5Controller(model_path=model_path)
    
    # 启动可视化窗口
    with viewer.launch_passive(controller.model, controller.data) as v:
        print("开始周期性运动，按 Ctrl+C 退出")
        step = 0
        direction = 1
        base_qpos = controller.data.qpos.copy()

        while v.is_running:

            offset = 0.5 * direction * np.sin(step * 0.01)
            q_target = base_qpos.copy()
            q_target[6] += offset  # shoulder_lift
            #q_target[2] -= offset  # elbow

            controller.move_group_to_joint_target(
                group="Arm",
                target=q_target[:6].tolist(),  # ✅ 修复点
                tolerance=0.05,
                max_steps=5,
                render=False,
                quiet=True
            )

            # 正反交替控制 joint 1 和 joint 2


            controller.simulate_once()
            v.sync()
            step += 1
            if step % 100 == 0:
                direction *= -1  # 改变方向

if __name__ == "__main__":
    main()
