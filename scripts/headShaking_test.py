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
    # 加载模型
    model_path = os.path.join(project_root, "assets", "UR5_gripper", "UR5gripper_2_finger.xml")
    controller = UR5Controller(model_path=model_path)

    # 设置目标末端位置
    target_pos = np.array([0.5, -0.5, 1.5])  # 根据模型调整

    with viewer.launch_passive(controller.model, controller.data) as v:
        print(f"📍 移动末端执行器到目标位置: {target_pos}")

        # 1️⃣ 先执行 move_ee，使用雅可比伪逆移动末端
        result = controller.move_ee(
            ee_position=target_pos,
            tolerance=0.08,
            max_steps=2000,
            render=False  # 外部 viewer 渲染
        )

        if result != "success":
            print(f"❌ 移动失败: {result}")
            return
        else:
            print("✅ 成功到达目标末端位置！")

        # 2️⃣ 到达目标后开始点头 or 摇头
        print("🤖 开始执行社交动作（点头 / 摇头）")

        base_qpos = controller.data.qpos.copy()
        mode = "shake"   # 可选: "nod"（点头）或 "shake"（摇头）
        amplitude = 0.5
        freq = 0.01
        step = 0
        direction = 1

        while v.is_running:
            offset = amplitude * direction * np.sin(step * freq)
            q_target = base_qpos.copy()

            if mode == "shake":
                q_target[4] += offset  # 控制末端（Yaw） => 摇头
            elif mode == "nod":
                q_target[2] -= offset  # 控制中部（Pitch）=> 点头

            controller.move_group_to_joint_target(
                group="Arm",
                target=q_target[:6].tolist(),
                tolerance=0.05,
                max_steps=5,
                render=False,
                quiet=True
            )

            controller.simulate_once()
            v.sync()
            step += 1

            if step % 100 == 0:
                direction *= -1

if __name__ == "__main__":
    main()
