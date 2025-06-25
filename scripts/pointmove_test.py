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

    # 目标末端位置（请根据你模型实际尺寸修改）
    target_pos = np.array([0.5, -0.5, 1.5])  # 单位: 米
    #sucessful data [0.05, -0.3, 1.3]  or [0.5, -0.3, 1.3],[0.5, -0.5, 1.5]

    # 启动 viewer
    with viewer.launch_passive(controller.model, controller.data) as v:
        print(f"移动末端执行器到目标位置: {target_pos}")

        # 调用雅可比伪逆运动
        result = controller.move_ee(
            ee_position=target_pos,
            tolerance=0.08,
            max_steps=2000,
            render=False  # 使用 viewer 外部渲染
        )

        if result == "success":
            print("✅ 成功到达目标末端位置！")
        else:
            print(f"❌ 移动失败: {result}")

        # 等待查看结果
        while v.is_running:
            v.sync()

if __name__ == "__main__":
    main()

