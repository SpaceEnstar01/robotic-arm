#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import mujoco
from mujoco import viewer
from termcolor import colored

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.controllers.second_controller import UR5Controller

class GraspEnv:
    def __init__(self, model_path=None, render=True):
        if model_path is None:
            model_path = os.path.join(project_root, "assets", "UR5_gripper", "UR5gripper_2_finger.xml")
        
        self.controller = UR5Controller(model_path=model_path)
        self.render = render
        self.viewer = None
        
        if self.render:
            self._init_viewer()

    def _init_viewer(self):
        """初始化查看器（MuJoCo 3.2.3兼容版本）"""
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.cam.fixedcamid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_down"
        )
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    @property
    def model(self):
        return self.controller.model

    @property
    def data(self):
        return self.controller.data

    def reset(self):
        """重置环境到初始状态"""
        self.controller.reset()
        
        # 移动到home位置
        home_joints = self.data.qpos[self.controller.actuated_joint_ids[:6]].copy()
        home_joints[1] = -0.2  # shoulder_lift_joint
        home_joints[2] = 0.2  # elbow_joint
        
        result = self.controller.move_group_to_joint_target(
            group="Arm",
            target=home_joints.tolist(),
            tolerance=0.05,
            max_steps=2000,
            render=self.render
        )
        
        self.controller.open_gripper(render=self.render)
        return self._get_ee_pos()

    def _get_ee_pos(self):
        """获取末端执行器当前位置"""
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        return self.data.xpos[ee_id].copy()

    def execute_grasp(self, target_pos, lift_height=0.1):
        """
        执行抓取动作（分阶段移动）
        :param target_pos: 目标位置[x,y,z]
        :param lift_height: 抓取后抬升高度
        :return: 是否成功
        """
        # 阶段1: 移动到目标上方
        approach_pos = target_pos.copy()
        approach_pos[2] += 0.5
        
        print("阶段1: 移动到接近位置")
        result = self.controller.move_ee(
            ee_position=approach_pos,
            tolerance=0.3,
            max_steps=2000,
            render=self.render
        )
        if result != "success":
            return False

        # 阶段2: 下降到预抓取位置
        pregrasp_pos = target_pos.copy()
        pregrasp_pos[2] += 0.3
        
        print("阶段2: 移动到预抓取位置")
        result = self.controller.move_ee(
            ee_position=pregrasp_pos,
            tolerance=0.3,
            max_steps=1500,
            render=self.render
        )
        if result != "success":
            return False

        # 阶段3: 最终抓取位置
        print("阶段3: 移动到抓取位置")
        result = self.controller.move_ee(
            ee_position=target_pos,
            tolerance=0.2,
            max_steps=1000,
            render=self.render
        )
        if result != "success":
            return False

        # 阶段4: 关闭夹爪
        print("阶段4: 关闭夹爪")
        self.controller.close_gripper(render=self.render)
        time.sleep(0.5)

        # 阶段5: 抬升物体
        lift_pos = target_pos.copy()
        lift_pos[2] += lift_height
        
        print("阶段5: 抬升物体")
        result = self.controller.move_ee(
            ee_position=lift_pos,
            tolerance=0.3,
            max_steps=1500,
            render=self.render
        )
        
        return result == "success"

def main():
    try:
        # 初始化环境
        env = GraspEnv(render=True)
        
        # 示例抓取位置（根据场景调整）
        grasp_pos = np.array([0, -0.3, 1.3])  # x,y,z
        
        # 执行抓取
        success = env.execute_grasp(grasp_pos)
        print(colored(f"抓取 {'成功' if success else '失败'}", "green" if success else "red"))
        
        # 保持窗口打开
        while env.viewer.is_running:
            env.viewer.sync()
            time.sleep(1/60)
            
    except Exception as e:
        print(colored(f"发生错误: {e}", "red"))
    finally:
        if 'env' in locals():
            env.viewer.close()

if __name__ == "__main__":
    main()
    print("程序结束，请手动关闭窗口")