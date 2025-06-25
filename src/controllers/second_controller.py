#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UR5机械臂控制器，基于MuJoCo 3.2.3，使用雅可比伪逆逆运动学
"""

import os
import numpy as np
from simple_pid import PID
from termcolor import colored
import mujoco
from mujoco import mjtObj
import mujoco.viewer
import time
from typing import List, Dict, Optional
from collections import defaultdict

class UR5Controller:
    """
    UR5机械臂控制器，支持关节空间控制和基于雅可比伪逆的笛卡尔空间控制
    适配MuJoCo 3.2.3
    """
    
    def __init__(self, model_path: str = None, data_path: str = None):
        """
        初始化控制器
        
        Args:
            model_path: MuJoCo模型文件路径
            data_path: 数据路径
        """
        # 设置模型路径
        self.data_path = data_path if data_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/UR5_gripper")
        self.model_path = model_path if model_path else os.path.join(self.data_path, "UR5gripper_2_finger.xml")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # 初始化控制器参数
        self.groups = defaultdict(list)
        self.create_lists()
        self.create_default_groups()
        self.reached_target = False
        self.current_output = np.zeros(self.model.nu)
        self.last_movement_steps = 0
        
        # 雅可比IK参数
        self.damping = 0.1  # 阻尼系数，处理奇异性
        self.max_joint_vel = 0.1  # 最大关节速度(rad/step)
        self.ee_body = "ee_link"  # 末端执行器体名称
        
        # 初始化仿真
        self.simulate_once()
    
    def create_lists(self):
        """
        创建PID控制器列表
        """
        self.controller_list = []
        sample_time = 0.0001
        p_scale = 3
        d_scale = 0.1
        
        # 6个手臂关节 + 1个夹爪
        self.controller_list.append(PID(7*p_scale, 0, 1.1*d_scale, setpoint=0, output_limits=(-2, 2), sample_time=sample_time))  # shoulder_pan
        self.controller_list.append(PID(10*p_scale, 0, 1.0*d_scale, setpoint=-1.57, output_limits=(-2, 2), sample_time=sample_time))  # shoulder_lift
        self.controller_list.append(PID(5*p_scale, 0, 0.5*d_scale, setpoint=1.57, output_limits=(-2, 2), sample_time=sample_time))  # elbow
        self.controller_list.append(PID(7*p_scale, 0, 0.1*d_scale, setpoint=-1.57, output_limits=(-1, 1), sample_time=sample_time))  # wrist_1
        self.controller_list.append(PID(5*p_scale, 0, 0.1*d_scale, setpoint=-1.57, output_limits=(-1, 1), sample_time=sample_time))  # wrist_2
        self.controller_list.append(PID(5*p_scale, 0, 0.1*d_scale, setpoint=0, output_limits=(-1, 1), sample_time=sample_time))  # wrist_3
        self.controller_list.append(PID(2.5*p_scale, 0, 0.0*d_scale, setpoint=0, output_limits=(-1, 1), sample_time=sample_time))  # gripper
        
        self.current_target_joint_values = np.array([controller.setpoint for controller in self.controller_list])
        self.actuators = self._create_actuators_list()
    
    def _create_actuators_list(self) -> List[List]:
        """创建执行器列表"""
        actuators = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            item = [
                i,
                mujoco.mj_id2name(self.model, mjtObj.mjOBJ_ACTUATOR, i),
                joint_id,
                mujoco.mj_id2name(self.model, mjtObj.mjOBJ_JOINT, joint_id),
                self.controller_list[i]
            ]
            actuators.append(item)
        return actuators
    
    def create_default_groups(self):
        """创建默认关节组"""
        self.groups["All"] = list(range(self.model.nu))
        self.create_group("Arm", list(range(6)))
        self.create_group("Gripper", [6])
        self.actuated_joint_ids = np.array([actuator[2] for actuator in self.actuators])
    
    def create_group(self, group_name: str, idx_list: List[int]):
        """
        创建关节组
        """
        try:
            assert len(idx_list) <= self.model.nu, "指定的关节数量过多!"
            assert group_name not in self.groups, f"名为 {group_name} 的组已存在!"
            assert max(idx_list) < self.model.nu, "包含无效的执行器ID"
            
            self.groups[group_name] = idx_list
            print(f"创建新控制组 '{group_name}'.")
            
        except Exception as e:
            print(f"Error creating group: {e}")
    
    def show_model_info(self):
        """显示模型信息"""
        print("\n物体数量: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print(f"物体 ID: {i}, 名称: {mujoco.mj_id2name(self.model, mjtObj.mjOBJ_BODY, i)}")
        
        print("\n关节数量: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(f"关节 ID: {i}, 名称: {mujoco.mj_id2name(self.model, mjtObj.mjOBJ_JOINT, i)}, 限制: {self.model.jnt_range[i]}")
        
        print("\n执行器数量: {}".format(self.model.nu))
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            print(f"执行器 ID: {i}, 名称: {mujoco.mj_id2name(self.model, mjtObj.mjOBJ_ACTUATOR, i)}, "
                  f"控制关节: {mujoco.mj_id2name(self.model, mjtObj.mjOBJ_JOINT, joint_id)}, 控制范围: {self.model.actuator_ctrlrange[i]}")
        
        print("\n相机数量: {}".format(self.model.ncam))
        for i in range(self.model.ncam):
            print(f"相机 ID: {i}, 名称: {mujoco.mj_id2name(self.model, mjtObj.mjOBJ_CAMERA, i)}")
    
    def compute_jacobian(self) -> np.ndarray:
        """
        计算末端执行器的雅可比矩阵
        Returns:
            J: 3x6雅可比矩阵（仅平移部分）
        """
        ee_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, self.ee_body)
        if ee_id == -1:
            raise ValueError(f"末端执行器 '{self.ee_body}' 未找到")
        
        # 初始化雅可比矩阵（3xnv，nv为自由度数）
        jacp = np.zeros((3, self.model.nv))  # 平移雅可比
        mujoco.mj_jac(self.model, self.data, jacp, None, self.data.xpos[ee_id], ee_id)
        
        # 提取手臂关节的雅可比（前6个自由度，假设每个关节1个自由度）
        arm_dofs = self.actuated_joint_ids[:6]
        J = jacp[:, arm_dofs]
        return J
    
    def move_ee(self, ee_position: List[float], tolerance: float = 0.01, max_steps: int = 2000, render: bool = True) -> str:
        """
        使用雅可比伪逆移动末端执行器到目标位置
        
        Args:
            ee_position: 目标位置 [x, y, z]
            tolerance: 位置容差(m)
            max_steps: 最大迭代步数
            render: 是否渲染
        
        Returns:
            移动结果 ("success" 或错误信息)
        """
        try:
            target_pos = np.array(ee_position)
            ee_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, self.ee_body)
            if ee_id == -1:
                return f"末端执行器 '{self.ee_body}' 未找到"
            
            steps = 0
            while steps < max_steps:
                # 当前末端位置
                current_pos = self.data.xpos[ee_id].copy()
                error = target_pos - current_pos
                error_norm = np.linalg.norm(error)
                
                if error_norm < tolerance:
                    print(colored(f"末端执行器到达目标，误差: {error_norm:.4f}m ({steps} 步)", "green"))
                    return "success"
                
                # 计算雅可比
                J = self.compute_jacobian()
                
                # 阻尼最小二乘法伪逆
                JJT = J @ J.T
                damping_matrix = self.damping * np.eye(3)
                J_pinv = J.T @ np.linalg.inv(JJT + damping_matrix)
                
                # 计算关节速度
                cartesian_vel = error * 0.5  # 比例增益
                joint_vel = J_pinv @ cartesian_vel
                
                # 限制关节速度
                joint_vel = np.clip(joint_vel, -self.max_joint_vel, self.max_joint_vel)
                
                # 当前关节角度
                current_joints = self.data.qpos[self.actuated_joint_ids[:6]].copy()
                # 更新关节目标
                target_joints = (current_joints + joint_vel).tolist()
                
                # 移动到新关节目标
                result = self.move_group_to_joint_target(
                    group="Arm",
                    target=target_joints,
                    tolerance=0.1,
                    max_steps=10000,
                    render=render,
                    quiet=True
                )
                
                if result != "success":
                    print(f"关节移动失败: {result}")
                    return result
                
                steps += 1
                if steps % 100 == 0:
                    print(f"步骤 {steps}: 位置误差 {error_norm:.4f}m")
                
                if render:
                    self.render()
            
            print(colored(f"达到最大步数: {max_steps}", "red"))
            return f"达到最大步数: {max_steps}"
            
        except Exception as e:
            print(f"移动末端执行器错误: {e}")
            return "error"
    
    def move_group_to_joint_target(
        self,
        group: str = "All",
        target: List[float] = None,
        tolerance: float = 0.1,
        max_steps: int = 10000,
        render: bool = True,
        quiet: bool = False
    ) -> str:
        """
        将指定关节组移动到目标位置
        """
        try:
            assert group in self.groups, f"不存在名为 {group} 的组!"
            if target is not None:
                assert len(target) == len(self.groups[group]), f"组 {group} 的目标维度不匹配!"
                
            ids = self.groups[group]
            steps = 1
            result = ""
            self.reached_target = False
            deltas = np.zeros(self.model.nu)
            
            if target is not None:
                for i, v in enumerate(ids):
                    self.current_target_joint_values[v] = target[i]
                    self.actuators[v][4].setpoint = target[i]
            
            while not self.reached_target:
                current_joint_values = self.data.qpos[self.actuated_joint_ids]
                
                for j in range(self.model.nu):
                    self.current_output[j] = self.actuators[j][4](current_joint_values[j])
                    self.data.ctrl[j] = self.current_output[j]
                
                for i in ids:
                    deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])
                
                if steps % 1000 == 0 and target is not None and not quiet:
                    print(f"将组 {group} 移动到关节目标! 最大误差: {max(deltas)}, 关节: {self.actuators[np.argmax(deltas)][3]}")
                
                if max(deltas) < tolerance:
                    if target is not None and not quiet:
                        print(colored(f"组 {group} 的关节值在要求的容差内! ({steps} 步)", "green"))
                    result = "success"
                    self.reached_target = True
                
                if steps > max_steps:
                    if not quiet:
                        print(colored(f"达到最大步数: {max_steps}", "red"))
                        print("误差: ", deltas)
                    result = f"达到最大步数: {max_steps}"
                    break
                
                self.simulate_once()
                if render:
                    self.render()
                steps += 1
                
            self.last_movement_steps = steps
            return result
            
        except Exception as e:
            print(f"关节目标移动错误: {e}")
            return "error"
    
    def open_gripper(self, render: bool = True) -> str:
        """张开夹爪"""
        return self.move_group_to_joint_target(
            group="Gripper", 
            target=[0.4], 
            max_steps=1000, 
            tolerance=0.05, 
            render=render
        )
    
    def close_gripper(self, render: bool = True) -> str:
        """关闭夹爪"""
        return self.move_group_to_joint_target(
            group="Gripper", 
            target=[-0.4], 
            tolerance=0.01, 
            render=render
        )
    
    def stay(self, duration: int, render: bool = True):
        """
        保持当前位置
        """
        start_time = time.time()
        elapsed = 0
        while elapsed < duration:
            self.move_group_to_joint_target(max_steps=10, tolerance=0.0001, render=render, quiet=True)
            elapsed = (time.time() - start_time) * 1000
    
    def start_viewer(self):
        """启动查看器"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def render(self):
        """渲染场景"""
        if self.viewer is None:
            self.start_viewer()
        self.viewer.sync()
    
    def simulate_once(self):
        """执行一次仿真步"""
        mujoco.mj_step(self.model, self.data)
    


    def close(self):
        """Closes the viewer and performs cleanup."""
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def reset(self):
        """Resets the simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.render()
