#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进版UR5机械臂控制器，专为抓取任务优化
基于MuJoCo 3.x，包含以下改进：
1. 更稳定的逆运动学求解
2. 增强的关节限制检查
3. 集成的调试工具
4. 优化的抓取控制逻辑
"""

import os
import numpy as np
from simple_pid import PID
from termcolor import colored
import mujoco
from mujoco import mjtObj, viewer
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

class GraspController:
    """
    改进版UR5抓取控制器，继承自second_controller的核心功能
    添加专为抓取任务优化的方法
    """
    
    def __init__(self, model_path: str = None, data_path: str = None, debug: bool = False):
        """
        初始化抓取控制器
        
        Args:
            model_path: MuJoCo模型文件路径
            data_path: 数据路径
            debug: 是否启用调试模式
        """
        # 设置模型路径
        self.data_path = data_path if data_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/UR5_gripper")
        self.model_path = model_path if model_path else os.path.join(self.data_path, "UR5gripper_2_finger_many_objects.xml")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.debug = debug
        self.debug_data = []
        
        # 初始化控制器参数
        self._init_control_params()
        self._create_actuator_groups()
        self._setup_pid_controllers()
        
        # 初始化仿真
        self.simulate_once()
    
    def _init_control_params(self):
        """初始化控制参数"""
        # 逆运动学参数
        self.ik_params = {
            'damping': 0.05,           # 阻尼系数
            'max_joint_vel': 0.2,      # 最大关节速度(rad/step)
            'step_scale': 0.3,         # 步长缩放因子
            'max_retry': 3,            # 失败重试次数
            'pos_tolerance': 0.005,    # 位置容差(m)
            'rot_tolerance': 0.1       # 旋转容差(rad)
        }
        
        # 抓取参数
        self.grasp_params = {
            'approach_height': 0.15,   # 接近高度(m)
            'pregrasp_height': 0.05,   # 预抓取高度(m)
            'lift_height': 0.1,        # 抬升高度(m)
            'gripper_open': 0.4,       # 夹爪张开位置
            'gripper_close': -0.4      # 夹爪闭合位置
        }
        
        # 末端执行器设置
        self.ee_body = "ee_link"
        self.ee_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, self.ee_body)
        if self.ee_id == -1:
            raise ValueError(f"末端执行器 '{self.ee_body}' 未找到")
    
    def _create_actuator_groups(self):
        """创建执行器组"""
        self.groups = defaultdict(list)
        self.groups["All"] = list(range(self.model.nu))
        self.create_group("Arm", list(range(6)))
        self.create_group("Gripper", [6])
        
        # 获取关节ID
        self.actuated_joint_ids = np.array([
            self.model.actuator_trnid[i, 0] 
            for i in range(self.model.nu)
        ])
    
    def _setup_pid_controllers(self):
        """设置PID控制器"""
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
    
    def create_group(self, group_name: str, idx_list: List[int]):
        """
        创建关节组
        
        Args:
            group_name: 组名称
            idx_list: 执行器索引列表
        """
        try:
            assert len(idx_list) <= self.model.nu, "指定的关节数量过多!"
            assert group_name not in self.groups, f"名为 {group_name} 的组已存在!"
            assert max(idx_list) < self.model.nu, "包含无效的执行器ID"
            
            self.groups[group_name] = idx_list
            print(f"创建新控制组 '{group_name}'.")
            
        except Exception as e:
            print(f"Error creating group: {e}")
    
    def compute_jacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算末端执行器的完整雅可比矩阵（平移+旋转）
        
        Returns:
            J_p: 3x6平移雅可比矩阵
            J_r: 3x6旋转雅可比矩阵
        """
        jacp = np.zeros((3, self.model.nv))  # 平移雅可比
        jacr = np.zeros((3, self.model.nv))  # 旋转雅可比
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.xpos[self.ee_id], self.ee_id)
        
        # 提取手臂关节的雅可比
        arm_dofs = self.actuated_joint_ids[:6]
        J_p = jacp[:, arm_dofs]
        J_r = jacr[:, arm_dofs]
        
        return J_p, J_r
    
    def _validate_joint_positions(self, joints: np.ndarray) -> bool:
        """
        验证关节位置是否在限制范围内
        
        Args:
            joints: 关节角度数组
            
        Returns:
            bool: 是否所有关节都在限制范围内
        """
        limits = self.model.jnt_range[self.actuated_joint_ids[:6]]
        return np.all((joints >= limits[:, 0]) & (joints <= limits[:, 1]))
    
    def move_ee(self, 
            target_pos: np.ndarray, 
            target_quat: Optional[np.ndarray] = None,
            tolerance: float = None,
            max_steps: int = None,
            render: bool = True) -> str:
        """
        修正后的末端执行器移动控制
        """
        # 参数处理
        tolerance = tolerance or self.ik_params['pos_tolerance']
        max_steps = max_steps or self.ik_params['max_steps']
        
        try:
            target_pos = np.asarray(target_pos)
            if target_quat is not None:
                target_quat = np.asarray(target_quat)
                if target_quat.shape != (4,):
                    raise ValueError("目标四元数必须是形状为(4,)的数组")
            
            steps = 0
            last_error = np.inf
            
            while steps < max_steps:
                # 当前位置和姿态
                current_pos = self.data.xpos[self.ee_id].copy()
                current_quat = self.data.xquat[self.ee_id].copy()
                
                # 计算位置误差
                pos_error = target_pos - current_pos
                pos_error_norm = np.linalg.norm(pos_error)
                
                # 计算姿态误差（如果指定了目标姿态）
                rot_error = 0.0
                if target_quat is not None:
                    rot_error = self._quat_distance(current_quat, target_quat)
                
                # 检查是否到达目标
                if pos_error_norm < tolerance and (target_quat is None or rot_error < self.ik_params['rot_tolerance']):
                    print(colored(f"末端执行器到达目标 (步数: {steps}, 位置误差: {pos_error_norm:.4f}m, 旋转误差: {rot_error:.4f}rad)", "green"))
                    return "success"
                
                # 计算雅可比
                J_p, J_r = self.compute_jacobian()
                
                # 根据是否控制姿态选择雅可比组合
                if target_quat is not None:
                    J = np.vstack([J_p, J_r])  # 6xn 雅可比
                    error = np.concatenate([
                        pos_error * self.ik_params['step_scale'],
                        self._quat_diff(current_quat, target_quat)
                    ])
                    damping = self.ik_params['damping'] * np.eye(6)  # 6x6 阻尼矩阵
                else:
                    J = J_p  # 3xn 雅可比
                    error = pos_error * self.ik_params['step_scale']
                    damping = self.ik_params['damping'] * np.eye(3)  # 3x3 阻尼矩阵
                
                # 阻尼最小二乘法伪逆
                JJT = J @ J.T
                J_pinv = J.T @ np.linalg.inv(JJT + damping)
                
                # 计算关节速度
                joint_vel = J_pinv @ error
                joint_vel = np.clip(joint_vel, -self.ik_params['max_joint_vel'], self.ik_params['max_joint_vel'])
                
                # 更新关节目标
                current_joints = self.data.qpos[self.actuated_joint_ids[:6]].copy()
                target_joints = current_joints + joint_vel
                
                # 验证关节位置
                if not self._validate_joint_positions(target_joints):
                    print(colored("警告: 关节超出限制!", "yellow"))
                    target_joints = np.clip(target_joints, 
                                        self.model.jnt_range[self.actuated_joint_ids[:6], 0],
                                        self.model.jnt_range[self.actuated_joint_ids[:6], 1])
                
                # 移动到新关节目标
                result = self.move_group_to_joint_target(
                    group="Arm",
                    target=target_joints.tolist(),
                    tolerance=0.1,
                    max_steps=100,
                    render=render,
                    quiet=True
                )
                
                if result != "success":
                    return result
                
                # 调试数据记录
                if self.debug:
                    self.debug_data.append({
                        'step': steps,
                        'pos_error': pos_error_norm,
                        'rot_error': rot_error if target_quat is not None else 0,
                        'joint_pos': current_joints.copy(),
                        'target_pos': target_pos.copy()
                    })
                
                steps += 1
                last_error = pos_error_norm
                
                if steps % 100 == 0:
                    print(f"步骤 {steps}: 位置误差 {pos_error_norm:.4f}m, 旋转误差 {rot_error:.4f}rad")
            
            print(colored(f"达到最大步数: {max_steps}, 最终误差: {last_error:.4f}m", "red"))
            return f"达到最大步数: {max_steps}"
            
        except Exception as e:
            print(f"移动末端执行器错误: {e}")
            return "error"
    
    def _quat_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """计算两个四元数之间的角度距离"""
        return 2 * np.arccos(np.abs(np.dot(q1, q2)))
    
    def _quat_diff(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """计算四元数误差（用于控制）"""
        return -mujoco.mju_subQuat(target, current)
    
    def execute_grasp(self, 
                     target_pos: np.ndarray,
                     target_quat: Optional[np.ndarray] = None,
                     render: bool = True) -> bool:
        """
        完整的抓取执行序列
        
        Args:
            target_pos: 抓取目标位置 [x,y,z]
            target_quat: 抓取目标姿态 (可选)
            render: 是否渲染
            
        Returns:
            bool: 抓取是否成功
        """
        try:
            # 1. 移动到接近位置
            approach_pos = target_pos.copy()
            approach_pos[2] += self.grasp_params['approach_height']
            
            print("阶段1: 移动到接近位置")
            result = self.move_ee(
                approach_pos,
                target_quat,
                tolerance=0.01,
                max_steps=2000,
                render=render
            )
            if result != "success":
                return False
            
            # 2. 移动到预抓取位置
            pregrasp_pos = target_pos.copy()
            pregrasp_pos[2] += self.grasp_params['pregrasp_height']
            
            print("阶段2: 移动到预抓取位置")
            result = self.move_ee(
                pregrasp_pos,
                target_quat,
                tolerance=0.005,
                max_steps=1500,
                render=render
            )
            if result != "success":
                return False
            
            # 3. 移动到抓取位置
            print("阶段3: 移动到抓取位置")
            result = self.move_ee(
                target_pos,
                target_quat,
                tolerance=0.003,
                max_steps=1000,
                render=render
            )
            if result != "success":
                return False
            
            # 4. 关闭夹爪
            print("阶段4: 关闭夹爪")
            self.close_gripper(render=render)
            time.sleep(0.5)
            
            # 5. 抬升物体
            lift_pos = target_pos.copy()
            lift_pos[2] += self.grasp_params['lift_height']
            
            print("阶段5: 抬升物体")
            result = self.move_ee(
                lift_pos,
                target_quat,
                tolerance=0.005,
                max_steps=1500,
                render=render
            )
            
            return result == "success"
            
        except Exception as e:
            print(f"抓取执行错误: {e}")
            return False
    
    # 保留second_controller.py中的其他方法（move_group_to_joint_target, open_gripper, close_gripper等）
    # 在 GraspController 类中添加以下方法（如果缺失）
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
            deltas = np.zeros(self.model.nu)
            
            if target is not None:
                for i, v in enumerate(ids):
                    self.current_target_joint_values[v] = target[i]
                    self.actuators[v][4].setpoint = target[i]
            
            while steps <= max_steps:
                current_joint_values = self.data.qpos[self.actuated_joint_ids]
                
                for j in range(self.model.nu):
                    self.data.ctrl[j] = self.actuators[j][4](current_joint_values[j])
                
                for i in ids:
                    deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])
                
                if steps % 1000 == 0 and target is not None and not quiet:
                    print(f"将组 {group} 移动到关节目标! 最大误差: {max(deltas)}")
                
                if max(deltas) < tolerance:
                    if target is not None and not quiet:
                        print(colored(f"组 {group} 的关节值在要求的容差内! ({steps} 步)", "green"))
                    result = "success"
                    break
                
                self.simulate_once()
                if render:
                    self.render()
                steps += 1
                
            if steps > max_steps:
                if not quiet:
                    print(colored(f"达到最大步数: {max_steps}", "red"))
                    print("误差: ", deltas)
                result = f"达到最大步数: {max_steps}"
                
            return result
            
        except Exception as e:
            print(f"关节目标移动错误: {e}")
            return "error"






    # 只需复制原方法即可，这里为节省空间省略
    
    def start_viewer(self):
        """启动查看器"""
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
    
    def render(self):
        """渲染场景"""
        if self.viewer is None:
            self.start_viewer()
        self.viewer.sync()
    
    def simulate_once(self):
        """执行一次仿真步"""
        mujoco.mj_step(self.model, self.data)
    
    def reset(self):
        """重置仿真"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
    
    def close(self):
        """关闭查看器"""
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def plot_debug_data(self):
        """绘制调试数据（仅在debug=True时可用）"""
        if not self.debug or not self.debug_data:
            print("无调试数据可用")
            return
        
        steps = [d['step'] for d in self.debug_data]
        pos_errors = [d['pos_error'] for d in self.debug_data]
        rot_errors = [d['rot_error'] for d in self.debug_data]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(steps, pos_errors, label='位置误差(m)')
        plt.xlabel('步数')
        plt.ylabel('误差(m)')
        plt.title('末端执行器位置误差')
        plt.grid(True)
        
        if any(rot_errors):
            plt.subplot(1, 2, 2)
            plt.plot(steps, rot_errors, label='旋转误差(rad)', color='orange')
            plt.xlabel('步数')
            plt.ylabel('误差(rad)')
            plt.title('末端执行器旋转误差')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
