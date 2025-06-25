#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础机械臂控制器类，适配 MuJoCo 3.2.3 环境
"""

import os
from pathlib import Path
import numpy as np
from simple_pid import PID
from termcolor import colored
from ikpy.chain import Chain
import cv2 as cv
import matplotlib.pyplot as plt
import copy
import mujoco
from mujoco import mjtObj
import mujoco.viewer
import time
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

class BaseController:
    """
    机械臂基础控制器类，用于控制 MuJoCo 中的 UR5 机械臂及其夹具
    适配 MuJoCo 3.2.3 官方 Python 绑定
    """
    
    def __init__(self, model_path: str = None, data_path: str = None):
        """
        初始化控制器
        
        Args:
            model_path: MuJoCo 模型文件路径
            data_path: 数据路径，用于加载模型和资源
        """
        # 设置模型路径
        self.data_path = data_path if data_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/UR5_gripper")
        self.model_path = model_path if model_path else os.path.join(self.data_path, "UR5gripper_2_finger.xml")
        
        # 确保模型文件存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载模型和数据
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # 初始化机械臂逆运动学链
        urdf_path = os.path.join(self.data_path, "ur5_gripper.urdf")
        if not os.path.exists(urdf_path):
            urdf_path = os.path.join(self.data_path, "ur5_gripper.URDF")  # 尝试大写格式
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")
            
        self.ee_chain = Chain.from_urdf_file(urdf_path)
        
        # 初始化控制器参数
        self.groups = defaultdict(list)
        self.create_lists()
        self.create_default_groups()
        self.reached_target = False
        self.current_output = np.zeros(self.model.nu)
        self.image_counter = 0
        self.cam_matrix = None
        self.cam_init = False
        self.last_movement_steps = 0
        
        # 初始化仿真
        self.simulate_once()  # 执行一次仿真以初始化状态
        
    def create_lists(self):
        """
        创建控制器列表和初始化 PID 控制器
        """
        self.controller_list = []
        sample_time = 0.0001
        p_scale = 3
        i_scale = 0.0
        i_gripper = 0
        d_scale = 0.1
        
        # 为每个关节创建 PID 控制器
        self.controller_list.append(
            PID(7 * p_scale, 0.0 * i_scale, 1.1 * d_scale, setpoint=0, output_limits=(-2, 2), sample_time=sample_time)
        )  # Shoulder Pan Joint
        self.controller_list.append(
            PID(10 * p_scale, 0.0 * i_scale, 1.0 * d_scale, setpoint=-1.57, output_limits=(-2, 2), sample_time=sample_time)
        )  # Shoulder Lift Joint
        self.controller_list.append(
            PID(5 * p_scale, 0.0 * i_scale, 0.5 * d_scale, setpoint=1.57, output_limits=(-2, 2), sample_time=sample_time)
        )  # Elbow Joint
        self.controller_list.append(
            PID(7 * p_scale, 0.0 * i_scale, 0.1 * d_scale, setpoint=-1.57, output_limits=(-1, 1), sample_time=sample_time)
        )  # Wrist 1 Joint
        self.controller_list.append(
            PID(5 * p_scale, 0.0 * i_scale, 0.1 * d_scale, setpoint=-1.57, output_limits=(-1, 1), sample_time=sample_time)
        )  # Wrist 2 Joint
        self.controller_list.append(
            PID(5 * p_scale, 0.0 * i_scale, 0.1 * d_scale, setpoint=0.0, output_limits=(-1, 1), sample_time=sample_time)
        )  # Wrist 3 Joint
        self.controller_list.append(
            PID(2.5 * p_scale, i_gripper, 0.00 * d_scale, setpoint=0.0, output_limits=(-1, 1), sample_time=sample_time)
        )  # Gripper Joint
        
        self.current_target_joint_values = np.array([controller.setpoint for controller in self.controller_list])
        self.actuators = self._create_actuators_list()
        
    def _create_actuators_list(self) -> List[List]:
        """创建执行器列表，包含每个执行器的详细信息"""
        actuators = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            item = [
                i,                          # 执行器 ID
                mujoco.mj_id2name(self.model, mjtObj.mjOBJ_ACTUATOR, i),  # 执行器名称
                joint_id,                   # 控制的关节 ID
                mujoco.mj_id2name(self.model, mjtObj.mjOBJ_JOINT, joint_id),  # 关节名称
                self.controller_list[i]     # PID 控制器
            ]
            actuators.append(item)
        return actuators

    def create_default_groups(self):
        """创建默认的关节组"""
        self.groups["All"] = list(range(self.model.nu))
        self.create_group("Arm", list(range(6)))  # Updated to include all arm joints
        self.create_group("Gripper", [6])
        self.actuated_joint_ids = np.array([actuator[2] for actuator in self.actuators])
    
    def create_group(self, group_name: str, idx_list: List[int]):
        """
        创建关节组
        
        Args:
            group_name: 组名
            idx_list: 关节索引列表
        """
        try:
            assert len(idx_list) <= self.model.nu, "指定的关节数量过多!"
            assert group_name not in self.groups, f"名为 {group_name} 的组已存在!"
            assert max(idx_list) < self.model.nu, "包含无效的执行器 ID"
            
            self.groups[group_name] = idx_list
            print(f"创建新控制组 '{group_name}'.")
            
        except Exception as e:
            print(e)
            print("无法创建新组.")
    
    def show_model_info(self):
        """显示模型信息，包括物体、关节、执行器等"""
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
    
    def actuate_joint_group(self, group: str, motor_values: List[float]):
        """
        驱动指定关节组
        
        Args:
            group: 组名
            motor_values: 关节控制值列表
        """
        try:
            assert group in self.groups, f"不存在名为 {group} 的组!"
            assert len(motor_values) == len(self.groups[group]), "执行器值数量无效!"
            
            for i, v in enumerate(self.groups[group]):
                self.data.ctrl[v] = motor_values[i]
                
        except Exception as e:
            print(e)
            print("无法驱动请求的关节组.")
    
    def move_group_to_joint_target(
        self,
        group: str = "All",
        target: List[float] = None,
        tolerance: float = 0.1,
        max_steps: int = 10000,
        plot: bool = False,
        marker: bool = False,
        render: bool = True,
        quiet: bool = False
    ) -> str:
        """
        将指定关节组移动到目标位置
        
        Args:
            group: 要移动的组名
            target: 关节目标值列表
            tolerance: 误差阈值
            max_steps: 最大步数
            plot: 是否绘制轨迹
            marker: 是否显示标记
            render: 是否渲染
            quiet: 是否静默模式
            
        Returns:
            移动结果描述
        """
        try:
            assert group in self.groups, f"不存在名为 {group} 的组!"
            if target is not None:
                assert len(target) == len(self.groups[group]), f"组 {group} 的目标维度不匹配!"
                
            ids = self.groups[group]
            steps = 1
            result = ""
            if plot:
                self.plot_list = defaultdict(list)
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
                
                if plot and steps % 2 == 0:
                    self.fill_plot_list(group, steps)
                
                if marker:
                    ee_pos = self.data.xpos[mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, "ee_link")]
                    self.add_marker(self.current_carthesian_target)
                    self.add_marker(ee_pos)
                
                if max(deltas) < tolerance:
                    if target is not None and not quiet:
                        print(colored(f"组 {group} 的关节值在要求的容差内! ({steps} 步)", color="green", attrs=["bold"]))
                    result = "success"
                    self.reached_target = True
                
                if steps > max_steps:
                    if not quiet:
                        print(colored(f"达到最大步数: {max_steps}", color="red", attrs=["bold"]))
                        print("误差: ", deltas)
                    result = f"达到最大步数: {max_steps}"
                    break
                
                self.simulate_once()
                if render:
                    self.render()
                steps += 1
                
            self.last_movement_steps = steps
            
            if plot:
                self.create_joint_angle_plot(group, tolerance)
                
            return result
            
        except Exception as e:
            print(e)
            print("无法移动到请求的关节目标.")
            return "error"
    
    def set_group_joint_target(self, group: str, target: List[float]):
        """
        设置关节组的目标值
        
        Args:
            group: 组名
            target: 目标值列表
        """
        idx = self.groups[group]
        try:
            assert len(target) == len(idx), "目标长度必须与组中执行器数量匹配."
            self.current_target_joint_values[idx] = target
            
        except Exception as e:
            print(e)
            print(f"无法为组 {group} 设置新的关节目标")
    
    def open_gripper(self, half: bool = False, **kwargs) -> str:
        """张开夹爪"""
        return self.move_group_to_joint_target(
            group="Gripper", 
            target=[0.0] if half else [0.4], 
            max_steps=1000, 
            tolerance=0.05, 
            **kwargs
        )
    
    def close_gripper(self, **kwargs) -> str:
        """关闭夹爪"""
        return self.move_group_to_joint_target(
            group="Gripper", 
            target=[-0.4], 
            tolerance=0.01, 
            **kwargs
        )
    
    def grasp(self, max_attempts: int = 3, check_threshold: float = 0.03, **kwargs) -> bool:
        """
        尝试抓取物体
        
        Args:
            max_attempts: 最大尝试次数
            check_threshold: 位置检查阈值
            
        Returns:
            抓取是否成功
        """
        for attempt in range(max_attempts):
            result = self.close_gripper(max_steps=500, **kwargs)
            gripper_pos = self.data.qpos[self.actuated_joint_ids[self.groups['Gripper'][0]]]
            
            if abs(gripper_pos - (-0.4)) < check_threshold:
                print(f"✅ 第 {attempt+1} 次抓取尝试：夹爪闭合正常")
                self.stay(500)  # 保持抓取状态
                return True
            else:
                print(f"❌ 第 {attempt+1} 次抓取尝试：夹爪未完全闭合（当前位置: {gripper_pos:.3f}，目标: -0.4）")
                self.open_gripper()
                self.stay(300)
        
        print("❗️ 抓取失败：夹爪未能正确闭合")
        return False
    
    def move_ee(self, ee_position: List[float], approach_speed: float = 0.8, **kwargs) -> str:
        """
        移动末端执行器到指定位置
        
        Args:
            ee_position: 末端执行器目标位置 [x, y, z]
            approach_speed: 接近速度因子
            
        Returns:
            移动结果描述
        """
        x, y, z = ee_position
        if not (-0.7 <= x <= 0.7 and -0.8 <= y <= 0.2 and 0.7 <= z <= 1.2):
            print(f"❌ 目标位置超出工作空间: {ee_position}")
            return "Position out of workspace"
        
        joint_angles = self.ik(ee_position)
        if joint_angles is None:
            return "No valid joint angles received"
        
        original_p = [controller.tunings[0] for controller in self.controller_list[:6]]
        for i in range(6):
            self.controller_list[i].tunings = (original_p[i] * approach_speed, 0, 0.1)
        
        result = self.move_group_to_joint_target(
            group="Arm", 
            target=joint_angles, 
            tolerance=0.03, 
            max_steps=2000, 
            **kwargs
        )
        
        for i in range(6):
            self.controller_list[i].tunings = (original_p[i], 0, 0.1)
        
        final_pos = self.data.xpos[mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, "ee_link")]
        error = np.sqrt(np.sum((final_pos - ee_position)** 2))
        if error > 0.05:
            print(f"⚠️ 位置误差较大: {error:.4f}m")
        
        return result
    
    def ik(self, ee_position: List[float], max_retries: int = 3, tolerance: float = 0.02) -> Optional[List[float]]:
        """
        逆运动学求解
        
        Args:
            ee_position: 末端执行器目标位置
            max_retries: 最大重试次数
            tolerance: 位置误差阈值
            
        Returns:
            关节角度列表，失败时返回 None
        """
        base_pos = self.data.xpos[mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, "base_link")]
        
        for attempt in range(max_retries):
            try:
                ee_position_base = ee_position - base_pos
                gripper_center = ee_position_base + np.array([0, -0.005, 0.16])
                
                current_joints = self.data.qpos[self.actuated_joint_ids][:6]
                initial_guess = [0, *current_joints, 0]
                
                joint_angles = self.ee_chain.inverse_kinematics(
                    gripper_center, 
                    [0, 0, -1], 
                    orientation_mode='X',
                    initial_position=initial_guess,
                    max_iter=100
                )
                
                prediction = self.ee_chain.forward_kinematics(joint_angles)[:3, 3]
                prediction_world = prediction + base_pos - np.array([0, -0.005, 0.16])
                error = np.sqrt(np.sum((prediction_world - ee_position)** 2))
                
                if error <= tolerance:
                    return joint_angles[1:-2].tolist()
                
                print(f"尝试 {attempt+1}/{max_retries}: 解的误差 {error:.4f}m，超过阈值 {tolerance}m")
                
                if attempt < max_retries - 1:
                    ee_position[2] += 0.01 * (attempt + 1) * (-1 if attempt % 2 == 0 else 1)
                    print(f"尝试 {attempt+2}/{max_retries}: 微调目标位置为 {ee_position}")
                    
            except Exception as e:
                print(f"尝试 {attempt+1}/{max_retries}: 求解失败 - {str(e)}")
        
        print(f"❌ IK求解失败: 尝试 {max_retries} 次后仍无法找到有效解")
        return None
    
    def display_current_values(self):
        """显示当前机械臂状态"""
        print("\n################################################")
        print("当前关节位置 (驱动关节)")
        print("################################################")
        for i in range(len(self.actuated_joint_ids)):
            print(f"关节 {self.actuators[i][3]} 的当前角度: {self.data.qpos[self.actuated_joint_ids[i]]}")
        
        print("\n################################################")
        print("当前所有关节位置")
        print("################################################")
        for i in range(self.model.njnt):
            print(f"关节 {mujoco.mj_id2name(self.model, mjtObj.mjOBJ_JOINT, i)} 的当前角度: {self.data.qpos[i]}")
        
        print("\n################################################")
        print("当前物体位置")
        print("################################################")
        for i in range(self.model.nbody):
            print(f"物体 {mujoco.mj_id2name(self.model, mjtObj.mjOBJ_BODY, i)} 的当前位置: {self.data.xpos[i]}")
        
        print("\n################################################")
        print("当前物体旋转矩阵")
        print("################################################")
        for i in range(self.model.nbody):
            print(f"物体 {mujoco.mj_id2name(self.model, mjtObj.mjOBJ_BODY, i)} 的当前旋转: {self.data.xmat[i].reshape(3, 3)}")
        
        print("\n################################################")
        print("当前执行器控制值")
        print("################################################")
        for i in range(self.model.nu):
            print(f"执行器 {self.actuators[i][1]} 的当前激活值: {self.data.ctrl[i]}")
    
    def stay(self, duration: int, render: bool = True):
        """
        保持当前位置
        
        Args:
            duration: 保持时间(毫秒)
            render: 是否渲染
        """
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            self.move_group_to_joint_target(max_steps=10, tolerance=0.0000001, plot=False, quiet=True, render=render)
            elapsed = (time.time() - starting_time) * 1000
    
    def fill_plot_list(self, group: str, step: int):
        """
        填充绘图数据列表
        
        Args:
            group: 组名
            step: 步骤
        """
        for i in self.groups[group]:
            self.plot_list[self.actuators[i][3]].append(self.data.qpos[self.actuated_joint_ids[i]])
        self.plot_list["Steps"].append(step)
    
    def create_joint_angle_plot(self, group: str, tolerance: float):
        """
        创建关节角度图
        
        Args:
            group: 组名
            tolerance: 容差
        """
        self.image_counter += 1
        keys = list(self.plot_list.keys())
        number_subplots = len(self.plot_list) - 1
        columns = 3
        rows = (number_subplots // columns) + (number_subplots % columns)
        
        position = range(1, number_subplots + 1)
        fig = plt.figure(1, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.4, left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        for i in range(number_subplots):
            axis = fig.add_subplot(rows, columns, position[i])
            axis.plot(self.plot_list["Steps"], self.plot_list[keys[i]])
            axis.set_title(keys[i])
            axis.set_xlabel(keys[-1])
            axis.set_ylabel("关节角度 [rad]")
            axis.xaxis.set_label_coords(0.05, -0.1)
            axis.yaxis.set_label_coords(1.05, 0.5)
            axis.axhline(self.current_target_joint_values[self.groups[group][i]], color="g", linestyle="--")
            axis.axhline(self.current_target_joint_values[self.groups[group][i]] + tolerance, color="r", linestyle="--")
            axis.axhline(self.current_target_joint_values[self.groups[group][i]] - tolerance, color="r", linestyle="--")
        
        filename = f"Joint_values_{self.image_counter}.png"
        plt.savefig(filename)
        print(colored(f"轨迹已保存到 {filename}.", color="yellow", on_color="on_grey", attrs=["bold"]))
        plt.clf()
    
    def get_image_data(self, show: bool = False, camera: str = "top_down", width: int = 200, height: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取相机图像数据
        
        Args:
            show: 是否显示图像
            camera: 相机名称
            width: 图像宽度
            height: 图像高度
            
        Returns:
            (RGB图像, 深度图像)
        """
        rgb, depth = mujoco.mjr_renderOffscreen(
            width, height, self.model, self.data,
            camera_id=mujoco.mj_name2id(self.model, mjtObj.mjOBJ_CAMERA, camera),
            depth=True
        )
        
        rgb = np.fliplr(np.flipud(rgb))
        depth = np.fliplr(np.flipud(depth))
        
        if show:
            cv.imshow("RGB", cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
            cv.waitKey(1)
        
        return rgb, depth
    
    def depth_2_meters(self, depth: np.ndarray) -> np.ndarray:
        """
        将深度图像转换为实际距离(米)
        
        Args:
            depth: 深度图像(0-1范围)
            
        Returns:
            实际距离图像(米)
        """
        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend
        return near / (1 - depth * (1 - near / far))
    
    def create_camera_data(self, width: int, height: int, camera: str):
        """
        初始化相机参数
        
        Args:
            width: 图像宽度
            height: 图像高度
            camera: 相机名称
        """
        cam_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_CAMERA, camera)
        fovy = self.model.cam_fovy[cam_id]
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        self.cam_rot_mat = self.model.cam_mat0[cam_id].reshape(3, 3)
        self.cam_pos = self.model.cam_pos0[cam_id]
        self.cam_init = True
    
    def world_2_pixel(self, world_coordinate: List[float], width: int = 200, height: int = 200, camera: str = "top_down") -> Tuple[int, int]:
        """
        将世界坐标转换为像素坐标
        
        Args:
            world_coordinate: 世界坐标
            width: 图像宽度
            height: 图像高度
            camera: 相机名称
            
        Returns:
            (像素x, 像素y)
        """
        if not self.cam_init:
            self.create_camera_data(width, height, camera)
        
        hom_pixel = self.cam_matrix @ self.cam_rot_mat @ (np.array(world_coordinate) - self.cam_pos)
        pixel = hom_pixel[:2] / hom_pixel[2]
        
        return np.round(pixel[0]).astype(int), np.round(pixel[1]).astype(int)
    
    def pixel_2_world(self, pixel_x: int, pixel_y: int, depth: float, width: int = 200, height: int = 200, camera: str = "top_down") -> np.ndarray:
        """
        将像素坐标转换为世界坐标
        
        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标
            depth: 像素深度值
            width: 图像宽度
            height: 图像高度
            camera: 相机名称
            
        Returns:
            世界坐标
        """
        if not self.cam_init:
            self.create_camera_data(width, height, camera)
        
        pixel_coord = np.array([pixel_x, pixel_y, 1]) * (-depth)
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        pos_w = np.linalg.inv(self.cam_rot_mat) @ (pos_c + self.cam_pos)
        
        return pos_w
    
    def add_marker(self, coordinates: List[float], label: bool = True, size: List[float] = None, color: List[float] = None):
        """
        在指定坐标添加标记
        
        Args:
            coordinates: 标记坐标
            label: 是否显示标签
            size: 标记大小
            color: 标记颜色
        """
        if self.viewer is None:
            self.start_viewer()
        
        if size is None:
            size = [0.015, 0.015, 0.015]
        if color is None:
            color = [1, 0, 0]
        
        label_str = str(coordinates) if label else ""
        rgba = np.concatenate((color, np.ones(1)))
        self.viewer.add_marker(pos=coordinates, label=label_str, size=size, rgba=rgba, type=mujoco.mjtGeom.mjGEOM_SPHERE)
    
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
    
    @property
    def current_carthesian_target(self) -> np.ndarray:
        """获取当前笛卡尔空间目标"""
        return self.data.xpos[mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, "ee_link")]
    
    @property
    def last_steps(self) -> int:
        """获取上次移动的步数"""
        return self.last_movement_steps
    
    def close(self):
        """关闭查看器"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


            