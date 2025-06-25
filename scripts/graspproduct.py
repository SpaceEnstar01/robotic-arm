
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import gym
import mujoco
import numpy as np
import torch
import torchvision.transforms as T
from mujoco import mjtObj, mjtCamera, viewer

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.controllers.second_controller import UR5Controller

CAMERA_NAME = "top_down"
TABLE_HEIGHT = 0.88
DEPTH_THRESHOLD = 1.0
HEIGHT = 200
WIDTH = 200

class GraspEnv(gym.Env):
    def __init__(self, image_height=HEIGHT, image_width=WIDTH, render=True):
        super(GraspEnv, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.render = render
        # Initialize controller
        model_path = os.path.join(project_root, "assets", "UR5_gripper", "UR5gripper_2_finger.xml")
        self.controller = UR5Controller(model_path=model_path)
        self.model = self.controller.model
        self.data = self.controller.data
        # Camera parameters
        self.cam_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        self.cam_pos = self.model.cam_pos[self.cam_id]
        cam_quat = np.array(self.model.cam_quat[self.cam_id], dtype=np.float64).reshape(4, 1)
        cam_mat_flat = np.zeros((9, 1), dtype=np.float64)
        mujoco.mju_quat2Mat(cam_mat_flat, cam_quat)
        self.cam_mat = cam_mat_flat.reshape(3, 3)
        self.cam_fovy = self.model.cam_fovy[self.cam_id]
        # Initialize viewer
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.fixedcamid = self.cam_id
            self.viewer.cam.type = mjtCamera.mjCAMERA_FIXED

    def reset(self):
        """Reset environment to home position."""
        self.controller.reset()
        mujoco.mj_forward(self.model, self.data)
        if self.render:
            self.viewer.sync()
        # Set home position
        current_joints = self.data.qpos[self.controller.actuated_joint_ids[:6]].copy()
        home_joints = current_joints.copy()
        home_joints[1] = -1.57  # shoulder_lift_joint
        home_joints[2] = 1.57   # elbow_joint
        result = self.controller.move_group_to_joint_target(
            group="Arm", target=home_joints.tolist(), tolerance=0.05, max_steps=2000, render=self.render
        )
        print(f"Home position result: {result}")
        self.controller.open_gripper(render=self.render)
        mujoco.mj_step(self.model, self.data)
        return self.get_observation()

    def get_observation(self):
        """Render RGB-D image from top_down camera."""
        if self.render and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.fixedcamid = self.cam_id
            self.viewer.cam.type = mjtCamera.mjCAMERA_FIXED
        # Render RGB and depth
        rgb = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        depth = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        self.viewer.render(width=self.image_width, height=self.image_height, camera_id=self.cam_id, rgb=rgb, depth=depth)
        rgb = np.flipud(rgb)
        depth = np.flipud(depth)
        # Convert depth to meters
        znear, zfar = self.model.vis.map.znear, self.model.vis.map.zfar
        depth = znear * zfar / (zfar - depth * (zfar - znear))
        print(f"Depth range in get_observation: {depth.min():.3f}, {depth.max():.3f}")
        return {"rgb": rgb, "depth": depth}

    def pixel_2_world(self, pixel, depth):
        """Convert pixel coordinates to world coordinates."""
        u, v = pixel
        # Normalize pixel coordinates
        fx = 1.0 / np.tan(self.cam_fovy * np.pi / 360.0) * self.image_width / 2.0
        fy = fx
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        # Camera to world coordinates
        camera_point = np.array([x, y, depth])
        world_point = self.cam_pos + self.cam_mat @ camera_point
        return world_point

class GraspAgent:
    def __init__(self, height=HEIGHT, width=WIDTH):
        self.env = GraspEnv(image_height=height, image_width=width, render=True)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5]*4, std=[0.5]*4)
        ])

    def transform_observation(self, observation):
        """Transform observation to network input."""
        rgb = observation["rgb"].astype(np.float32) / 255.0
        depth = observation["depth"]
        depth = np.clip(depth, 0, DEPTH_THRESHOLD)
        depth = depth / DEPTH_THRESHOLD
        input_array = np.concatenate([rgb, depth[..., np.newaxis]], axis=-1)
        input_tensor = self.transform(input_array).to(dtype=torch.float32)
        return input_tensor.unsqueeze(0)

    def select_grasp_point(self, state):
        """Select grasp point from depth image."""
        depth = state["depth"]
        valid_depth = (depth > TABLE_HEIGHT) & (depth < DEPTH_THRESHOLD)
        if not np.any(valid_depth):
            print("No valid grasp point found.")
            return None
        valid_indices = np.where(valid_depth)
        idx = np.random.choice(len(valid_indices[0]))
        v, u = valid_indices[0][idx], valid_indices[1][idx]
        depth_value = depth[v, u]
        world_point = self.env.pixel_2_world((u, v), depth_value)
        print(f"Grasp point: pixel=({u}, {v}), world={world_point}")
        return world_point

    def execute_grasp(self):
        """Perform grasping sequence."""
        state = self.env.reset()
        observation = self.transform_observation(state)
        print("Observation shape:", observation.shape)
        print("Depth range:", state["depth"].min(), state["depth"].max())
        grasp_point = self.select_grasp_point(state)
        if grasp_point is None:
            return False
        # Stage 1: Move above grasp point
        approach_pos = grasp_point.copy()
        approach_pos[2] += 0.5
        print("Moving above grasp point")
        result = self.env.controller.move_ee(
            ee_position=approach_pos, tolerance=0.3, max_steps=2000, render=True
        )
        if result != "success":
            return False
        # Stage 2: Move to pregrasp
        pregrasp_pos = grasp_point.copy()
        pregrasp_pos[2] += 0.3
        print("Moving to pregrasp position")
        result = self.env.controller.move_ee(
            ee_position=pregrasp_pos, tolerance=0.3, max_steps=1500, render=True
        )
        if result != "success":
            return False
        # Stage 3: Move to grasp
        print("Moving to grasp position")
        result = self.env.controller.move_ee(
            ee_position=grasp_point, tolerance=0.2, max_steps=1000, render=True
        )
        if result != "success":
            return False
        # Stage 4: Close gripper
        print("Closing gripper")
        self.env.controller.close_gripper(render=True)
        time.sleep(0.5)
        # Stage 5: Lift
        lift_pos = grasp_point.copy()
        lift_pos[2] += 0.1
        print("Lifting object")
        result = self.env.controller.move_ee(
            ee_position=lift_pos, tolerance=0.3, max_steps=1500, render=True
        )
        return result == "success"

def main():
    agent = GraspAgent()
    try:
        success = agent.execute_grasp()
        print(f"Grasp {'succeeded' if success else 'failed'}.")
        print("测试完成，MuJoCo GUI 保持开启，请手动关闭。")
        while agent.env.viewer.is_running():
            agent.env.viewer.sync()
            time.sleep(1/60)
    except Exception as e:
        print(f"错误: {e}")
        print("测试完成，MuJoCo GUI 保持开启，请手动关闭。")
        while agent.env.viewer.is_running():
            agent.env.viewer.sync()
            time.sleep(1/60)
    finally:
        if hasattr(agent.env, "viewer") and agent.env.viewer is not None:
            agent.env.viewer.close()

if __name__ == "__main__":
    main()
    print("程序结束，请手动关闭窗口")

