#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import mujoco
from mujoco import viewer
import imageio

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.controllers.second_controller import UR5Controller

class GraspEnv:
    """Simplified grasping environment for UR5 with MuJoCo"""
    def __init__(self, model_path=None, render=True):
        if model_path is None:
            model_path = os.path.join(project_root, "assets", "UR5_gripper", "UR5gripper_2_finger_many_objects.xml")
        
        self.controller = UR5Controller(model_path=model_path)
        self.model = self.controller.model
        self.data = self.controller.data
        self.render = render
        
        # Camera setup
        self.camera_name = "top_down"
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        self.viewer = None
        
        if self.render:
            self.init_viewer()
    
    def init_viewer(self):
        """Initialize the MuJoCo viewer"""
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.cam.fixedcamid = self.cam_id
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    
    def reset(self):
        """Reset the environment to home position"""
        self.controller.reset()
        mujoco.mj_forward(self.model, self.data)
        
        # Move to home position
        home_joints = self.data.qpos[self.controller.actuated_joint_ids[:6]].copy()
        home_joints[1] = -1.57  # shoulder_lift_joint
        home_joints[2] = 1.57   # elbow_joint
        
        result = self.controller.move_group_to_joint_target(
            group="Arm",
            target=home_joints.tolist(),
            tolerance=0.05,
            max_steps=2000,
            render=self.render
        )
        
        self.controller.open_gripper(render=self.render)
        return self.get_observation()
    
    def get_observation(self):
        """Capture RGB and depth images from the camera"""
        if self.viewer is None and self.render:
            self.init_viewer()
        
        # Update the viewer if rendering
        if self.render:
            self.viewer.sync()
            time.sleep(0.01)
        
        # Get camera parameters
        cam_pos = self.data.cam_xpos[self.cam_id].copy()
        cam_mat = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        fovy = self.model.cam_fovy[self.cam_id]
        
        # For MuJoCo 3.x, we'll use a simpler projection method
        # Calculate projection matrix
        f = 0.5 * self.model.vis.global_.fovy * np.pi / 180.0
        focal_length = 1.0 / np.tan(f)
        
        return {
            'cam_pos': cam_pos,
            'cam_mat': cam_mat,
            'focal_length': focal_length,
            'fovy': fovy
        }
    
    def pixel_to_world(self, pixel_x, pixel_y, depth, observation):
        """Convert pixel coordinates to world coordinates"""
        # Normalize pixel coordinates
        x = (2.0 * pixel_x / 640) - 1.0
        y = 1.0 - (2.0 * pixel_y / 480)
        
        # Get camera parameters
        cam_pos = observation['cam_pos']
        cam_mat = observation['cam_mat']
        focal_length = observation['focal_length']
        
        # Calculate ray direction in camera frame
        ray_cam = np.array([
            x / focal_length,
            y / focal_length,
            -1.0  # Camera looks along negative z-axis
        ])
        
        # Transform ray to world frame
        ray_world = cam_mat @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        
        # Calculate world position
        world_pos = cam_pos + ray_world * depth
        return world_pos
    
    def execute_grasp(self, grasp_pos, lift_height=0.1):
        """Execute a grasp at the specified position"""
        # Approach position (5cm above)
        approach_pos = grasp_pos.copy()
        approach_pos[2] += 0.05
        
        # Move to approach position
        result = self.controller.move_ee(
            ee_position=approach_pos,
            tolerance=0.005,
            max_steps=2000,
            render=self.render
        )
        if result != "success":
            return False
        
        time.sleep(0.5)
        
        # Move to grasp position
        result = self.controller.move_ee(
            ee_position=grasp_pos,
            tolerance=0.005,
            max_steps=2000,
            render=self.render
        )
        if result != "success":
            return False
        
        time.sleep(0.5)
        
        # Close gripper
        self.controller.close_gripper(render=self.render)
        time.sleep(0.5)
        
        # Lift object
        lift_pos = grasp_pos.copy()
        lift_pos[2] += lift_height
        
        result = self.controller.move_ee(
            ee_position=lift_pos,
            tolerance=0.005,
            max_steps=2000,
            render=self.render
        )
        
        return result == "success"
    
    def close(self):
        """Close the environment"""
        if self.viewer is not None:
            self.viewer.close()

class GraspAgent:
    """Agent for performing grasping tasks"""
    def __init__(self, env):
        self.env = env
        self.last_observation = None
    
    def find_grasp_point(self, observation):
        """Simple heuristic to find grasp points"""
        # For this simplified version, we'll just pick a point in front of the robot
        grasp_pos = observation['cam_pos'].copy()
        grasp_pos[0] += 0.2  # 20cm in front of camera
        grasp_pos[1] += 0.0  # Centered
        grasp_pos[2] -= 0.3  # 30cm below camera (adjust based on your setup)
        
        return grasp_pos, (320, 240)  # Center pixel coordinates
    
    def run(self):
        """Main execution loop"""
        try:
            self.last_observation = self.env.reset()
            
            # Find grasp point
            grasp_pos, (px, py) = self.find_grasp_point(self.last_observation)
            print(f"Attempting grasp at world position: {grasp_pos}")
            
            # Execute grasp
            success = self.env.execute_grasp(grasp_pos)
            print(f"Grasp {'successful' if success else 'failed'}")
            
            # Keep viewer open if rendering
            while self.env.render and self.env.viewer.is_running:
                self.env.viewer.sync()
                time.sleep(1/60)
                
        except Exception as e:
            print(f"Error during execution: {str(e)}")
        finally:
            self.env.close()

if __name__ == "__main__":
    # Initialize environment
    env = GraspEnv(render=True)
    
    # Create and run agent
    agent = GraspAgent(env)
    agent.run()