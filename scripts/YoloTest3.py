#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import gym
import mujoco
import numpy as np
import torch
from ultralytics import YOLO
from mujoco import mjtCamera
import graspproduct  # Import GraspEnv from graspproduct.py
from src.controllers.second_controller import UR5Controller

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

CAMERA_NAME = "top_down"
TABLE_HEIGHT = 0.88  # Table surface at z=0.88m
DEPTH_THRESHOLD = 1.0  # For box_1 at ~0.97m
HEIGHT = 200
WIDTH = 200

class YoloGraspAgent:
    def __init__(self, height=HEIGHT, width=WIDTH):
        # Initialize environment
        self.env = graspproduct.GraspEnv(image_height=height, image_width=width, render=True)
        # Load pretrained YOLOv8-nano model
        #self.yolo_model = YOLO("yolov8n.pt")  # Pretrained model
        self.yolo_model = YOLO(os.path.join(project_root, "scripts", "yolov8n.pt"))
        # self.transform = torch.nn.Sequential(
        #     torch.nn.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # )

    def detect_objects(self, rgb):
        """Run YOLO detection on RGB image."""
        # Convert RGB to YOLO input format (BGR, 0-255)
        rgb_bgr = rgb[:, :, ::-1].copy()
        results = self.yolo_model.predict(rgb_bgr, conf=0.7, classes=[0])  # Class 0: general "box"
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x_min, y_min, x_max, y_max]
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confs, classes):
                detections.append({"box": box, "conf": conf, "class": int(cls)})
        return detections

    def select_grasp_point(self, state):
        """Select grasp point for box_1 using YOLO."""
        rgb = state["rgb"]
        depth = state["depth"]
        # Detect objects
        detections = self.detect_objects(rgb)
        if not detections:
            print("No box detected.")
            return None
        # Select highest-confidence box (assuming class 0 is "box")
        box_detection = max(detections, key=lambda x: x["conf"])
        box = box_detection["box"]
        print(f"Box detected: {box}, confidence: {box_detection['conf']}")
        # Compute bounding box center
        u = (box[0] + box[2]) / 2  # x_center
        v = (box[1] + box[3]) / 2  # y_center
        u, v = int(u), int(v)
        # Get depth at center
        depth_value = depth[v, u]
        if not (TABLE_HEIGHT < depth_value < DEPTH_THRESHOLD):
            print(f"Invalid depth at ({u}, {v}): {depth_value:.3f}")
            # Fallback: Average depth in bounding box
            x_min, y_min, x_max, y_max = map(int, box)
            depth_patch = depth[y_min:y_max, x_min:x_max]
            valid_depth = depth_patch[(TABLE_HEIGHT < depth_patch) & (depth_patch < DEPTH_THRESHOLD)]
            if valid_depth.size == 0:
                print("No valid depth in bounding box.")
                return None
            depth_value = np.mean(valid_depth)
        # Convert to world coordinates
        world_point = self.env.pixel_2_world((u, v), depth_value)
        print(f"Grasp point: pixel=({u}, {v}), depth={depth_value:.3f}, world={world_point}")
        return world_point

    def execute_grasp(self):
        """Perform single grasp of box_1."""
        try:
            # Reset environment
            state = self.env.reset()
            print("Environment reset complete.")
            # Select grasp point
            grasp_point = self.select_grasp_point(state)
            if grasp_point is None:
                print("Grasp failed: No valid grasp point.")
                return False
            # Stage 1: Move above grasp point
            approach_pos = grasp_point.copy()
            approach_pos[2] += 0.5
            print("Moving above grasp point")
            result = self.env.controller.move_ee(
                ee_position=approach_pos, tolerance=0.3, max_steps=2000, render=True
            )
            if result != "success":
                print(f"Approach failed: {result}")
                return False
            # Stage 2: Move to pregrasp
            pregrasp_pos = grasp_point.copy()
            pregrasp_pos[2] += 0.3
            print("Moving to pregrasp position")
            result = self.env.controller.move_ee(
                ee_position=pregrasp_pos, tolerance=0.3, max_steps=1500, render=True
            )
            if result != "success":
                print(f"Pregrasp failed: {result}")
                return False
            # Stage 3: Move to grasp
            print("Moving to grasp position")
            result = self.env.controller.move_ee(
                ee_position=grasp_point, tolerance=0.2, max_steps=1000, render=True
            )
            if result != "success":
                print(f"Grasp move failed: {result}")
                return False
            # Stage 4: Close gripper
            print("Closing gripper")
            result = self.env.controller.close_gripper(render=True)
            if result != "success":
                print(f"Gripper close failed: {result}")
                return False
            # Stage 5: Lift
            lift_pos = grasp_point.copy()
            lift_pos[2] += 0.1
            print("Lifting object")
            result = self.env.controller.move_ee(
                ee_position=lift_pos, tolerance=0.3, max_steps=1500, render=True
            )
            if result != "success":
                print(f"Lift failed: {result}")
                return False
            return True
        except Exception as e:
            print(f"Grasp execution error: {e}")
            return False

def main():
    agent = YoloGraspAgent()
    try:
        
        success = agent.execute_grasp()
        print(f"Grasp {'succeeded' if success else 'failed'}.")
        print("测试完成，MuJoCo GUI 保持开启，请手动关闭。")
        while agent.env.viewer.is_running():
            agent.env.viewer.sync()
            time.sleep(1/60)
    except Exception as e:
        print(f"Main error: {e}")
    finally:
        if hasattr(agent.env, "viewer") and agent.env.viewer is not None:
            agent.env.viewer.close()
        print("程序结束，请手动关闭窗口")

if __name__ == "__main__":
    main()