import os
import sys
import mujoco  # Added to fix 'name mujoco is not defined'
import time  # Added for sleep in GUI loop
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.controllers.second_controller import UR5Controller

def main():
    controller = None
    try:
        # Initialize controller
        controller = UR5Controller()
        # Display model information
        controller.show_model_info()
        
        # Set home position to lift arm off table
        print("\n=== 设置初始位置 ===")
        current_joints = controller.data.qpos[controller.actuated_joint_ids[:6]].copy()
        print(f"初始关节位置: {current_joints}")
        home_joints = current_joints.copy()
        home_joints[1] = -1.57  # shoulder_lift_joint
        home_joints[2] = 1.57   # elbow_joint
        print("移动到初始位置")
        result = controller.move_group_to_joint_target(
            group="Arm",
            target=home_joints.tolist(),
            tolerance=0.05,
            max_steps=2000,
            render=True
        )
        print(f"初始位置结果: {result}")
        controller.stay(2000)
        
        # Test 1: Joint-space control (rotate shoulder_pan_joint)
        print("\n=== 测试1: 关节空间控制 ===")
        current_joints = controller.data.qpos[controller.actuated_joint_ids[:6]].copy()
        print(f"当前关节位置: {current_joints}")
        
        # Rotate left
        print("向左旋转 (shoulder_pan_joint to +1.0 rad)")
        target_joints = current_joints.copy()
        target_joints[0] = 1.0
        result = controller.move_group_to_joint_target(
            group="Arm",
            target=target_joints.tolist(),
            tolerance=0.05,
            max_steps=2000,
            render=True
        )
        print(f"左旋转结果: {result}")
        controller.stay(2000)
        
        # Rotate right
        print("向右旋转 (shoulder_pan_joint to -1.0 rad)")
        target_joints[0] = -1.0
        result = controller.move_group_to_joint_target(
            group="Arm",
            target=target_joints.tolist(),
            tolerance=0.05,
            max_steps=2000,
            render=True
        )
        print(f"右旋转结果: {result}")
        controller.stay(2000)
        
        # Test 2: Cartesian control (Jacobian IK)
        print("\n=== 测试2: 笛卡尔空间控制 (雅可比IK) ===")
        ee_id = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        initial_pos = controller.data.xpos[ee_id].copy()
        print(f"初始末端执行器位置: {initial_pos}")
        
        # Small Cartesian movement
        target_pos = initial_pos + [0.02, 0.0, 0.0]  # Move 2cm along x-axis
        print(f"尝试移动到目标位置: {target_pos}")
        result = controller.move_ee(
            ee_position=target_pos.tolist(),
            tolerance=0.05,
            max_steps=2000,
            render=True
        )
        print(f"笛卡尔移动结果: {result}")
        controller.stay(5000)
        
        # Test 3: Gripper control
        print("\n=== 测试3: 夹爪控制 ===")
        print("张开夹爪")
        result = controller.open_gripper(render=True)
        print(f"张开夹爪结果: {result}")
        controller.stay(2000)
        
        print("关闭夹爪")
        result = controller.close_gripper(render=True)
        print(f"关闭夹爪结果: {result}")
        controller.stay(2000)
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        print("测试完成，MuJoCo GUI 保持开启，请手动关闭。")
        while True:
            controller.render()  # Keep rendering to maintain GUI
            time.sleep(0.01)    # Prevent high CPU usage


if __name__ == "__main__":
    main()