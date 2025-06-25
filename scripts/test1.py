import os
import sys
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.controllers.base_controller import BaseController

def main():
    controller = None
    try:
        # Initialize controller
        controller = BaseController()
        # Display model information
        controller.show_model_info()
        
        # Get current joint positions for the Arm group (6 joints)
        current_joints = controller.data.qpos[controller.actuated_joint_ids[:6]].copy()
        print(f"Initial joint positions: {current_joints}")
        
        # Rotate left: set shoulder_pan_joint (index 0) to +1.0 rad
        print("Rotating arm left (shoulder_pan_joint to +1.0 rad)")
        target_joints = current_joints.copy()
        target_joints[0] = 1.0  # shoulder_pan_joint
        result = controller.move_group_to_joint_target(
            group="Arm",
            target=target_joints.tolist(),
            tolerance=0.05,
            max_steps=2000,
            render=True
        )
        print(f"Left rotation result: {result}")
        controller.stay(2000)  # Pause for 2 seconds
        
        # Rotate right: set shoulder_pan_joint to -1.0 rad
        print("Rotating arm right (shoulder_pan_joint to -1.0 rad)")
        target_joints[0] = -1.0
        result = controller.move_group_to_joint_target(
            group="Arm",
            target=target_joints.tolist(),
            tolerance=0.05,
            max_steps=2000,
            render=True
        )
        print(f"Right rotation result: {result}")
        controller.stay(20000)  # Pause for 2 seconds
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close the viewer
        if controller is not None:
            controller.close()

if __name__ == "__main__":
    main()

    