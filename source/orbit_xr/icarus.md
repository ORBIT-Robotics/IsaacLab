# ICARUS Teleoperation Integration for IsaacLab

This document explains, step by step, how the ICARUS robot is wrapped for teleoperation in IsaacLab using Meta Quest hand tracking, OpenXR retargeters, and a Differential Inverse Kinematics (IK) controller. It covers the data flow, configuration, code structure, and practical integration details.

---

## Overview

**Goal:**  
Enable real-time teleoperation of the ICARUS robot arm and dexterous ORCA hand using Meta Quest hand tracking, leveraging IsaacLab’s OpenXR device, retargeters, and built-in IK controllers.

**Key Components:**
- **Meta Quest + OpenXR:** Provides hand keypoints in a standardized format.
- **OpenXR Retargeters:** Map XR hand data to robot-friendly commands (end-effector pose, hand joints).
- **ICARUS Robot Asset:** The simulated robot (arm + hand) in IsaacLab.
- **Differential IK Controller:** Converts desired end-effector pose to arm joint targets.
- **TeleopEnv:** Orchestrates device polling, command splitting, and action assembly.
- **Main Control Loop:** Runs the teleop pipeline in IsaacLab’s simulation context.

---

## Data Flow

1. **Hand Tracking Input:**  
   Meta Quest streams hand keypoints via OpenXR.

2. **Retargeting:**  
   - **Se3AbsRetargeter:** Converts XR hand pose to a 7D end-effector (EE) pose for the robot arm.
   - **OrcaRetargeter:** Maps XR hand joints to ORCA hand DOFs (joint angles).

3. **TeleopEnv Splitting:**  
   The retargeted tensor is split into:
   - `ee_pose`: 7D target for arm IK ([x, y, z, qw, qx, qy, qz])
   - `right_wrist_pose`: 7D (diagnostic)
   - `hand_joints`: K-dimensional vector for the ORCA hand

4. **IK Solving:**  
   The Differential IK controller computes the next arm joint targets to reach `ee_pose`.

5. **Action Assembly:**  
   The arm and hand targets are mapped into the robot’s full DOF vector and sent to the simulation.

---

## File Structure

```
source/
  orbit_xr/
    tasks/
      teleop_env_cfg.py      # Config dataclasses for robot, device, IK
      teleop_env.py          # TeleopEnv: device polling, command splitting, action assembly
      teleop_ik.py           # TeleopIK: Differential IK wrapper
      run_teleop.py          # Main script: wiring, loop, and simulation control
    devices/
      retargeter.py          # Device/retargeter factory (uses IsaacLab retargeters)
isaaclab_assets/
  data/robot/icarus/unimanual/
    urdf/icarus.urdf         # ICARUS robot definition (DOF names, kinematics)
    usd/icarus.usd           # ICARUS robot USD asset
```

---

## Configuration Details

### TeleopEnvCfg (teleop_env_cfg.py)

- **robot.arm_dof_names:**  
  List of ICARUS arm joint names (from URDF, e.g., `"Revolute 1", ..., "right_wrist"`).
- **robot.hand_dof_names:**  
  List of ORCA hand DOFs (right hand only for uni-hand setup).
- **robot.ee_link_name:**  
  Name of the robot’s end-effector link (e.g., `"right_hand_joint"`).
- **robot.base_link_name:**  
  Name of the robot’s base link (e.g., `"base_link"`).
- **device:**  
  Configures OpenXR device and retargeters (Se3AbsRetargeter for EE, OrcaRetargeter for hand).
- **ik:**  
  Damping and step limits for the Differential IK controller.
- **sim_device:**  
  Device string for torch computations (e.g., `"cuda:0"`).

### Example DOF Names

**ICARUS Arm:**
```python
["Revolute 1", "Revolute 3", "Revolute 9", "Revolute 20", "Revolute 26", "right_wrist"]
```

**ORCA Hand (right):**
```python
[
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd", "right_index_mcp", "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd", "right_ring_mcp", "right_ring_pip",
    "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip"
]
```

---

## TeleopEnv: Device Polling and Action Assembly

- **advance():**  
  Polls the OpenXR device, splits the retargeted tensor into `ee_pose`, `right_wrist_pose`, and `hand_joints`.
- **bind_robot_dofs():**  
  Maps arm/hand DOF names to indices in the robot’s DOF vector.
- **build_action():**  
  Assembles a full action vector from arm joint targets and hand joint commands.

---

## Differential IK Controller (teleop_ik.py)

- **TeleopIKConfig:**  
  Holds device, EE link, base link, arm DOF names, and damping.
- **TeleopIK:**  
  - Resolves EE body index and arm joint indices in the articulation.
  - At each step:
    - Reads current EE pose and robot base pose.
    - Computes EE pose in base frame.
    - Extracts and rotates the Jacobian into the base frame.
    - Reads current arm joint positions.
    - Sets the XR target as the IK command.
    - Calls the DifferentialIKController to compute next arm joint targets.

---

## Main Control Loop (run_teleop.py)

1. **App and Simulation Setup:**  
   Launches IsaacLab’s AppLauncher and SimulationContext.

2. **Robot Loading:**  
   Loads the ICARUS robot asset (USD) and creates an Articulation.

3. **TeleopEnv and IK Setup:**  
   - Instantiates TeleopEnv with config.
   - Binds robot DOF names.
   - Instantiates TeleopIK with the articulation and config.

4. **Teleop Loop:**  
   - Each frame:
     - Polls the XR device for the latest command.
     - Runs IK to get arm joint targets.
     - Builds the full action vector (arm + hand).
     - Converts action to torch tensor and sends to robot.
     - Steps the simulation.

---

## OpenXR Retargeters: How They Fit

- **Se3AbsRetargeter:**  
  Outputs a 7D pose for the robot’s end-effector (used as IK target).
- **OrcaRetargeter:**  
  Outputs right-wrist pose and right-hand joint angles (mapped directly to hand DOFs).
- **Configuration:**  
  Both are instantiated via TeleopDeviceCfg and passed to the OpenXR device factory.

---

## ICARUS Asset and DOF Mapping

- **URDF/Asset:**  
  The ICARUS robot’s URDF and USD define the available joints and their names.
- **DOF Consistency:**  
  The lists in TeleopEnvCfg must match the robot’s actual DOF names and order.
- **Binding:**  
  TeleopEnv.bind_robot_dofs(robot.joint_names) ensures the mapping is correct and raises errors if any names are missing.

---

## Troubleshooting & Tips

- **DOF Name Mismatches:**  
  If you get errors about missing DOFs, check that your config lists match the robot asset exactly.
- **Jacobian Access:**  
  Use `robot.root_physx_view.get_jacobians()` and rotate into base frame as shown in teleop_ik.py.
- **Action Types:**  
  Always convert numpy action vectors to torch tensors before calling `set_joint_position_target`.
- **Simulation Loop:**  
  Use `simulation_app.is_running()` for the main loop and `simulation_app.close()` for shutdown.
- **Meta Quest Compatibility:**  
  OpenXR joint names and ordering are standardized; the retargeters and device expect the same.

---

## Extending and Customizing

- **Multi-Hand/Arm:**  
  Extend the DOF lists and retargeter configs for bimanual setups.
- **Alternative IK:**  
  Swap in PinkIKController if you need multi-task or null-space behaviors.
- **Custom Retargeters:**  
  Add new retargeters under isaaclab.devices.openxr.retargeters and export them in the init file.

---

## References

- [IsaacLab API Docs](https://isaac-sim.github.io/IsaacLab/main/source/api/index.html)
- [OpenXR Device/Retargeter Source](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#openxr-device)
- [DexSuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting)
- ICARUS URDF: `isaaclab_assets/data/robot/icarus/unimanual/urdf/icarus.urdf`

---

## Summary

This setup enables seamless teleoperation of the ICARUS robot in IsaacLab using Meta Quest hand tracking, OpenXR retargeters, and Differential IK. The configuration is modular, robust to asset changes, and leverages IsaacLab’s built-in APIs for simulation and control. All key mappings, data flows, and integration points are documented above for easy extension and debugging.
