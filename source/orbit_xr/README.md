# ICARUS XR Teleoperation in IsaacLab (Windows, No CloudXR)

This guide explains how to run **ICARUS teleoperation** in IsaacLab on **Windows** using Meta Quest hand tracking via OpenXR—**without CloudXR**. It covers setup, dependencies, and step-by-step instructions, including handling DexPilot (dex_retargeting) on WSL2 and Windows.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Project Layout](#project-layout)
- [How It Works](#how-it-works)
- [Running Without CloudXR](#running-without-cloudxr)
- [DexPilot (dex_retargeting) on Windows](#dexpilot-dex_retargeting-on-windows)
- [Step-by-Step: Running ICARUS Teleop](#step-by-step-running-icarus-teleop)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Key Parameters Reference](#key-parameters-reference)

---

## Overview

- **Live Meta Quest hand tracking** via OpenXR (Oculus/Meta Link or Air Link)
- **End-effector (EE) pose** from right hand, mapped to ICARUS arm via Differential IK
- **ORCA right-hand joint actuation** from XR hand keypoints
- **No CloudXR required** (runs locally on Windows)
- **DexPilot (dex_retargeting)** for hand retargeting (see [DexPilot on Windows](#dexpilot-dex_retargeting-on-windows))

---

## Prerequisites

- **Windows machine**
- **Omniverse Isaac Sim** (2023.1+ recommended)
- **IsaacLab** (this repo) installed in the same Python environment as Isaac Sim
- **NVIDIA GPU drivers** up to date
- **Meta Quest headset** with:
  - Oculus desktop app installed
  - Connected via Link cable or Air Link
  - Oculus OpenXR Runtime set as active (Oculus app → Settings → General → OpenXR Runtime)
- **No CloudXR needed**
- **Python packages**: All IsaacLab dependencies are included. For hand retargeting with DexPilot, see [DexPilot on Windows](#dexpilot-dex_retargeting-on-windows).

---

## Project Layout

Key files for teleoperation:

```
source/orbit_xr/tasks/teleop_env_cfg.py      # Robot DOF names, IK, XR retargeter config
source/orbit_xr/tasks/teleop_env.py          # OpenXR device wrapper, command splitting
source/orbit_xr/tasks/teleop_ik.py           # Differential IK controller wrapper
source/orbit_xr/tasks/run_teleop.py          # Main launch script
source/isaaclab/isaaclab/devices/openxr/retargeters/orca/  # Orca retargeter & DexPilot glue
source/isaaclab_assets/data/robot/icarus/unimanual/urdf/icarus.urdf  # ICARUS DOF names
source/isaaclab_assets/data/robot/icarus/unimanual/usd/icarus.usd    # ICARUS robot asset
```

---

## How It Works

1. **OpenXR Device + Retargeters**
   - `Se3AbsRetargeter`: Converts XR hand pose to 7D EE target for the arm.
   - `OrcaRetargeter`: Converts XR hand joints to ORCA hand DOFs (uses DexPilot if available).

2. **TeleopEnv**
   - Polls XR device, splits retargeted command into EE pose and hand joints.

3. **TeleopIK**
   - Differential IK controller computes arm joint targets to reach EE pose.

4. **Action Assembly**
   - Merges arm and hand targets into a full DOF action vector for the robot.

---

## Running Without CloudXR

- **Connect Meta Quest** via Link cable or Air Link.
- **Oculus desktop app** must be running.
- **Oculus OpenXR Runtime** must be active.
- **No CloudXR required**—everything runs locally on Windows.

---

## DexPilot (dex_retargeting) on Windows

- **DexPilot** (dex_retargeting) is used by the Orca retargeter for accurate hand mapping.
- If you have dex_retargeting **only on WSL2**, it is **not available** to IsaacSim running on Windows Python.
- **Recommended:** Install dex_retargeting in your Windows Python environment (may require extra dependencies).
- **Fallback:** Use a simpler geometric hand retargeter (e.g., DexRetargeter), or keep OrcaRetargeter and ensure your code handles missing DexPilot gracefully (hand joints will not be mapped, but arm teleop will still work).

**Note:**  
If DexPilot is not available, the Orca retargeter will not compute hand joints. You can still test arm teleop and IK.

---

## Step-by-Step: Running ICARUS Teleop

1. **Launch Isaac Sim Teleop Script**
   ```powershell
   python source/orbit_xr/tasks/run_teleop.py
   ```
   Add `--headless` if you do not need the GUI.

2. **Connect Meta Quest**
   - Wear the headset, connect via Link or Air Link.
   - Ensure hand tracking is enabled.

3. **Check OpenXR Connection**
   - The script should detect the XR device and start producing commands.
   - If not, check that the Oculus OpenXR runtime is active and the headset is awake.

4. **Verify Robot DOF Binding**
   - The script binds robot DOF names from `teleop_env_cfg.py` to the loaded asset.
   - If names do not match, you will get a clear error—edit the config to fix.

5. **Tune IK Parameters (Optional)**
   - Adjust `damping` in `TeleopEnvCfg.ik` if the arm is twitchy or unresponsive.

---

## Troubleshooting & Tips

- **XR device returns None**
  - Make sure Oculus runtime is the active OpenXR provider.
  - Headset must be awake and tracking hands.

- **DexPilot import errors**
  - Install dex_retargeting in your Windows Python environment.
  - Or, switch to a non-optimizer hand retargeter in `TeleopEnvCfg`.

- **DOF name mismatches**
  - Compare `TeleopEnvCfg.robot.*` DOF lists with `robot.joint_names` at runtime.
  - Fix names in `teleop_env_cfg.py` as needed.

- **Action tensor type**
  - `robot.set_joint_position_target` expects a `torch.Tensor`. The code converts from numpy to torch on the configured device.

---

## Key Parameters Reference

- **Arm DOF names** (`TeleopEnvCfg.robot.arm_dof_names`)
  - Example:  
    `["Revolute 1", "Revolute 3", "Revolute 9", "Revolute 20", "Revolute 26", "right_wrist"]`
  - Source: `icarus.urdf`

- **Hand DOF names** (`TeleopEnvCfg.robot.hand_dof_names`)
  - Example:  
    `["right_thumb_mcp", ..., "right_pinky_pip"]`

- **EE retargeter** (`TeleopEnvCfg.device.ee_retargeter`)
  - `Se3AbsRetargeter` config (right hand, wrist position, etc.)

- **Hand retargeter** (`TeleopEnvCfg.device.hand_retargeter`)
  - `OrcaRetargeter` config (uses DexPilot if available)

- **IK damping** (`TeleopIKConfig.damping`)
  - Damped least-squares lambda for IK (start at `0.05`, tune as needed)

---

## What If DexPilot Is Not Available on Windows?

- In `TeleopEnvCfg`, temporarily use a simpler hand retargeter or ensure OrcaRetargeter handles missing DexPilot.
- Arm teleop and IK will still work; only optimizer-based hand mapping will be skipped.
- To enable full hand teleop, install `dex_retargeting` in your Windows Python environment.

---

## Summary

- **No CloudXR needed:** Everything runs locally on Windows.
- **Meta Quest hand tracking** via OpenXR.
- **ICARUS arm** is controlled by Differential IK.
- **ORCA hand** is mapped via OrcaRetargeter (DexPilot required for full mapping).
- **Troubleshoot** by checking OpenXR runtime, DOF names, and Python dependencies.

---

**For further details, see the code and configs in `source/orbit_xr/tasks/`.**