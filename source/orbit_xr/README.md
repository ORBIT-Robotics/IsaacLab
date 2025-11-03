# ICARUS XR Teleoperation in IsaacLab (Linux, SteamVR + ALVR, No CloudXR)

This guide explains how to run ICARUS teleoperation in IsaacLab on Linux (Ubuntu 22.04/24.04) using a Meta Quest via ALVR + SteamVR (OpenXR), without CloudXR. It covers setup, dependencies, step-by-step instructions, and Linux-specific troubleshooting.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites (Linux)](#prerequisites-linux)
- [Project Layout](#project-layout)
- [How It Works](#how-it-works)
- [Running Without CloudXR](#running-without-cloudxr)
- [Step-by-Step: Running ICARUS Teleop (Linux)](#step-by-step-running-icarus-teleop-linux)
- [DexPilot (dex_retargeting) on Linux](#dexpilot-dex_retargeting-on-linux)
- [Troubleshooting & Tips (Linux)](#troubleshooting--tips-linux)
- [Key Parameters Reference](#key-parameters-reference)
- [Summary](#summary)
- [Quick Start](#quick-start)

---

## Overview

- Meta Quest streaming with ALVR → SteamVR as OpenXR runtime on Linux.
- End-effector (EE) pose from controller/hand → ICARUS arm via Differential IK.
- ORCA right-hand joint actuation from XR keypoints (via hand retargeter; DexPilot optional).
- No CloudXR required. Works with GUI (recommended) or headless.

---

## Prerequisites (Linux)

- Ubuntu 22.04 / 24.04, NVIDIA GPU (RTX/Ada recommended).
- NVIDIA drivers and Vulkan working:
  ```bash
  nvidia-smi
  glxinfo -B | grep "OpenGL renderer"
  vulkaninfo | grep -m1 deviceName
  ```
- Isaac Sim 2023.1+ / Isaac Lab installed. From repo root, ensure launcher is available (example):
  ```bash
  ls -l ./isaaclab.sh
  # Optional: link Isaac Sim if your workflow expects it
  ls -l _isaac_sim || ln -s /path/to/isaac-sim _isaac_sim
  ```
- Python env (conda/venv) for Isaac Lab.
- Steam + SteamVR installed.
- ALVR installed (PC server + Quest client) and paired.
  - Set SteamVR as OpenXR Runtime: SteamVR → Settings → Developer → “Set SteamVR as OpenXR Runtime”.
  - Verify:
    ```bash
    cat ~/.config/openxr/1/active_runtime.json  # should end with .../SteamVR/steamxr_linux64.json
    ```
- Network: Prefer PC on Ethernet; Quest on 5 GHz.

---

## Project Layout

Key files for teleoperation (paths may vary slightly):

```
source/orbit_xr/tasks/teleop_env_cfg.py      # Robot DOF names, IK, XR retargeter config
source/orbit_xr/tasks/teleop_env.py          # XR device wrapper, command splitting
source/orbit_xr/tasks/teleop_ik.py           # Differential IK controller wrapper
source/orbit_xr/tasks/run_teleop.py          # Main launch script
source/orbit_xr/mappings/                    # Controller/hand → action mappings
source/orbit_xr/retargeters/                 # Hand retargeters (e.g., ORCA/DexPilot glue)
source/isaaclab_assets/.../icarus.urdf       # ICARUS DOF names
source/isaaclab_assets/.../icarus.usd        # ICARUS asset
```

---

## How It Works

1) OpenXR device + retargeters
- EE retargeter (e.g., Se3AbsRetargeter): XR hand/controller pose → 7D EE target.
- Hand retargeter (e.g., OrcaRetargeter): XR hand joints → ORCA hand DOFs (DexPilot optional).

2) TeleopEnv
- Polls HMD/controllers/hand data and splits into EE pose + hand joints.

3) TeleopIK
- Differential IK computes arm joint targets for the EE target.

4) Action assembly
- Merges arm + hand into a full DOF action and applies in Isaac Lab.

---

## Running Without CloudXR

- Quest ↔ PC uses ALVR streaming over Wi‑Fi/Ethernet.
- SteamVR runs as the OpenXR runtime on Linux.
- No CloudXR or Meta Link (USB video) required.

---

## Step-by-Step: Running ICARUS Teleop (Linux)

Order matters. Follow on each run.

0) One-time check (OpenXR runtime):
```bash
cat ~/.config/openxr/1/active_runtime.json  # → steamxr_linux64.json path
```

1) Start streaming & SteamVR
- Launch ALVR on PC → ensure the Quest shows “Streaming”.
- From ALVR dashboard, click “Launch SteamVR”.
- Put on the headset; you should see the SteamVR environment (SteamVR Home can be disabled).

2) Run the teleop script (repo root)
```bash
# Activate your Isaac Lab env
conda activate <your_env>  # or source <venv>/bin/activate

# Optional: ensure Isaac Sim link (if your workflow uses it)
ls -l _isaac_sim || ln -s /path/to/isaac-sim _isaac_sim

# GUI run (recommended first time)
./isaaclab.sh -p source/orbit_xr/tasks/run_teleop.py

# Or headless (if supported by your setup)
# ./isaaclab.sh -p source/orbit_xr/tasks/run_teleop.py --headless
```

Optional flags (only if supported by your script):
```bash
./isaaclab.sh -p source/orbit_xr/tasks/run_teleop.py \
  --xr-runtime steamvr \
  --ee-source right_controller \
  --hand-mode controllers \
  --ik-damping 0.05
```

3) Verify device & DOF bindings
- Console should report XR device present and robot DOFs bound.
- If names mismatch, edit `source/orbit_xr/tasks/teleop_env_cfg.py`.

4) Use it
- Move the right controller to drive the EE.
- Buttons/axes are mapped in `source/orbit_xr/mappings/`.
- For hand tracking, ensure the runtime exposes hand joints and enable the hand retargeter.

---

## DexPilot (dex_retargeting) on Linux

If your hand retargeter uses DexPilot:

- Install in the same Python env as Isaac Lab.

Option A — Submodule (recommended if the repo includes it):
```bash
cd /home/orbit/Desktop/IsaacLab
git submodule sync --recursive
git submodule update --init --recursive
git submodule status
```

If you see an empty or broken symlink folder (e.g., dex-retargeting):
```bash
# Replace a plain symlink with a real submodule (or clone)
rm -f dex-retargeting
# As submodule:
git submodule add https://github.com/<org>/<dex_retargeting_repo>.git dex-retargeting
git submodule update --init --recursive
git commit -m "Add dex-retargeting as submodule"

# Or just clone:
# git clone https://github.com/<org>/<dex_retargeting_repo>.git dex-retargeting
```

Then install it (editable recommended for development):
```bash
pip install -e dex-retargeting
```

Option B — Local repo
```bash
git clone https://github.com/<org>/<dex_retargeting_repo>.git
pip install -e ./<dex_retargeting_repo>
```

If DexPilot is unavailable, use a geometric hand retargeter fallback or handle absence gracefully (arm teleop still works; only hand joints are skipped).

---

## Troubleshooting & Tips (Linux)

A) SteamVR / OpenXR
- Set runtime: SteamVR → Settings → Developer → “Set SteamVR as OpenXR Runtime”.
- Verify:
  ```bash
  cat ~/.config/openxr/1/active_runtime.json
  ```
- “Some Add-ons Blocked” after a crash:
  - SteamVR → Manage Add-ons: enable ALVR, disable extras; turn OFF SteamVR Home.
- Clean restart if compositor is stuck:
  ```bash
  pkill -f vrcompositor ; pkill -f vrserver ; pkill -f steam
  steam &
  ```
- Launch option workaround (fix certain black-screen cases):
  - Steam → Library → SteamVR → Properties → Launch Options:
    ```
    /home/<USER>/.steam/steam/steamapps/common/SteamVR/bin/vrmonitor.sh %command%
    ```

B) ALVR stream black but “Streaming”
- Use H.264 (NVENC), 72 Hz, Constant 37–45 Mbps.
- Confirm NVENC in use while streaming:
  ```bash
  nvidia-smi --query-gpu=encoder.stats.sessionCount,encoder.stats.averageFps --format=csv
  ```
- Check listeners:
  ```bash
  ss -lntp | grep 8082   # ALVR Web UI (TCP)
  ss -lunp | grep 9944   # ALVR stream (UDP)
  ```

C) No menu / scenery only
- Use the left controller menu/system button for the SteamVR dashboard.
- In ALVR client first-run, disable “Only touch” so gestures map to buttons.

D) Isaac Lab specifics
- DOF mismatches: compare `TeleopEnvCfg.robot.*` with your asset’s joint names and fix the config.
- IK tuning: increase `damping` for stability (e.g., 0.05 → 0.1), decrease for responsiveness.

E) Network
- Prefer PC on Ethernet, Quest on 5 GHz. If discovery fails, connect via manual IP in ALVR.

---

## Key Parameters Reference

- Arm DOF names (`TeleopEnvCfg.robot.arm_dof_names`)
  - Example (adapt to your asset):
    ```
    ["shoulder_yaw", "shoulder_pitch", "shoulder_roll",
     "elbow_pitch", "wrist_pitch", "wrist_yaw"]
    ```
- Hand DOF names (`TeleopEnvCfg.robot.hand_dof_names`)
  - Match your ORCA hand joint list.
- EE retargeter (`TeleopEnvCfg.device.ee_retargeter`)
  - Se3AbsRetargeter (source = right controller/hand).
- Hand retargeter (`TeleopEnvCfg.device.hand_retargeter`)
  - OrcaRetargeter (DexPilot optional).
- IK damping (`TeleopIKConfig.damping`)
  - Start at 0.05; tune 0.01–0.2 based on stability.

---

## Summary

- Linux path: ALVR (Quest) → SteamVR (OpenXR) → Isaac Lab. No CloudXR required.
- ICARUS arm via Differential IK; ORCA hand via retargeter (DexPilot optional).
- If stuck: verify OpenXR runtime, enable ALVR add-on, check NVENC stats, confirm ALVR ports, fix DOF names.

---

## Quick Start

```bash
# 1) Ensure SteamVR is OpenXR runtime
cat ~/.config/openxr/1/active_runtime.json

# 2) Start ALVR and click "Launch SteamVR"

# 3) Run teleop from repo root
conda activate <your_env>
cd /home/orbit/Desktop/IsaacLab
ls -l _isaac_sim || ln -s /path/to/isaac-sim _isaac_sim
./isaaclab.sh -p source/orbit_xr/tasks/run_teleop.py
```