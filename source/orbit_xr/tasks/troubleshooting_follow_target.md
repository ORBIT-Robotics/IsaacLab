# Ikarus Follow-Target: Troubleshooting and Correct Usage

This document captures the issues we hit bringing up the Ikarus follow‑target example in Isaac Sim/Isaac Lab, what caused them, and the precise code and launch practices that fixed them. Use it as a checklist when wiring up a new scene or controller.

---

## 1) Launching Isaac Sim correctly

### Symptom
- UI appears frozen. Hiding/deleting prims in Stage doesn’t change the viewport. Sometimes the window closes right away after printing “Closing SimulationApp”.

### Root Cause
- Launching a second `SimulationApp` inside a process that already has Kit running (for example via `isaaclab.sh -p`) causes conflicts. Additionally, `isaaclab.sh` injects `--/app/fastShutdown=True`, which can make a newly created `SimulationApp` exit immediately.

### Correct Practices

- Run under Kit (preferred for interactive debugging). Do not create a new `SimulationApp` — attach a per‑frame callback:

```bash
# From repository root
./isaaclab.sh -p source/orbit_xr/tasks/follow_target/run_follow_target.py

# With script arguments (note the "--" separator after the script path)
./isaaclab.sh -p source/orbit_xr/tasks/follow_target/run_follow_target.py -- \
  --target-x 0.60 --target-y 0.00 --target-z 0.90 --ik_damping 0.05
```

- If you want the script to own the app loop, run via Isaac Sim’s python wrapper (no fastShutdown):

```bash
./_isaac_sim/python.sh source/orbit_xr/tasks/follow_target/run_follow_target.py
```

- Inside the script, detect whether Kit already exists and choose the right loop:

```python
# run_follow_target.py
from isaacsim.simulation_app import SimulationApp

def _maybe_get_kit_app():
    try:
        import omni.kit.app as kitapp
        return kitapp.get_app()
    except Exception:
        return None

kit_app = _maybe_get_kit_app()
if kit_app is None:
    simulation_app = SimulationApp({"headless": False})  # standalone
else:
    simulation_app = None  # –p mode: register per‑frame callback on Kit
```

---

## 2) sys.path reset after app initialization

### Symptom
- `ModuleNotFoundError: No module named 'orbit_xr'` after creating `SimulationApp`.

### Root Cause
- Some builds reset `sys.path` during app init.

### Fix

```python
# After SimulationApp creation
import sys
from pathlib import Path
REPO_SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(REPO_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_SOURCE_DIR))
```

---

## 3) Ground plane crashes via content spawner

### Symptom
- Immediate crash in `GroundPlaneCfg` (`bind_physics_material(...)` on a `None` path).

### Root Cause
- The standard grid USD wasn’t present/compatible locally; the spawner tried to bind a material to a missing child prim.

### Fix: Minimal local ground + light

```python
import isaacsim.core.utils.prims as prim_utils

ground_prim = "/World/Environment/Ground"
prim_utils.create_prim("/World/Lights/KeyLight", "DistantLight")
prim_utils.create_prim(ground_prim, "Xform", translation=(0.0, 0.0, -1.0))
prim_utils.create_prim(f"{ground_prim}/Plane", "Plane", translation=(0.0, 0.0, 0.0))
prim_utils.set_prim_property(f"{ground_prim}/Plane", "xformOp:scale", (100.0, 100.0, 1.0))
```

---

## 4) Target prim not reacting to hide/delete

### Symptom
- Hiding/deleting the cube in Stage doesn’t change the viewport.

### Root Cause
- The shape spawner was cloning into Fabric. Fabric clones aren’t controlled via regular USD hide/delete.

### Fix: Spawn without Fabric cloning

```python
sim_utils.CuboidCfg(
    size=(0.15, 0.15, 0.15),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
).func(
    "/World/Targets/ikarus_goal",
    target_cfg,
    translation=(0.6, 0.0, 0.9),
    orientation=(1.0, 0.0, 0.0, 0.0),
    clone_in_fabric=False,
)
```

---

## 5) Asset loading: sublayer vs reference

### Symptom
- Robot Xform exists but meshes/materials don’t render.

### Root Cause
- The USD asset uses absolute internal prim paths (e.g., `"/icarus_orca/..."`). Referencing under `/World` breaks those relationships.

### Fix: Sublayer the robot USD

```python
from isaacsim.core.utils.stage import get_current_stage

stage = get_current_stage()
root_layer = stage.GetRootLayer()
usd_path = str(IKARUS_USD_PATH)
if usd_path not in root_layer.subLayerPaths:
    root_layer.subLayerPaths.append(usd_path)

for candidate in ["/icarus_orca", "/Root/icarus_orca", "/ICARUS/icarus_orca"]:
    if stage.GetPrimAtPath(candidate).IsValid():
        prim_path = candidate
        break
```

---

## 6) GPU tensor → NumPy crash in `get_target_pose()`

### Symptom
- `TypeError: can't convert cuda:0 device type tensor to numpy` when reading cube pose.

### Root Cause
- `XFormPrim.get_world_poses()` returns torch tensors on the sim device. NumPy can’t consume CUDA tensors.

### Fix

```python
pos, quat = positions[0], orientations[0]
if isinstance(pos, torch.Tensor):
    pos = pos.detach().cpu().numpy()
if isinstance(quat, torch.Tensor):
    quat = quat.detach().cpu().numpy()
pose = np.concatenate([pos.astype(np.float32), quat.astype(np.float32)])
```

---

## 7) IK stability fixes

### Problems observed
- Wrist oscillates wildly; arm doesn’t converge to the cube; fingers jitter although we’re not commanding them.

### Causes & Fixes
1. **Frame mismatch** — target provided in world frame, Jacobian/current EE in base frame.
   - Convert target pose from world → base before calling the controller.
2. **Wrong EE link** — using `right_wrist_jointbody` instead of the rigid palm.
   - Set `IKARUS_EE_LINK = "right_palm"`.
3. **Large per-step deltas** — DLS solution can jump violently near singularities.
   - Clamp per-step joint change via `max_step` (default 0.05 rad).
4. **Jacobian body index heuristic** — use the EE body index directly instead of `ee_id - 1`.
5. **Orientation tracking not needed** — we only care about position, so run the controller in `command_type="position"` mode.

```python
@dataclass
class TeleopIKConfig:
    ...
    command_type: Literal["position", "pose"] = "position"
    max_step: float = 0.05

# In solve()
target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(...)
if self._cfg.command_type == "position":
    self._controller.set_command(target_pos_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
else:
    target_tensor = torch.cat([target_pos_b, target_quat_b], dim=-1)
    self._controller.set_command(target_tensor)

q_des = self._controller.compute(...)
dq = torch.clamp(q_des - joint_pos, min=-self._max_step, max=self._max_step)
q_next = joint_pos + dq
```

Update the joint target composition so hand DOFs stay untouched:

```python
def compose_joint_targets(arm_joint_targets: np.ndarray) -> torch.Tensor:
    joint_targets = robot.data.joint_pos.clone()
    arm_tensor = torch.as_tensor(arm_joint_targets, dtype=torch.float32, device=joint_targets.device)
    joint_targets[:, arm_joint_ids] = arm_tensor.unsqueeze(0)
    return joint_targets
```

---

## 8) Quick sanity checklist

- Launch under Kit with `./isaaclab.sh -p source/orbit_xr/tasks/follow_target/run_follow_target.py`.
- Ground plane = plain Plane + DistantLight.
- Target cube spawned with `clone_in_fabric=False`.
- Robot USD added as a sublayer; prim path auto-detected (e.g., `/Root/icarus_orca`).
- Always convert torch CUDA tensors to CPU/NumPy when leaving the sim world.
- IK config: `command_type="position"`, `max_step≈0.03–0.05`, DLS damping tuned (start at 0.05).
- Compose joint targets from the current joint state so uncommanded DOFs (fingers) stay put.

If instability persists:
- Increase DLS damping (0.1–0.2).
- Reduce `max_step` (e.g., 0.02) for smoother motion.
- Switch IK method to `svd` with a higher `min_singular_value`.

---

This guide mirrors the fixes in:
- `source/orbit_xr/tasks/follow_target/run_follow_target.py`
- `source/orbit_xr/tasks/follow_target/follow_target.py`
- `source/orbit_xr/tasks/ikarus_ik.py`

Use the snippets above whenever you wire new tasks/controllers to avoid repeating these pitfalls.
