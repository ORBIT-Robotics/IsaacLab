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
REPO_SOURCE_DIR = Path(__file__).resolve().parents[3]
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

This avoids any Nucleus/content dependencies and yields a readable, stable background.

---

## 4) Target prim not reacting to hide/delete

### Symptom
- Hiding/deleting the cube in Stage doesn’t change the viewport.

### Root Cause
- The shape spawner was cloning into Fabric. Fabric clones aren’t controlled via regular USD hide/delete.

### Fix: Spawn without Fabric cloning

```python
# follow_target.py
sim_utils.CuboidCfg(
    size=(0.15, 0.15, 0.15),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
).func(
    "/World/Targets/ikarus_goal",
    target_cfg,
    translation=(0.6, 0.0, 0.9),
    orientation=(1.0, 0.0, 0.0, 0.0),
    clone_in_fabric=False,   # the key line
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
# follow_target.py
from isaacsim.core.utils.stage import get_current_stage

stage = get_current_stage()
root_layer = stage.GetRootLayer()
usd_path = str(IKARUS_USD_PATH)
if usd_path not in root_layer.subLayerPaths:
    root_layer.subLayerPaths.append(usd_path)

# Then auto‑detect prim
for candidate in ["/icarus_orca", "/Root/icarus_orca", "/ICARUS/icarus_orca"]:
    if stage.GetPrimAtPath(candidate).IsValid():
        prim_path = candidate
        break
```

Sublayering preserves the asset’s absolute paths in the composed stage.

---

## 6) GPU tensor → NumPy crash in `get_target_pose()`

### Symptom
- `TypeError: can't convert cuda:0 device type tensor to numpy` when reading cube pose.

### Root Cause
- `XFormPrim.get_world_poses()` returns torch tensors on the sim device. NumPy can’t consume CUDA tensors.

### Fix

```python
# follow_target.py
pos, quat = positions[0], orientations[0]
if isinstance(pos, torch.Tensor):
    pos = pos.detach().cpu().numpy()
if isinstance(quat, torch.Tensor):
    quat = quat.detach().cpu().numpy()
pose = np.concatenate([pos.astype(np.float32), quat.astype(np.float32)])
```

---

## 7) IK was unstable and didn’t track the cube

### Symptoms
- Wrist shakes; arm doesn’t converge; moving the cube barely changes behavior.

### Root Causes
1. Frame mismatch: target pose given in world frame while Jacobian/current EE were in base frame.
2. End‑effector link mismatch: used a wrist helper link instead of the rigid palm.
3. Large per‑step deltas near singularities caused oscillations.
4. Jacobian body index heuristic was wrong for our articulation ordering.

### Fixes (all applied)

- Use the palm as the EE link:

```python
# follow_target.py
IKARUS_EE_LINK = "right_palm"
```

- Transform target pose from world → base before IK:

```python
# ikarus_ik.py (inside solve)
# root_pos_w, root_quat_w, ee_pos_b, ee_quat_b, jacobian_b already computed

target_pos_w = torch.tensor(target_pose[:3], device=self._device).unsqueeze(0)
target_quat_w = torch.tensor(target_pose[3:], device=self._device).unsqueeze(0)

target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
    root_pos_w, root_quat_w, target_pos_w, target_quat_w
)

cmd = torch.cat([target_pos_b, target_quat_b], dim=-1)
self._controller.set_command(cmd)
```

- Use the resolved EE body index directly for the Jacobian:

```python
self._ee_body_id = ee_ids[0]
self._jacobian_body_id = self._ee_body_id
jacobian_w = self._articulation.root_physx_view.get_jacobians()[:, self._jacobian_body_id, :, self._arm_joint_ids]
```

- Clamp per‑step joint change:

```python
# Config
@dataclass
class TeleopIKConfig:
    ...
    max_step: float = 0.05  # rad per step

# Compute
q_des = self._controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
dq = torch.clamp(q_des - joint_pos, min=-self._max_step, max=self._max_step)
q_next = joint_pos + dq
return q_next[0].detach().cpu().numpy()
```

- Keep DLS damping modest (e.g., `lambda_val=0.05`) and consider increasing if you still see chattering.

---

## 8) Minimal, robust main loop patterns

### Under Kit (–p): register a per‑frame callback

```python
kit_app = _maybe_get_kit_app()
stream = kit_app.get_update_event_stream()
stream.create_subscription_to_pop(_on_frame)  # _on_frame calls sim.step(), solves IK, applies joint targets
```

### Standalone: own the SimulationApp

```python
simulation_app = SimulationApp({"headless": False})
try:
    while simulation_app.is_running():
        simulation_app.update()
        _on_frame()
finally:
    simulation_app.close()
```

---

## 9) Quick sanity checklist

- Launch:
  - Kit UI: `./isaaclab.sh -p source/orbit_xr/tasks/follow_target/run_follow_target.py`
  - Standalone: `./_isaac_sim/python.sh source/orbit_xr/tasks/follow_target/run_follow_target.py`
- Ground: simple `Plane` + `DistantLight`, no external dependencies.
- Target cube: spawned with `clone_in_fabric=False`, hide/delete works.
- Robot asset: sublayered; prim path auto‑detected (e.g., `/Root/icarus_orca`).
- Poses: always convert torch CUDA tensors to CPU before NumPy.
- IK: target ↦ base frame, EE=`right_palm`, Jacobian index = EE body, clamp per‑step Δq.

If you still see instability, try:
- Increase DLS damping (e.g., 0.1–0.2).
- Reduce `max_step` (e.g., 0.02) for smoother motion.
- Switch to `ik_method="svd"` and set `min_singular_value=1e-3`.

---

This guide reflects the exact fixes encoded in:
- `source/orbit_xr/tasks/follow_target/run_follow_target.py`
- `source/orbit_xr/tasks/follow_target/follow_target.py`
- `source/orbit_xr/tasks/ikarus_ik.py`

Use the code snippets above when wiring new tasks or controllers to avoid the same pitfalls.

