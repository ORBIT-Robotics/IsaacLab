from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from isaacsim.simulation_app import SimulationApp

# Ensure the repository's `source` directory is available on sys.path so we can import orbit_xr modules.
import sys
# Add the repository's `source` folder (parent of the `orbit_xr` package)
REPO_SOURCE_DIR = Path(__file__).resolve().parents[3]
if str(REPO_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_SOURCE_DIR))

# Isaac Sim/Omniverse modules may still import NumPy aliases that were removed in NumPy 2.x.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the Ikarus arm toward an interactive cube target.")
    parser.add_argument("--target-x", type=float, default=0.6, help="Initial X coordinate of the cube in meters.")
    parser.add_argument("--target-y", type=float, default=0.0, help="Initial Y coordinate of the cube in meters.")
    parser.add_argument("--target-z", type=float, default=0.9, help="Initial Z coordinate of the cube in meters.")
    parser.add_argument("--ik_damping", type=float, default=0.15, help="Damping term for the IK solver.")
    parser.add_argument("--max_step", type=float, default=0.015, help="Max joint change per step (radians).")
    parser.add_argument("--ik_method", type=str, default="svd", choices=["dls", "svd", "trans"], help="IK method.")
    parser.add_argument("--min_singular_value", type=float, default=1e-3, help="Min singular value for SVD.")
    parser.add_argument("--low_pass_alpha", type=float, default=0.3, help="Low-pass filter coefficient (0=off, 0.9=aggressive).")
    return parser.parse_args()


def _maybe_get_kit_app():
    try:
        import omni.kit.app as kitapp  # noqa: WPS433
        return kitapp.get_app()
    except Exception:
        return None


def main(args: argparse.Namespace) -> None:
    kit_app = _maybe_get_kit_app()
    simulation_app = None
    if kit_app is None:
        print("[run_follow_target] Launch mode: standalone (SimulationApp)", flush=True)
        simulation_app = SimulationApp({"headless": False})
    else:
        print("[run_follow_target] Launch mode: -p (existing Kit app)", flush=True)

    # Some Isaac Sim builds reset sys.path during SimulationApp init.
    # Ensure the repository's source directory is still importable.
    if str(REPO_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_SOURCE_DIR))

    from orbit_xr.tasks.follow_target.follow_target import (
        FollowTargetCfg,
        FollowTargetTask,
        IKARUS_ARM_DOF_NAMES,
        IKARUS_BASE_LINK,
        IKARUS_EE_LINK,
        create_ikarus_articulation,
    )
    from orbit_xr.tasks.ikarus_ik import TeleopIK, TeleopIKConfig

    import isaaclab.sim as sim_utils  # noqa: WPS433
    import isaacsim.core.utils.prims as prim_utils  # noqa: WPS433
    from pxr import Gf  # noqa: WPS433

    GROUND_Z_OFFSET = -1.0

    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    print(f"[run_follow_target] SimulationContext ready | device={sim.device} dt={sim_cfg.dt}", flush=True)
    sim.set_camera_view(eye=[1.5, 0.0, 1.3], target=[0.0, 0.0, 0.8])  # type: ignore[attr-defined]

    # Ensure transform gizmo is available for interactive editing
    try:
        import omni.kit.app  # noqa: WPS433
        from isaacsim.core.utils.extensions import enable_extension  # noqa: WPS433

        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled("omni.kit.manipulator.transform"):
            enable_extension("omni.kit.manipulator.transform")
    except Exception:
        pass

    # Spawn default grid ground plane and lighting
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    # Add distant light for better visibility
    light_cfg = sim_utils.DistantLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Lights/KeyLight", light_cfg)
    print("[run_follow_target] Grid ground plane + light spawned", flush=True)

    target_cfg = FollowTargetCfg(
        initial_position=(args.target_x, args.target_y, args.target_z),
    )
    follow_task = FollowTargetTask(target_cfg)
    print("[run_follow_target] Spawning target cube…", flush=True)
    follow_task.spawn()

    print("[run_follow_target] Loading Ikarus articulation…", flush=True)
    robot = create_ikarus_articulation()
    print("[run_follow_target] Ikarus articulation loaded", flush=True)

    print("[run_follow_target] Resetting simulation…", flush=True)
    sim.reset()

    robot_joint_names: Sequence[str] = tuple(robot.joint_names)
    missing = [name for name in IKARUS_ARM_DOF_NAMES if name not in robot_joint_names]
    if missing:
        print(f"[run_follow_target] Robot joints: {robot_joint_names}", flush=True)
        raise RuntimeError(f"Missing IK arm joints in the Ikarus articulation: {missing}")
    arm_joint_ids = [robot_joint_names.index(name) for name in IKARUS_ARM_DOF_NAMES]
    
    # Verify that arm DOF names don't accidentally include hand joints
    from orbit_xr.tasks.follow_target.follow_target import IKARUS_HAND_DOF_NAMES
    arm_hand_overlap = set(IKARUS_ARM_DOF_NAMES) & set(IKARUS_HAND_DOF_NAMES)
    if arm_hand_overlap:
        raise RuntimeError(f"Arm DOF names overlap with hand DOF names: {arm_hand_overlap}")
    print(f"[run_follow_target] Verified arm DOFs ({len(IKARUS_ARM_DOF_NAMES)}): {IKARUS_ARM_DOF_NAMES}", flush=True)
    print(f"[run_follow_target] Hand DOFs ({len(IKARUS_HAND_DOF_NAMES)}) will remain fixed", flush=True)

    ik_solver = TeleopIK(
        TeleopIKConfig(
            device=sim.device,
            ee_link_name=IKARUS_EE_LINK,
            base_link_name=IKARUS_BASE_LINK,
            arm_dof_names=IKARUS_ARM_DOF_NAMES,
            damping=args.ik_damping,
            command_type="position",
            max_step=args.max_step,
            ik_method=args.ik_method,
            min_singular_value=args.min_singular_value,
            low_pass_alpha=args.low_pass_alpha,
        ),
        articulation=robot,
    )
    print(f"[run_follow_target] IK solver: method={args.ik_method}, damping={args.ik_damping}, max_step={args.max_step}, low_pass={args.low_pass_alpha}", flush=True)

    def compose_joint_targets(arm_joint_targets: np.ndarray) -> torch.Tensor:
        """Insert IK arm targets into the full joint vector while keeping the ORCA hand fixed."""
        if arm_joint_targets.shape[0] != len(arm_joint_ids):
            raise ValueError(
                f"Expected {len(arm_joint_ids)} arm joint targets but received {arm_joint_targets.shape[0]} values."
            )
        joint_targets = robot.data.joint_pos.clone()
        arm_tensor = torch.as_tensor(arm_joint_targets, dtype=torch.float32, device=joint_targets.device)
        joint_targets[:, arm_joint_ids] = arm_tensor.unsqueeze(0)
        return joint_targets

    # Get hand joint indices for verification
    hand_joint_ids = [robot_joint_names.index(name) for name in IKARUS_HAND_DOF_NAMES if name in robot_joint_names]
    
    # Frame counter for periodic hand DOF verification
    frame_counter = 0
    prev_hand_positions = None

    sim.reset()
    print(
        "\nMove the cube prim at "
        f"{follow_task.cfg.prim_path} to a new pose and the Ikarus wrist will follow it.\n"
    )

    def _on_frame(_=None):
        nonlocal frame_counter, prev_hand_positions
        sim.step()
        target_pose = follow_task.get_target_pose()
        arm_targets = ik_solver.solve(target_pose)
        full_targets = compose_joint_targets(arm_targets)
        
        # Verify hand DOFs are not being changed (every 100 frames)
        if hand_joint_ids and frame_counter % 100 == 0:
            current_hand_positions = robot.data.joint_pos[0, hand_joint_ids].cpu().numpy()
            target_hand_positions = full_targets[0, hand_joint_ids].cpu().numpy()
            hand_diff = np.abs(current_hand_positions - target_hand_positions)
            max_hand_change = hand_diff.max()
            if max_hand_change > 1e-6:
                print(f"[WARNING] Hand DOFs changed! Max change: {max_hand_change:.6f} rad", flush=True)
            if prev_hand_positions is not None:
                hand_drift = np.abs(current_hand_positions - prev_hand_positions)
                max_drift = hand_drift.max()
                if max_drift > 1e-4:
                    print(f"[INFO] Hand DOF drift detected: {max_drift:.6f} rad", flush=True)
            prev_hand_positions = current_hand_positions.copy()
        
        frame_counter += 1
        robot.set_joint_position_target(full_targets)
        robot.write_data_to_sim()

    if simulation_app is not None:
        try:
            while simulation_app.is_running():
                simulation_app.update()
                _on_frame()
        finally:
            print("[run_follow_target] Closing SimulationApp", flush=True)
            simulation_app.close()
    else:
        # Running under existing Kit (e.g., -p). Register per-frame callback and return.
        print("[run_follow_target] Registering per-frame callback", flush=True)
        stream = kit_app.get_update_event_stream()
        stream.create_subscription_to_pop(_on_frame)


if __name__ == "__main__":
    main(parse_args())
