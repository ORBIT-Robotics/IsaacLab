from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from isaaclab.app import AppLauncher

# Ensure the repository's `source` directory is on sys.path so modules like
# `orbit_xr` are importable even after SimulationApp initialization.
import sys
SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

# Isaac Sim 5.1 ships XR extensions that still reference NumPy 1.x dtypes.
# NumPy 2.0 removed aliases like np.float_ / np.complex_. Re-introduce them so
# omni.kit.xr.* imports succeed when running with newer NumPy builds.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # type: ignore[attr-defined]

# Provide type-only imports so editors (Pylance) know these names
# without importing Omniverse/Isaac at runtime before SimulationApp exists.
if TYPE_CHECKING:  # pragma: no cover
    from orbit_xr.tasks.teleop_env_cfg import TeleopEnvCfg

import torch

# Defer all other imports (that touch omni/isaacsim) until after SimulationApp instantiation


def create_robot_articulation(sim: Any, cfg: TeleopEnvCfg) -> Any:
    """
    Placeholder loader: replace this with your actual robot-loading code.

    Return an isaaclab.assets.articulation.Articulation positioned and ready in the scene.
    The 'sim' type is intentionally Any to avoid SimulationContext type mismatches across packages.
    """
    assets_root = Path(__file__).resolve().parents[3] / "source" / "isaaclab_assets" / "data" / "robot" / "icarus" / "unimanual"
    usd_path = assets_root / "usd" / "icarus.usd"

    robot_prim_path = "/World/ICARUS"

    # Import here to ensure SimulationApp is already instantiated
    import isaaclab.sim as sim_utils  # noqa: WPS433
    from isaaclab.assets.articulation.articulation_cfg import (
        ArticulationCfg as ArticulationCfgType,  # noqa: WPS433
    )
    from isaaclab.assets import Articulation  # noqa: WPS433
    from isaaclab.actuators import ImplicitActuatorCfg  # noqa: WPS433

    spawn_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    spawn_cfg.func(robot_prim_path, spawn_cfg)

    def _pattern(name: str) -> str:
        # Isaac assets often use underscores in joint names; allow config with spaces
        sanitized = name.replace(" ", "_")
        return f"^{sanitized}$"

    arm_joint_patterns = [_pattern(name) for name in cfg.robot.arm_dof_names]
    hand_joint_patterns = [_pattern(name) for name in cfg.robot.hand_dof_names]

    robot_cfg = ArticulationCfgType(
        prim_path=robot_prim_path,
        spawn=None,
        init_state=ArticulationCfgType.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=arm_joint_patterns,
                effort_limit=torch.inf,
                velocity_limit=torch.inf,
                stiffness=None,
                damping=None,
                armature=0.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=hand_joint_patterns,
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
            ),
        },
    )

    return Articulation(cfg=robot_cfg)


def main(args: argparse.Namespace) -> None:
    # Launch the app using IsaacLab's AppLauncher (wraps isaacsim.SimulationApp)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import modules that rely on Omniverse/Isaac after SimulationApp is created
    from orbit_xr.tasks.teleop_env_cfg import TeleopEnvCfg  # noqa: WPS433
    from orbit_xr.tasks.teleop_env import TeleopEnv  # noqa: WPS433
    from orbit_xr.tasks.teleop_ik import TeleopIK, TeleopIKConfig  # noqa: WPS433

    # Import Isaac Lab sim utils only after SimulationApp is created
    import isaaclab.sim as sim_utils  # noqa: WPS433

    # IsaacLab SimulationContext
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    import isaacsim.core.utils.prims as prim_utils  # noqa: WPS433
    from pxr import Gf  # noqa: WPS433

    # Spawn default grid environment as ground plane (default_environment.usd)
    # Uses the built-in GroundPlaneCfg which points to
    # ISAAC_NUCLEUS_DIR/Environments/Grid/default_environment.usd
    ground_prim = "/World/Environment/DefaultGround"
    ground_cfg = sim_utils.GroundPlaneCfg()
    try:
        ground_cfg.func(ground_prim, ground_cfg)
    except FileNotFoundError:
        prim_utils.create_prim(ground_prim, "Xform", translation=(0.0, 0.0, 0.0))
        prim_utils.create_prim(
            f"{ground_prim}/Plane",
            "Plane",
            translation=(0.0, 0.0, 0.0),
            attributes={"size": Gf.Vec2f(100.0, 100.0)},
        )
    else:
        if not prim_utils.is_prim_path_valid(f"{ground_prim}/Environment"):
            prim_utils.create_prim(
                f"{ground_prim}/Plane",
                "Plane",
                translation=(0.0, 0.0, 0.0),
                attributes={"size": Gf.Vec2f(100.0, 100.0)},
            )

    # Teleop env + config
    cfg = TeleopEnvCfg()
    teleop_env = TeleopEnv(cfg)

    # Load robot, perform an initial reset to initialize PhysX handles, and bind DOFs
    robot = create_robot_articulation(sim, cfg)
    sim.reset()
    teleop_env.bind_robot_dofs(robot.joint_names)

    # Differential IK wrapper
    ik_helper = TeleopIK(
        TeleopIKConfig(
            device=cfg.sim_device,
            ee_link_name=cfg.robot.ee_link_name,
            base_link_name=cfg.robot.base_link_name,
            arm_dof_names=cfg.robot.arm_dof_names,
            damping=cfg.ik.damping,
        ),
        articulation=robot,
    )

    try:
        sim.reset()
        # Drive the main loop using the SimulationApp API
        while simulation_app.is_running():
            sim.step()

            cmd = teleop_env.advance()
            if cmd is None:
                continue

            # Solve for next arm joint targets
            arm_targets = ik_helper.solve(cmd.ee_pose)

            # Assemble full action vector (arm + hand)
            action_np = teleop_env.build_action(arm_targets, cmd, robot_dof_names=robot.joint_names)
            action = torch.tensor(action_np, dtype=torch.float32, device=cfg.sim_device)

            # Apply positional targets
            robot.set_joint_position_target(action)
            robot.write_data_to_sim()
    finally:
        # Close the SimulationApp via AppLauncher handle
        simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    main(args)
