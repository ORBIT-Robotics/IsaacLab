from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg as ArticulationCfgType

import torch

from orbit_xr.tasks.teleop_env_cfg import TeleopEnvCfg
from orbit_xr.tasks.teleop_env import TeleopEnv
from orbit_xr.tasks.teleop_ik import TeleopIK, TeleopIKConfig


def create_robot_articulation(sim: Any, cfg: TeleopEnvCfg) -> Articulation:
    """
    Placeholder loader: replace this with your actual robot-loading code.

    Return an isaaclab.assets.articulation.Articulation positioned and ready in the scene.
    The 'sim' type is intentionally Any to avoid SimulationContext type mismatches across packages.
    """
    assets_root = Path(__file__).resolve().parents[3] / "isaaclab_assets" / "data" / "robot" / "icarus" / "unimanual"
    usd_path = assets_root / "usd" / "icarus.usd"

    robot_prim_path = "/World/ICARUS"

    spawn_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    spawn_cfg.func(robot_prim_path, spawn_cfg)

    robot_cfg = ArticulationCfgType(
        prim_path=robot_prim_path,
        spawn=None,
        init_state=ArticulationCfgType.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
    )

    return Articulation(cfg=robot_cfg)


def main(args: argparse.Namespace) -> None:
    # Launch the app using IsaacLab's AppLauncher (wraps isaacsim.SimulationApp)
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    # IsaacLab SimulationContext
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Teleop env + config
    cfg = TeleopEnvCfg()
    teleop_env = TeleopEnv(cfg)

    # Load robot and bind DOF order for arm/hand indices
    robot = create_robot_articulation(sim, cfg)
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
    parser.add_argument("--headless", action="store_true")
    main(parser.parse_args())
