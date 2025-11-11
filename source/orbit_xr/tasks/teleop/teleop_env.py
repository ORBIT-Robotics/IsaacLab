from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from isaaclab.devices import DeviceBase

from orbit_xr.devices.retargeter import create_openxr_device
from orbit_xr.tasks.teleop.teleop_env_cfg import TeleopEnvCfg


from isaacsim.core.utils import prims as prim_utils
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom



@dataclass
class TeleopCommand:
    ee_pose: np.ndarray
    right_wrist_pose: np.ndarray
    hand_joints: np.ndarray
    raw: torch.Tensor


class TeleopEnv:
    def __init__(self, cfg: TeleopEnvCfg):
        self.cfg = cfg
        self._device: DeviceBase = create_openxr_device(cfg.device)

        self._arm_dof_names = tuple(cfg.robot.arm_dof_names)
        hand_names = cfg.robot.hand_dof_names
        if not hand_names:
            hand_names = cfg.device.hand_retargeter.hand_joint_names or []
        self._hand_dof_names = tuple(hand_names)
        if len(self._hand_dof_names) == 0:
            raise ValueError(
                "No hand DOF names provided. Set TeleopEnvCfg.robot.hand_dof_names "
                "or the hand retargeter configuration."
            )

        self._robot_dof_names: tuple[str, ...] | None = None
        self._arm_indices: list[int] = []
        self._hand_indices: list[int] = []
        
        #target for interactive ik testing
        self.target = None

    @property
    def device(self) -> DeviceBase:
        return self._device

    def reset(self) -> None:
        self._device.reset()

    def bind_robot_dofs(self, dof_names: Sequence[str]) -> None:
        self._robot_dof_names = tuple(dof_names)

        if self._robot_dof_names is None:
            raise RuntimeError("Robot DOF names must be bound before computing indices.")
        robot_dofs: tuple[str, ...] = self._robot_dof_names

        def _compute_indices(requested: Sequence[str], robot_dofs_seq: Sequence[str]) -> list[int]:
            missing = [name for name in requested if name not in robot_dofs_seq]
            if missing:
                raise ValueError(f"Missing DOFs in robot model: {missing}")
            return [robot_dofs_seq.index(name) for name in requested]

        if self._arm_dof_names:
            self._arm_indices = _compute_indices(self._arm_dof_names, list(robot_dofs))
        else:
            self._arm_indices = []
        self._hand_indices = _compute_indices(self._hand_dof_names, list(robot_dofs))

    def advance(self) -> TeleopCommand | None:
        raw = self._device.advance()
        if raw is None:
            return None

        if raw.dim() > 1:
            raw = raw.reshape(-1)

        raw_cpu = raw.detach().cpu()
        ee_pose = raw_cpu[:7].numpy()
        right_wrist_pose = raw_cpu[7:14].numpy()
        hand_joints = raw_cpu[14:].numpy()

        return TeleopCommand(
            ee_pose=ee_pose,
            right_wrist_pose=right_wrist_pose,
            hand_joints=hand_joints,
            raw=raw,
        )

    def build_action(
        self,
        arm_joint_targets: np.ndarray,
        teleop_cmd: TeleopCommand,
        robot_dof_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        if robot_dof_names is not None and (
            self._robot_dof_names is None or tuple(robot_dof_names) != self._robot_dof_names
        ):
            self.bind_robot_dofs(robot_dof_names)

        if self._robot_dof_names is None:
            raise RuntimeError("Call bind_robot_dofs() before building actions.")

        if self._arm_indices and arm_joint_targets.shape[0] != len(self._arm_indices):
            raise ValueError(
                f"Expected {len(self._arm_indices)} arm joint targets but received "
                f"{arm_joint_targets.shape[0]} entries."
            )

        action = np.zeros(len(self._robot_dof_names), dtype=np.float32)
        if self._arm_indices:
            action[self._arm_indices] = arm_joint_targets

        hand_target = teleop_cmd.hand_joints[: len(self._hand_indices)]
        if hand_target.shape[0] != len(self._hand_indices):
            raise ValueError(
                f"Hand command has {hand_target.shape[0]} values but "
                f"{len(self._hand_indices)} DOFs are bound."
            )
        action[self._hand_indices] = hand_target
        return action
