from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg


@dataclass
class TeleopIKConfig:
    device: str
    ee_link_name: str
    base_link_name: str
    arm_dof_names: Sequence[str]
    damping: float


class TeleopIK:
    def __init__(self, cfg: TeleopIKConfig, articulation: Articulation):
        self._articulation = articulation
        self._device = cfg.device

        if len(cfg.arm_dof_names) == 0:
            raise ValueError("TeleopIK requires at least one arm joint name.")

        ee_ids, ee_names = self._articulation.find_bodies([cfg.ee_link_name])
        if len(ee_ids) != 1:
            raise ValueError(
                f"Expected one match for EE link '{cfg.ee_link_name}', "
                f"found {len(ee_ids)} ({ee_names})."
            )
        self._ee_body_id = ee_ids[0]
        self._is_fixed_base = self._articulation.is_fixed_base
        self._jacobian_body_id = self._ee_body_id - 1 if self._is_fixed_base else self._ee_body_id

        joint_ids, joint_names = self._articulation.find_joints(list(cfg.arm_dof_names))
        if len(joint_ids) != len(cfg.arm_dof_names):
            raise ValueError(
                "Mismatch between requested arm DOFs and articulation joints. "
                f"Requested: {cfg.arm_dof_names}, resolved: {joint_names}"
            )
        self._arm_joint_ids = joint_ids

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": cfg.damping},
        )
        self._controller = DifferentialIKController(ik_cfg, num_envs=1, device=self._device)

    def solve(self, target_pose: np.ndarray) -> np.ndarray:
        if target_pose.shape[0] != 7:
            raise ValueError("Target pose must be length 7 (x, y, z, qw, qx, qy, qz).")

        root_pose_w = self._articulation.data.root_pose_w
        root_pos_w = root_pose_w[:, 0:3]
        root_quat_w = root_pose_w[:, 3:7]

        ee_pose_w = self._articulation.data.body_pose_w[:, self._ee_body_id]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        jacobian_w = self._articulation.root_physx_view.get_jacobians()[
            :, self._jacobian_body_id, :, self._arm_joint_ids
        ]
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_b[:, 3:, :])

        joint_pos = self._articulation.data.joint_pos[:, self._arm_joint_ids]

        target_tensor = torch.tensor(target_pose, dtype=torch.float32, device=self._device).unsqueeze(0)
        self._controller.set_command(target_tensor)
        joint_targets = self._controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
        return joint_targets[0].detach().cpu().numpy()

