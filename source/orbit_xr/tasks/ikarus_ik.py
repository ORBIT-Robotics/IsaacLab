from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

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
    command_type: Literal["position", "pose"] = "position"
    max_step: float = 0.05
    ik_method: Literal["dls", "svd", "trans"] = "dls"
    min_singular_value: float = 1e-5
    low_pass_alpha: float = 0.0  # 0.0 = no filtering, 0.9 = aggressive filtering


class TeleopIK:
    def __init__(self, cfg: TeleopIKConfig, articulation: Articulation):
        self._articulation = articulation
        self._device = cfg.device
        self._cfg = cfg

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
        # Use the EE body index directly for jacobians
        self._jacobian_body_id = self._ee_body_id

        joint_ids, joint_names = self._articulation.find_joints(list(cfg.arm_dof_names))
        if len(joint_ids) != len(cfg.arm_dof_names):
            raise ValueError(
                "Mismatch between requested arm DOFs and articulation joints. "
                f"Requested: {cfg.arm_dof_names}, resolved: {joint_names}"
            )
        self._arm_joint_ids = joint_ids

        # Build IK params based on method
        ik_params = {}
        if cfg.ik_method == "dls":
            ik_params["lambda_val"] = cfg.damping
        elif cfg.ik_method == "svd":
            ik_params["min_singular_value"] = cfg.min_singular_value
        elif cfg.ik_method == "trans":
            ik_params["k_val"] = cfg.damping

        ik_cfg = DifferentialIKControllerCfg(
            command_type=cfg.command_type,
            use_relative_mode=False,
            ik_method=cfg.ik_method,
            ik_params=ik_params,
        )
        self._controller = DifferentialIKController(ik_cfg, num_envs=1, device=self._device)
        # cache clamp tensor
        self._max_step = torch.tensor(self._cfg.max_step, dtype=torch.float32, device=self._device)
        
        # Low-pass filter state: previous joint targets
        self._q_prev = None

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

        # Compute Jacobian condition number for diagnostics
        J = jacobian_b[0, :3, :]  # Position Jacobian only for position control
        try:
            U, S, Vh = torch.linalg.svd(J)
            condition_number = S[0] / (S[-1] + 1e-12)
            # Print every 100 frames to avoid spam
            if not hasattr(self, "_frame_counter"):
                self._frame_counter = 0
            self._frame_counter += 1
            if self._frame_counter % 100 == 0:
                print(f"[TeleopIK] Jacobian condition number: {condition_number.item():.2f} | singular values: {S.cpu().numpy()}", flush=True)
        except Exception:
            pass

        joint_pos = self._articulation.data.joint_pos[:, self._arm_joint_ids]

        # Transform target from world to base frame to match Jacobian frame
        target_pos_w = torch.tensor(target_pose[:3], dtype=torch.float32, device=self._device).unsqueeze(0)
        target_quat_w = torch.tensor(target_pose[3:], dtype=torch.float32, device=self._device).unsqueeze(0)
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )

        if self._cfg.command_type == "position":
            self._controller.set_command(target_pos_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        else:
            target_tensor = torch.cat([target_pos_b, target_quat_b], dim=-1)
            self._controller.set_command(target_tensor)
        q_des = self._controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
        
        # Clamp per-step joint change for stability
        dq = torch.clamp(q_des - joint_pos, min=-self._max_step, max=self._max_step)
        q_next = joint_pos + dq
        
        # Apply low-pass filter if configured
        if self._cfg.low_pass_alpha > 0.0:
            if self._q_prev is None:
                self._q_prev = joint_pos.clone()
            alpha = self._cfg.low_pass_alpha
            q_next = alpha * self._q_prev + (1.0 - alpha) * q_next
            self._q_prev = q_next.clone()
        
        return q_next[0].detach().cpu().numpy()
