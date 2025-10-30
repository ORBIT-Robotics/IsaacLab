from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from isaaclab.devices import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import Se3AbsRetargeterCfg
from isaaclab.devices.openxr.retargeters.orca.orca_retargeter import OrcaRetargeterCfg

DEFAULT_ORCA_HAND_JOINTS: List[str] = [
    "right_thumb_mcp",
    "right_thumb_abd",
    "right_thumb_pip",
    "right_thumb_dip",
    "right_index_abd",
    "right_index_mcp",
    "right_index_pip",
    "right_middle_abd",
    "right_middle_mcp",
    "right_middle_pip",
    "right_ring_abd",
    "right_ring_mcp",
    "right_ring_pip",
    "right_pinky_abd",
    "right_pinky_mcp",
    "right_pinky_pip",
]


@dataclass
class TeleopRobotCfg:
    ee_link_name: str = "right_hand_joint"
    base_link_name: str = "base_link"
    arm_dof_names: List[str] = field(
        default_factory=lambda: [
            "Revolute 1",
            "Revolute 3",
            "Revolute 9",
            "Revolute 20",
            "Revolute 26",
            "right_wrist",
        ]
    )
    hand_dof_names: List[str] = field(default_factory=lambda: list(DEFAULT_ORCA_HAND_JOINTS))


@dataclass
class TeleopIKCfg:
    damping: float = 0.05
    max_iterations: int = 20
    max_joint_step: float = 0.05


@dataclass
class TeleopDeviceCfg:
    xr: OpenXRDeviceCfg = field(default_factory=OpenXRDeviceCfg)
    ee_retargeter: Se3AbsRetargeterCfg = field(
        default_factory=lambda: Se3AbsRetargeterCfg(
            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_position=True,
            enable_visualization=False,
        )
    )
    hand_retargeter: OrcaRetargeterCfg = field(
        default_factory=lambda: OrcaRetargeterCfg(
            enable_visualization=False,
            num_open_xr_hand_joints=21,
            hand_joint_names=list(DEFAULT_ORCA_HAND_JOINTS),
        )
    )


@dataclass
class TeleopEnvCfg:
    robot: TeleopRobotCfg = field(default_factory=TeleopRobotCfg)
    device: TeleopDeviceCfg = field(default_factory=TeleopDeviceCfg)
    ik: TeleopIKCfg = field(default_factory=TeleopIKCfg)
    sim_device: str = "cuda:0"
