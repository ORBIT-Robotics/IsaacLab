from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Tuple

from networkx import radius

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
    ee_link_name: str = "right_palm"
    base_link_name: str = "base_link"
    arm_dof_names: List[str] = field(
        default_factory=lambda: [
            "Revolute_1",
            "Revolute_3",
            "Revolute_9",
            "Revolute_20",
            "Revolute_26",
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
class TeleopInteractionCfg:
    mode: Literal["cloudxr", "interactive_ik"] = "cloudxr"
    target_prim_path: str = "/World/Targets/ee_goal"    
    target_prim_radius: float = 0.03
    target_color: Tuple[float, float, float] = (0.9, 0.1, 0.1)
    

@dataclass
class TeleopEnvCfg:
    robot: TeleopRobotCfg = field(default_factory=TeleopRobotCfg)
    device: TeleopDeviceCfg = field(default_factory=TeleopDeviceCfg)
    ik: TeleopIKCfg = field(default_factory=TeleopIKCfg)
    sim_device: str = "cuda:0"
    interaction: TeleopInteractionCfg = field(default_factory=TeleopInteractionCfg)


