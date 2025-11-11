from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import get_current_stage

# Absolute path to the repository's `source` directory
REPO_SOURCE_DIR = Path(__file__).resolve().parents[3]
IKARUS_ASSET_ROOT = REPO_SOURCE_DIR / "isaaclab_assets" / "data" / "robot" / "icarus" / "unimanual"
IKARUS_USD_PATH = IKARUS_ASSET_ROOT / "usd" / "icarus.usd"
IKARUS_URDF_PATH = IKARUS_ASSET_ROOT / "urdf" / "icarus.urdf"

IKARUS_ARM_DOF_NAMES: tuple[str, ...] = (
    "Revolute_1",
    "Revolute_3",
    "Revolute_9",
    "Revolute_20",
    "Revolute_26",
    "right_wrist",
)

IKARUS_HAND_DOF_NAMES: tuple[str, ...] = (
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
)

IKARUS_EE_LINK = "right_palm"
IKARUS_BASE_LINK = "base_link"


@dataclass
class FollowTargetCfg:
    prim_path: str = "/World/Targets/ikarus_goal"
    size: tuple[float, float, float] = (0.05, 0.05, 0.05)
    color: tuple[float, float, float] = (1.0, 0.2, 0.2)
    initial_position: tuple[float, float, float] = (0.6, 0.0, 0.9)
    initial_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation is expressed as (w, x, y, z)."""


class FollowTargetTask:
    """Utility that spawns a colored cube and exposes its pose for IK targets."""

    def __init__(self, cfg: FollowTargetCfg | None = None) -> None:
        self.cfg = cfg or FollowTargetCfg()
        self._view: XFormPrim | None = None

    def spawn(self) -> None:
        """Create the target cube if it does not yet exist."""
        if self._view is not None:
            return

        material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=self.cfg.color,
            emissive_color=(0.3, 0.05, 0.05),
            roughness=0.2,
            metallic=0.05,
            opacity=1.0,
        )
        # Spawn as a visual-only cube (no rigid-body schema) so it can be
        # moved interactively in the viewport using the transform gizmo.
        target_cfg = sim_utils.CuboidCfg(
            size=self.cfg.size,
            visual_material=material_cfg,
        )
        # Spawn without fabric cloning so viewport hide/delete work as expected
        target_cfg.func(
            self.cfg.prim_path,
            target_cfg,
            translation=self.cfg.initial_position,
            orientation=self.cfg.initial_orientation,
            clone_in_fabric=False,
        )
        self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
        self._view.initialize()

    def get_target_pose(self) -> np.ndarray:
        """Return the current world pose of the cube as [x, y, z, qw, qx, qy, qz]."""
        if self._view is None:
            raise RuntimeError("FollowTargetTask.spawn() must be called before querying poses.")
        positions, orientations = self._view.get_world_poses()
        # XFormPrim returns torch tensors on the sim device (often CUDA). Bring to CPU for numpy.
        pos = positions[0]
        quat = orientations[0]
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        if isinstance(quat, torch.Tensor):
            quat = quat.detach().cpu().numpy()
        pos_np = np.asarray(pos, dtype=np.float32)
        quat_np = np.asarray(quat, dtype=np.float32)
        return np.concatenate([pos_np, quat_np]).astype(np.float32)


def create_ikarus_articulation(
    prim_path: str | None = None,
    arm_dof_names: Sequence[str] = IKARUS_ARM_DOF_NAMES,
    hand_dof_names: Sequence[str] = IKARUS_HAND_DOF_NAMES,
) -> Articulation:
    """Load the Ikarus USD as a sublayer so its internal absolute paths stay valid.

    The asset's prims live at absolute paths (e.g., "/icarus_orca/..."). Referencing it under a different
    prim can break those relationships. Sublayering preserves the original absolute paths in the composed stage.
    """

    usd_path = IKARUS_USD_PATH
    if not usd_path.is_file():
        raise FileNotFoundError(f"Ikarus USD was not found at {usd_path}")

    # Add the USD as a sublayer if not already present
    stage = get_current_stage()
    root_layer = stage.GetRootLayer()
    if str(usd_path) not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(str(usd_path))

    def _pattern(name: str) -> str:
        return f"^{name.replace(' ', '_')}$"

    arm_joint_patterns = [_pattern(name) for name in arm_dof_names]
    hand_joint_patterns = [_pattern(name) for name in hand_dof_names]

    # Resolve robot prim path inside the composed stage
    if prim_path is None:
        candidates = [
            "/icarus_orca",
            "/Root/icarus_orca",
            "/ICARUS/icarus_orca",
        ]
        found_path: str | None = None
        for p in candidates:
            if stage.GetPrimAtPath(p).IsValid():
                found_path = p
                break
        if found_path is None:
            # Help the user by listing top-level prims
            root_children = [c.GetPath().pathString for c in stage.GetPseudoRoot().GetChildren()]
            raise RuntimeError(
                "Could not locate Ikarus prim. Tried: "
                f"{candidates}. Top-level prims: {root_children}. Make sure the USD has an 'icarus_orca' Xform."
            )
        prim_path = found_path

    robot_cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
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
    robot = Articulation(cfg=robot_cfg)

    # Ensure the robot cannot fall: fix the root link and disable gravity on all links.
    try:
        sim_utils.modify_articulation_root_properties(
            prim_path,
            sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
        )
        sim_utils.modify_rigid_body_properties(
            prim_path,
            sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        )
    except Exception:
        # Best-effort: properties depend on USD schemas present in the asset.
        pass

    return robot


__all__ = [
    "FollowTargetCfg",
    "FollowTargetTask",
    "IKARUS_ASSET_ROOT",
    "IKARUS_USD_PATH",
    "IKARUS_URDF_PATH",
    "IKARUS_ARM_DOF_NAMES",
    "IKARUS_HAND_DOF_NAMES",
    "IKARUS_EE_LINK",
    "IKARUS_BASE_LINK",
    "create_ikarus_articulation",
]
