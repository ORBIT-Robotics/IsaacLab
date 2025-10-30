from isaaclab.devices import OpenXRDevice
from isaaclab.devices.openxr.retargeters import Se3AbsRetargeter
from isaaclab.devices.openxr.retargeters.orca.orca_retargeter import OrcaRetargeter

from orbit_xr.tasks.teleop_env_cfg import TeleopDeviceCfg


def create_openxr_device(cfg: TeleopDeviceCfg) -> OpenXRDevice:
    """Construct an OpenXR device with the configured retargeters attached."""
    retargeters = [
        Se3AbsRetargeter(cfg.ee_retargeter),
        OrcaRetargeter(cfg.hand_retargeter),
    ]
    return OpenXRDevice(cfg=cfg.xr, retargeters=retargeters)
