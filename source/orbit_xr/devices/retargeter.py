from isaaclab.devices import OpenXRDevice, DeviceBase
import isaaclab.devices.openxr.openxr_device as _openxr_mod
from isaaclab.devices.openxr.retargeters import Se3AbsRetargeter
from isaaclab.devices.openxr.retargeters.orca.orca_retargeter import OrcaRetargeter

from orbit_xr.tasks.teleop_env_cfg import TeleopDeviceCfg


class _NoOpDevice(DeviceBase):
    """Fallback device when OpenXR is unavailable. Returns no commands."""

    def reset(self) -> None:  # noqa: D401
        pass

    def advance(self):  # noqa: D401
        return None


def create_openxr_device(cfg: TeleopDeviceCfg) -> DeviceBase:
    """Construct an OpenXR device with configured retargeters, with safe fallback.

    If OpenXR runtime/extensions are unavailable, returns a no-op device so the
    simulation can still run (ground + robot load) without crashing.
    """
    # Detect XR core availability without raising. When the XR extension is not
    # loaded, the OpenXRDevice module leaves XRCore as None.
    if getattr(_openxr_mod, "XRCore", None) is None:
        import omni.log

        omni.log.warn("OpenXR runtime/extensions not available; using NoOp device.")
        return _NoOpDevice(retargeters=None)

    retargeters = [Se3AbsRetargeter(cfg.ee_retargeter), OrcaRetargeter(cfg.hand_retargeter)]
    try:
        return OpenXRDevice(cfg=cfg.xr, retargeters=retargeters)
    except Exception as exc:  # pragma: no cover
        import omni.log

        omni.log.warn(f"OpenXR unavailable, using NoOp device. Reason: {exc}")
        return _NoOpDevice(retargeters=None)
