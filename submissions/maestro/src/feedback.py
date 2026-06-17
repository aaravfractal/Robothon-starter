#!/usr/bin/env python3
"""Touch-sensor feedback for the closed-loop piano controller.

Wraps the MuJoCo touch sensors so the controller can ask, each timestep,
"has this key been pressed yet?" without knowing how sensors are laid out
in `data.sensordata`. This is the feedback half of the closed loop: the
state machine only advances to the next note once `is_pressed()` returns
True for the current key.
"""
import mujoco


class TouchFeedback:
    """Reads scalar `touch` sensors by name (addresses cached on first use)."""

    def __init__(self, model):
        self.model = model
        self._adr = {}   # sensor name -> index into data.sensordata

    def _address(self, sensor_name):
        adr = self._adr.get(sensor_name)
        if adr is None:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR,
                                    sensor_name)
            if sid < 0:
                raise KeyError(f"no sensor named {sensor_name!r} in model")
            adr = int(self.model.sensor_adr[sid])
            self._adr[sensor_name] = adr
        return adr

    def force(self, data, sensor_name):
        """Current normal-force reading (N) of the named touch sensor."""
        return float(data.sensordata[self._address(sensor_name)])

    def is_pressed(self, data, sensor_name, threshold):
        """True once the sensor force meets/exceeds `threshold` (N)."""
        return self.force(data, sensor_name) >= threshold

    def is_released(self, data, sensor_name, threshold):
        """True once contact force has dropped back below `threshold` (N)."""
        return self.force(data, sensor_name) < threshold
