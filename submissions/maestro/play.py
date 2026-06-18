#!/usr/bin/env python3
"""Load scene.xml and open the interactive MuJoCo viewer.

Drag the actuator sliders in the viewer's "Control" panel
(left side, expand the Control group) to manually flex the 5
fingers, slide the whole hand sideways on its prismatic wrist
joint (the `wrist_act` slider), and press the 7 piano keys. The
keys spring back up on release and their touch sensors report
contact force.
"""
import os
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.xml")


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    actuators = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    sensors = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        for i in range(model.nsensor)
    ]

    print(f"Loaded {SCENE}")
    print(f"\n{model.nu} actuators (drag these sliders in the viewer):")
    for n in actuators:
        print(f"  - {n}")
    print(f"\n{model.nsensor} sensors:")
    for n in sensors:
        print(f"  - {n}")

    print("\nOpening viewer. Use the Control panel to actuate the fingers and "
          "the wrist slide.\n")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
