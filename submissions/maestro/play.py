#!/usr/bin/env python3
"""Load scene.xml and open the interactive MuJoCo viewer.

Drag the actuator sliders in the viewer's "Control" panel
(left side, expand the Control group) to manually flex the
fingers and press the piano keys. The keys spring back up on
release and their touch sensors report contact force.
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
    # Rest the unused pinky (f4) in a slightly curled, relaxed pose so it
    # looks natural on startup. Its position actuators hold this until you
    # drag the f4 sliders yourself.
    for j, target in zip((1, 2, 3), (-0.35, -0.60, -0.60)):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"f4_j{j}_act")
        data.ctrl[aid] = target

    print("\nOpening viewer. Use the Control panel to actuate fingers.\n")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
