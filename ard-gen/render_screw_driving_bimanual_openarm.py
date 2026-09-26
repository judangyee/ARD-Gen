"""assets/screw_driving_bimanual_openarm.xml 롤아웃을 mp4로 렌더링한다.

render_peg_in_hole_bimanual_openarm.py와 같은 구조.

사용 예:
    MUJOCO_GL=osmesa python render_screw_driving_bimanual_openarm.py --out-dir .
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco
import numpy as np

from sim.screw_driving_bimanual_openarm_sim import (
    MAX_CONTROL_STEPS,
    TARGET_DEPTH,
    ScrewDrivingBimanualOpenArmSim,
    _default_scene_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--cameras", type=str, default="wide_cam,top_cam")
    parser.add_argument("--torque-limit", type=float, default=1.79)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--freeze-frames", type=int, default=20)
    parser.add_argument("--capture-every", type=int, default=3, help="몇 제어 틱마다 프레임을 캡처할지 (영상 길이 조절용)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gains = {"torque_limit": args.torque_limit}
    cfg = _default_scene_config()
    print(f"[render] gains={gains}")

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]

    sim = ScrewDrivingBimanualOpenArmSim()
    renderers = {name: mujoco.Renderer(sim.model, height=args.height, width=args.width) for name in camera_names}

    sim.reset(cfg)
    target_depth = float(cfg["target_depth"])

    frames_by_cam = {name: [] for name in camera_names}

    def capture() -> None:
        for name, renderer in renderers.items():
            renderer.update_scene(sim.data, camera=name)
            frames_by_cam[name].append(renderer.render().copy())

    capture()

    success = False
    insertion_depth = 0.0
    max_torque = 0.0

    for step in range(1, MAX_CONTROL_STEPS + 1):
        info = sim.step(gains)
        if step % args.capture_every == 0:
            capture()

        max_torque = max(max_torque, abs(info["torque"]))
        insertion_depth = sim.get_insertion_depth()

        if insertion_depth >= target_depth * 0.99:
            success = True
            break

    os.makedirs(args.out_dir, exist_ok=True)
    for name, frames in frames_by_cam.items():
        for _ in range(args.freeze_frames):
            frames.append(frames[-1])
        out_path = os.path.join(args.out_dir, f"screw_driving_openarm_{name}.mp4")
        imageio.mimsave(out_path, frames, fps=args.fps, quality=8)
        print(f"[render] {name}: {len(frames)} frames -> {out_path}")

    print(
        f"[render] steps={step} success={success} insertion_depth_mm={insertion_depth * 1000:.2f} "
        f"max_torque={max_torque:.3f}"
    )
    for renderer in renderers.values():
        renderer.close()


if __name__ == "__main__":
    main()
