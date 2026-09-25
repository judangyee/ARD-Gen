"""assets/peg_in_hole_bimanual_openarm.xml 롤아웃을 mp4로 렌더링한다.

render_bimanual.py와 같은 구조. **아직 삽입이 성공하지 않는다**
(sim/peg_in_hole_bimanual_openarm_sim.py 모듈 docstring의 "현재 상태" 참고)
-- 이 스크립트는 그 실패 양상(오른팔 진동, peg가 hole에서 xy로 벗어나는 것)을
눈으로 보기 위한 것이다.

사용 예:
    MUJOCO_GL=osmesa python render_peg_in_hole_bimanual_openarm.py --out-dir .
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco
import numpy as np

from sim.peg_in_hole_bimanual_openarm_sim import (
    DT,
    MAX_STEPS,
    BimanualPegInHoleOpenArmSim,
    _default_scene_config,
    adaptive_z_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--cameras", type=str, default="wide_cam,top_cam")
    parser.add_argument("--kp-xy", type=float, default=0.000515)
    parser.add_argument("--kd-xy", type=float, default=2.4e-05)
    parser.add_argument("--offset-x", type=float, default=0.014)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--freeze-frames", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gains = {"Kp_xy": args.kp_xy, "Kd_xy": args.kd_xy}
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = (args.offset_x, args.offset_y)
    print(f"[render] gains={gains} offset=({args.offset_x},{args.offset_y})")

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]

    sim = BimanualPegInHoleOpenArmSim()
    renderers = {name: mujoco.Renderer(sim.model, height=args.height, width=args.width) for name in camera_names}

    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    frames_by_cam = {name: [] for name in camera_names}

    def capture() -> None:
        for name, renderer in renderers.items():
            renderer.update_scene(sim.data, camera=name)
            frames_by_cam[name].append(renderer.render().copy())

    capture()

    kp_xy, kd_xy = gains["Kp_xy"], gains["Kd_xy"]
    prev_force_error_xy = np.zeros(2)
    success = False
    insertion_depth = 0.0
    max_force = 0.0
    max_xy_drift = 0.0

    for step in range(1, MAX_STEPS + 1):
        force, _torque = sim.get_force_torque()
        force_error_xy = force[:2]
        d_force_error_xy = (force_error_xy - prev_force_error_xy) / DT
        prev_force_error_xy = force_error_xy
        delta_xy = -kp_xy * force_error_xy - kd_xy * d_force_error_xy

        cur_peg_tip = sim.get_peg_tip_pos()
        cur_hole_center = sim.get_hole_center_pos()
        cur_dx = float(cur_hole_center[0] - cur_peg_tip[0])
        cur_dy = float(cur_hole_center[1] - cur_peg_tip[1])
        z_rate = adaptive_z_rate(cur_dx, cur_dy, outer_half)
        delta = np.array([delta_xy[0], delta_xy[1], -z_rate])

        sim.step(delta)
        capture()

        max_force = max(max_force, float(np.linalg.norm(force)))

        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        max_xy_drift = max(max_xy_drift, (dx**2 + dy**2) ** 0.5)
        xy_ok = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = max(0.0, float(hole_center[2] - peg_tip[2]))
        insertion_depth = raw_depth if xy_ok else 0.0
        if insertion_depth >= target_depth:
            success = True
            break

    os.makedirs(args.out_dir, exist_ok=True)
    for name, frames in frames_by_cam.items():
        for _ in range(args.freeze_frames):
            frames.append(frames[-1])
        out_path = os.path.join(args.out_dir, f"peg_in_hole_openarm_{name}.mp4")
        imageio.mimsave(out_path, frames, fps=args.fps, quality=8)
        print(f"[render] {name}: {len(frames)} frames -> {out_path}")

    print(
        f"[render] steps={step} success={success} insertion_depth={insertion_depth:.4f} "
        f"max_xy_drift_mm={max_xy_drift * 1000:.2f} max_force={max_force:.2f}"
    )
    for renderer in renderers.values():
        renderer.close()


if __name__ == "__main__":
    main()
