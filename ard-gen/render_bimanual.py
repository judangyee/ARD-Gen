"""양팔 peg-in-hole(assets/peg_in_hole_bimanual.xml) 롤아웃을 mp4로 렌더링한다.
render_result.py와 같은 구조지만 seed_trajectory.npz를 읽는 대신 게인을
CLI 인자로 받는다 -- 이 태스크는 아직 CMA-ES 재탐색을 안 했고(오른팔의
기존 게인이 왼팔 관절 강화 후 그대로 통했다, sim/peg_in_hole_bimanual_sim.py
참고), seed 파일이 없다.

카메라:
  - wide_cam: 양팔이 다 보이는 넓은 시점 (world 고정)
  - top_cam:  hole 위에서 내려다보는 뷰 (world 고정, 단일 팔 버전과 동일)

사용 예:
    MUJOCO_GL=osmesa python render_bimanual.py --out-dir . \
        --offset-x 0.0099 --offset-y 0.0099
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco
import numpy as np

from sim.peg_in_hole_bimanual_sim import BimanualPegInHoleSim
from sim.peg_in_hole_sim import DT, MAX_STEPS, Z_RATE, _default_scene_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--cameras", type=str, default="wide_cam,top_cam")
    parser.add_argument("--kp-xy", type=float, default=0.000515)
    parser.add_argument("--kd-xy", type=float, default=2.4e-05)
    parser.add_argument("--offset-x", type=float, default=0.0099)
    parser.add_argument("--offset-y", type=float, default=0.0099)
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
    print(f"[render_bimanual] gains={gains} offset=({args.offset_x},{args.offset_y})")

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]

    sim = BimanualPegInHoleSim()
    renderers = {name: mujoco.Renderer(sim.model, height=args.height, width=args.width) for name in camera_names}

    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    hole_id = sim.model.body("hole_socket").id
    nominal_hole_pos = sim._nominal_hole_pos.copy()

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
    max_hole_drift = 0.0
    max_force = 0.0

    for step in range(1, MAX_STEPS + 1):
        force, _torque = sim.get_force_torque()
        force_error_xy = force[:2]
        d_force_error_xy = (force_error_xy - prev_force_error_xy) / DT
        prev_force_error_xy = force_error_xy
        delta_xy = -kp_xy * force_error_xy - kd_xy * d_force_error_xy
        delta = np.array([delta_xy[0], delta_xy[1], -Z_RATE])

        sim.step(delta)
        capture()

        max_hole_drift = max(max_hole_drift, float(np.linalg.norm(sim.data.xpos[hole_id] - nominal_hole_pos)))
        max_force = max(max_force, float(np.linalg.norm(force)))

        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
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
        out_path = os.path.join(args.out_dir, f"bimanual_{name}.mp4")
        imageio.mimsave(out_path, frames, fps=args.fps, quality=8)
        print(f"[render_bimanual] {name}: {len(frames)} frames -> {out_path}")

    print(
        f"[render_bimanual] steps={step} success={success} insertion_depth={insertion_depth:.4f} "
        f"max_hole_drift_mm={max_hole_drift * 1000:.2f} max_force={max_force:.2f}"
    )
    for renderer in renderers.values():
        renderer.close()


if __name__ == "__main__":
    main()
