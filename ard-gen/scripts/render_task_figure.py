"""Render Figure 3 (task keyframe montage) for the ARD-VLA paper.

Runs one real, deterministic, successful OpenArm bimanual peg-in-hole rollout
(same CMA-ES-tuned gains as tasks/peg_in_hole.yaml's first eval scenario,
reproduced in render_peg_in_hole_bimanual_openarm.py) and picks 4 keyframes
from it: approach -> left arm stabilizes the part -> right arm peg contact ->
insertion complete.

Keyframes are auto-selected from real rollout state, not hand-picked:
  - "Right arm: peg contact" = the first step where insertion_depth > 0 (the
    same xy_ok/raw_depth logic render_peg_in_hole_bimanual_openarm.py uses).
  - "Insertion complete" = the step where the episode's own success condition
    fires (hold_count >= _SUCCESS_HOLD_STEPS).
  - "Approach" / "Left arm stabilizes part" are fixed fractions (1/3, 2/3) of
    the steps leading up to first contact -- the only two stages without a
    distinct state-change event to key off (the left arm is held in place by
    strong position gains from frame 0, so there is no separate "grasp" event
    to detect; see module docstring of peg_in_hole_bimanual_openarm_sim.py).
    The peg does move a real, verified ~1.6cm between these two steps (it's
    continuously being driven toward the hole from frame 1) -- it just isn't
    visible at the original camera's zoomed-out framing, which is why this
    revision crops tightly around the work area (see compute_crop_box).

No episode dataset exists for this task (a fresh deterministic rollout is run
on demand) and this scenario samples no randomness, so there is no "episode
number" or RNG seed to report -- the reproducible identity is the
(gains, peg_init_offset_xy) tuple printed at the end of this script's run,
which exactly reproduces tasks/peg_in_hole.yaml's first eval_scenario.

Left/right arm identity (verified against assets/peg_in_hole_bimanual_openarm.xml
and sim/peg_in_hole_bimanual_openarm_sim.py, not assumed):
  - <weld body1="openarm_left_ee_base_link" body2="hole_socket">: the LEFT arm
    grasps the part with the hole and holds it fixed (position-controlled,
    _LEFT_ARM_HOME_QPOS) for the whole episode -- this is the Stabilizer.
  - <weld body1="openarm_right_ee_base_link" body2="peg">: the RIGHT arm
    carries the peg and is actively torque-driven to insert it -- the Actuator.
  This matches the paper's left=Stabilizer/right=Actuator convention. BUT the
  physical left/right arm does NOT map onto screen-left/screen-right in the
  `wide_cam` view: projecting both end-effectors with the camera's own
  pos/forward/up (see project_world_to_px) shows the physically-left arm
  (Stabilizer) renders further toward screen-right than the physically-right
  arm (Actuator) -- i.e. the camera angle mirrors them. This is exactly why
  role labels are now colored rings drawn AT each gripper instead of
  "left arm"/"right arm" text: text naming screen sides would be wrong here.

Cap-twisting has not been ported to OpenArm yet (sim/cap_twist_env.py is the
older vx300s rig) -- this script only renders the peg-in-hole row. TASK_ROWS
below is the extension point for a future cap_twist row.

ponytail: do not import torch in this script/process -- OSMesa (or EGL) and
torch in the same process segfaults in this sandbox; this script only needs
mujoco/numpy/matplotlib/PIL.

Usage:
    MUJOCO_GL=egl python scripts/render_task_figure.py --out-dir paper/figures
    # falls back to: MUJOCO_GL=osmesa python scripts/render_task_figure.py ...
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # prefer EGL per project convention; override via env

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from sim.peg_in_hole_bimanual_openarm_sim import (
    DT,
    MAX_STEPS,
    BimanualPegInHoleOpenArmSim,
    _default_scene_config,
    adaptive_z_rate,
    _MAX_SANE_DEPTH_M,
    _SUCCESS_HOLD_STEPS,
)

STABILIZER_COLOR = "#0072B2"  # left arm -- must match paper/figures/fig2_architecture.* and fig1
ACTUATOR_COLOR = "#E69F00"  # right arm

TASK_ROWS = ["peg_in_hole"]  # extend with "cap_twist" once it's ported to OpenArm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=str, default="paper/figures")
    p.add_argument("--camera", type=str, default="wide_cam")
    p.add_argument("--kp-xy", type=float, default=0.124630)
    p.add_argument("--kd-xy", type=float, default=0.001125)
    p.add_argument("--offset-x", type=float, default=0.014)
    p.add_argument("--offset-y", type=float, default=0.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args()


class Projector:
    """Projects world points to pixel coords for a fixed MuJoCo camera, using the
    same pos/forward/up/fovy the renderer itself used. Validated empirically by
    projecting peg_tip/hole_center and overlaying on a render (they land exactly
    on the peg and the hole, see git history / PR description for the check)."""

    def __init__(self, renderer: "mujoco.Renderer", width: int, height: int):
        cam = renderer.scene.camera[0]
        self.pos = np.array(cam.pos, dtype=np.float64)
        forward = np.array(cam.forward, dtype=np.float64)
        self.forward = forward / np.linalg.norm(forward)
        up = np.array(cam.up, dtype=np.float64)
        up = up / np.linalg.norm(up)
        self.right = np.cross(self.forward, up)
        self.right /= np.linalg.norm(self.right)
        self.up2 = np.cross(self.right, self.forward)
        self.width, self.height = width, height

    def set_fovy(self, fovy_deg: float) -> None:
        self.f = (self.height / 2) / np.tan(np.radians(fovy_deg) / 2)

    def project(self, world_pos: np.ndarray) -> tuple[float, float]:
        rel = np.asarray(world_pos) - self.pos
        x_cam, y_cam, z_cam = np.dot(rel, self.right), np.dot(rel, self.up2), np.dot(rel, self.forward)
        px = self.width / 2 + self.f * (x_cam / z_cam)
        py = self.height / 2 - self.f * (y_cam / z_cam)
        return px, py


def run_and_pick_keyframes(args: argparse.Namespace) -> tuple[list[np.ndarray], list[tuple[str, int]], dict, Projector]:
    gains = {"Kp_xy": args.kp_xy, "Kd_xy": args.kd_xy}
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = (args.offset_x, args.offset_y)

    sim = BimanualPegInHoleOpenArmSim()
    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    def capture() -> np.ndarray:
        renderer.update_scene(sim.data, camera=args.camera)
        return renderer.render().copy()

    frames = [capture()]  # index 0 = initial reset pose
    cam_id = sim.model.camera(args.camera).id
    projector = Projector(renderer, args.width, args.height)
    projector.set_fovy(float(sim.model.cam_fovy[cam_id]))

    def ee_positions() -> dict[str, np.ndarray]:
        return {
            "left_ee": sim.data.xpos[sim._left_ee_body_id].copy(),
            "right_ee": sim.data.xpos[sim._right_ee_body_id].copy(),
            "peg_tip": sim.get_peg_tip_pos(),
            "hole_center": sim.get_hole_center_pos(),
        }

    positions = [ee_positions()]  # positions[step] matches frames[step]

    kp_xy, kd_xy = gains["Kp_xy"], gains["Kd_xy"]
    prev_dx = prev_dy = 0.0
    hold_count = 0
    first_contact_step: int | None = None
    success_step: int | None = None
    step = 0

    for step in range(1, MAX_STEPS + 1):
        cur_peg_tip = sim.get_peg_tip_pos()
        cur_hole_center = sim.get_hole_center_pos()
        cur_dx = float(cur_hole_center[0] - cur_peg_tip[0])
        cur_dy = float(cur_hole_center[1] - cur_peg_tip[1])
        d_dx = (cur_dx - prev_dx) / DT
        d_dy = (cur_dy - prev_dy) / DT
        prev_dx, prev_dy = cur_dx, cur_dy
        delta_xy = np.array([kp_xy * cur_dx + kd_xy * d_dx, kp_xy * cur_dy + kd_xy * d_dy])

        cur_raw_depth = max(0.0, float(cur_hole_center[2] - cur_peg_tip[2]))
        z_rate = adaptive_z_rate(cur_dx, cur_dy, outer_half, cur_raw_depth, target_depth)
        delta = np.array([delta_xy[0], delta_xy[1], -z_rate])

        sim.step(delta)
        frames.append(capture())
        positions.append(ee_positions())

        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_ok = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))
        insertion_depth = raw_depth if xy_ok else 0.0

        if first_contact_step is None and insertion_depth > 0.0:
            first_contact_step = step

        hold_count = hold_count + 1 if insertion_depth >= target_depth else 0
        if hold_count >= _SUCCESS_HOLD_STEPS:
            success_step = step
            break

    try:
        renderer.close()
    except Exception:
        pass  # EGL throws a harmless teardown error on some drivers; frames are already captured

    if success_step is None:
        raise RuntimeError(
            f"episode did not succeed within {MAX_STEPS} steps (gains={gains}, "
            f"offset=({args.offset_x},{args.offset_y})) -- cannot pick a real 'insertion complete' keyframe"
        )
    if first_contact_step is None:
        raise RuntimeError("episode never registered insertion_depth > 0 -- cannot pick a real contact keyframe")

    approach_step = max(1, first_contact_step // 3)
    stabilize_step = max(approach_step + 1, (2 * first_contact_step) // 3)
    if stabilize_step >= first_contact_step:
        stabilize_step = max(approach_step + 1, first_contact_step - 1)

    keyframes = [
        ("Approach", approach_step),
        ("Left arm stabilizes part", stabilize_step),
        ("Right arm: peg contact", first_contact_step),
        ("Insertion complete", success_step),
    ]
    peg_disp = float(np.linalg.norm(positions[stabilize_step]["peg_tip"] - positions[approach_step]["peg_tip"]))
    meta = dict(
        gains=gains,
        peg_init_offset_xy=(args.offset_x, args.offset_y),
        total_steps=step,
        success=True,
        first_contact_step=first_contact_step,
        success_step=success_step,
        camera=args.camera,
        rng_seed=None,  # deterministic scenario, no randomness sampled -- see module docstring
        peg_displacement_approach_to_stabilize_m=peg_disp,
    )
    return frames, keyframes, meta, projector, positions


def compute_crop_box(keyframes: list[tuple[str, int]], positions: list[dict], projector: Projector,
                      width: int, height: int, margin_frac: float = 0.35) -> tuple[int, int, int, int]:
    """Bounding box (in pixels) around the peg/hole/both end-effectors across every
    keyframe, expanded by margin_frac of its own size, clipped to the frame. Using
    the SAME box for all 4 panels keeps a single consistent framing -- it's a crop
    of one fixed camera, not four different cameras."""
    xs, ys = [], []
    for _, step in keyframes:
        for key in ("left_ee", "right_ee", "peg_tip", "hole_center"):
            px, py = projector.project(positions[step][key])
            xs.append(px)
            ys.append(py)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    mx, my = (x1 - x0) * margin_frac, (y1 - y0) * margin_frac
    x0, x1 = x0 - mx, x1 + mx
    y0, y1 = y0 - my, y1 + my
    # keep it roughly 4:3 so panels don't look stretched
    box_w, box_h = x1 - x0, y1 - y0
    target_ratio = width / height
    if box_w / box_h > target_ratio:
        extra = box_w / target_ratio - box_h
        y0, y1 = y0 - extra / 2, y1 + extra / 2
    else:
        extra = box_h * target_ratio - box_w
        x0, x1 = x0 - extra / 2, x1 + extra / 2
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    return x0, y0, x1, y1


def compose_figure(frames: list[np.ndarray], keyframes: list[tuple[str, int]], positions: list[dict],
                    projector: Projector, out_dir: str) -> None:
    crop = compute_crop_box(keyframes, positions, projector, frames[0].shape[1], frames[0].shape[0])
    cx0, cy0, cx1, cy1 = crop
    cropped = [frames[step][cy0:cy1, cx0:cx1] for _, step in keyframes]

    n = len(keyframes)
    aspect = cropped[0].shape[0] / cropped[0].shape[1]
    fig_w_in = 7.0  # two-column full width (~17.5cm)
    fig, axes = plt.subplots(1, n, figsize=(fig_w_in, fig_w_in / n * aspect))
    for i, ((label, step), ax, img) in enumerate(zip(keyframes, axes, cropped, strict=True)):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#999999")
        ax.set_title(label, fontsize=9, fontfamily="sans-serif")
        if i == 0:
            # role labels as colored rings drawn directly on each arm's gripper (not
            # "left arm"/"right arm" text -- in this camera view the physically-left
            # (Stabilizer) arm renders toward screen-right, so text naming screen
            # sides would mislead; see module docstring).
            for key, color, tag in [("left_ee", STABILIZER_COLOR, "Stabilizer"), ("right_ee", ACTUATOR_COLOR, "Actuator")]:
                px, py = projector.project(positions[step][key])
                px, py = px - cx0, py - cy0
                ax.add_patch(mpatches.Circle((px, py), radius=22, fill=False, edgecolor=color, linewidth=2.5))
                label_dx = 34 if key == "left_ee" else -34
                ha = "left" if key == "left_ee" else "right"
                label_x = px + label_dx
                ax.annotate(tag, xy=(px, py), xytext=(label_x, py - 30), fontsize=6.5, fontweight="bold",
                            color="white", ha=ha, va="center",
                            bbox=dict(facecolor=color, alpha=0.9, pad=1.5, edgecolor="none"),
                            arrowprops=dict(arrowstyle="-", color=color, linewidth=1.2))
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig3_tasks.svg"))
    fig.savefig(os.path.join(out_dir, "fig3_tasks.pdf"))
    # dpi chosen so each cropped panel clears 640px in the saved PNG (confirmation render)
    dpi = max(400, int(640 / (fig_w_in / n)) + 20)
    fig.savefig(os.path.join(out_dir, "fig3_tasks.png"), dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frames, keyframes, meta, projector, positions = run_and_pick_keyframes(args)
    compose_figure(frames, keyframes, positions, projector, args.out_dir)
    print(f"[render_task_figure] episode: gains={meta['gains']} offset_xy={meta['peg_init_offset_xy']} "
          f"camera={meta['camera']} rng_seed={meta['rng_seed']} (deterministic scenario, no sampling)")
    print(f"[render_task_figure] total_steps={meta['total_steps']} first_contact_step={meta['first_contact_step']} "
          f"success_step={meta['success_step']}")
    print(f"[render_task_figure] peg displacement approach->stabilize keyframe: "
          f"{meta['peg_displacement_approach_to_stabilize_m']*100:.2f} cm (verifies the two pre-contact "
          f"keyframes are genuinely different states, not a frozen frame)")
    for label, step in keyframes:
        print(f"[render_task_figure] keyframe '{label}': step {step}")
    print(f"[render_task_figure] wrote fig3_tasks.{{svg,pdf,png}} to {args.out_dir}")


if __name__ == "__main__":
    main()
