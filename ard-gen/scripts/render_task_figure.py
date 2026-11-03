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

No episode dataset exists for this task (a fresh deterministic rollout is run
on demand) and this scenario samples no randomness, so there is no "episode
number" or RNG seed to report -- the reproducible identity is the
(gains, peg_init_offset_xy) tuple printed at the end of this script's run,
which exactly reproduces tasks/peg_in_hole.yaml's first eval_scenario.

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


def run_and_pick_keyframes(args: argparse.Namespace) -> tuple[list[np.ndarray], list[tuple[str, int]], dict]:
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
    meta = dict(
        gains=gains,
        peg_init_offset_xy=(args.offset_x, args.offset_y),
        total_steps=step,
        success=True,
        first_contact_step=first_contact_step,
        success_step=success_step,
        camera=args.camera,
        rng_seed=None,  # deterministic scenario, no randomness sampled -- see module docstring
    )
    return frames, keyframes, meta


def compose_figure(frames: list[np.ndarray], keyframes: list[tuple[str, int]], out_dir: str) -> None:
    n = len(keyframes)
    fig_w_in = 7.0  # two-column full width (~17.5cm)
    fig, axes = plt.subplots(1, n, figsize=(fig_w_in, fig_w_in / n * (frames[0].shape[0] / frames[0].shape[1])))
    for i, ((label, step), ax) in enumerate(zip(keyframes, axes, strict=True)):
        ax.imshow(frames[step])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#999999")
        ax.set_title(label, fontsize=9, fontfamily="sans-serif")
        if i == 0:
            # stacked (not side-by-side): the panel is too narrow for both labels on one line
            ax.text(
                0.03, 0.14, "Stabilizer: left arm", transform=ax.transAxes, fontsize=7.5,
                fontfamily="sans-serif", color="white", weight="bold", va="center",
                bbox=dict(facecolor=STABILIZER_COLOR, alpha=0.9, pad=2, edgecolor="none"),
            )
            ax.text(
                0.03, 0.05, "Actuator: right arm", transform=ax.transAxes, fontsize=7.5,
                fontfamily="sans-serif", color="white", weight="bold", va="center",
                bbox=dict(facecolor=ACTUATOR_COLOR, alpha=0.9, pad=2, edgecolor="none"),
            )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig3_tasks.svg"))
    fig.savefig(os.path.join(out_dir, "fig3_tasks.pdf"))
    # dpi chosen so each ~1.75in-wide panel clears 640px in the saved PNG (confirmation render)
    fig.savefig(os.path.join(out_dir, "fig3_tasks.png"), dpi=400)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frames, keyframes, meta = run_and_pick_keyframes(args)
    compose_figure(frames, keyframes, args.out_dir)
    print(f"[render_task_figure] episode: gains={meta['gains']} offset_xy={meta['peg_init_offset_xy']} "
          f"camera={meta['camera']} rng_seed={meta['rng_seed']} (deterministic scenario, no sampling)")
    print(f"[render_task_figure] total_steps={meta['total_steps']} first_contact_step={meta['first_contact_step']} "
          f"success_step={meta['success_step']}")
    for label, step in keyframes:
        print(f"[render_task_figure] keyframe '{label}': step {step}")
    print(f"[render_task_figure] wrote fig3_tasks.{{svg,pdf,png}} to {args.out_dir}")


if __name__ == "__main__":
    main()
