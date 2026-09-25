"""OpenArm peg-in-hole 오른팔 admittance 게인(Kp_xy, Kd_xy) CMA-ES 탐색.

sim/peg_in_hole_bimanual_openarm_sim.py 모듈 최상단 docstring의 "아직
검증 안 된 것" 절 -- Kp_xy=0.000515, Kd_xy=2.4e-05는 VX300s 단일팔
버전(sim/peg_in_hole_sim.py)에서 그대로 가져온 값이고, 이 팔(7-DOF,
관절별로 강화된 kp/kv, 훨씬 무거운 팔)에 맞게 재탐색한 적이 한 번도
없었다. peg_in_hole_openarm_gain_search.py(관절 kp/kv + 널스페이스)와
peg_in_hole_openarm_home_pose_search.py(home 자세)로 관절 한계 문제는
해결했지만, xy 드리프트 자체는 여전히 목표 허용치(~19.5mm)보다 2~4배
넓게 진동한다("2번으로 해보자" -- grasp anchor를 손끝/힌지 사이로 옮기는
대신 이 admittance 게인 쪽을 파보기로 함).

관절 kp/kv/nullspace는 지금 XML/모듈에 반영된 값으로 고정해두고(그래서
XML을 다시 빌드할 필요 없이 BimanualPegInHoleOpenArmSim()을 그대로
재사용한다 -- VX300s의 optimize/cma_search.py와 같은 방식), Kp_xy/Kd_xy
2개만 찾는다. 평가는 render 스크립트와 똑같이 adaptive_z_rate를 쓴다
(이전 kp/kv 탐색은 고정 Z_RATE를 썼는데, 실제로 렌더링/에피소드에 쓰이는
건 adaptive_z_rate라 그것과 다르게 평가하면 결과가 실제 동작과 어긋날
수 있다). 평가 길이는 1500스텝(짧은 구간에서 "진동처럼 보이지만 사실은
서서히 발산" 착시를 겪은 적이 있어 sim 모듈 docstring 참고 -- 2000~4000이
이상적이지만 세대당 평가 수 x 스텝 수 예산을 고려해 1500으로 절충) --
성공 판정은 sim 모듈에 이미 고친 버그(raw_depth clip + 연속 유지) 그대로
가져다 쓴다.

사용법:
    python optimize/peg_in_hole_openarm_admittance_gain_search.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cma
import numpy as np

from sim.peg_in_hole_bimanual_openarm_sim import (
    BimanualPegInHoleOpenArmSim,
    _default_scene_config,
    DT,
    adaptive_z_rate,
    _MAX_SANE_DEPTH_M,
    _SUCCESS_HOLD_STEPS,
)

N_STEPS = 1500
TAIL_WINDOW = 600
OFFSET_XY = (0.014, 0.0)

KP_BOUNDS = (0.00002, 0.003)
KD_BOUNDS = (0.0, 0.0008)


def eval_gains(sim: BimanualPegInHoleOpenArmSim, kp_xy: float, kd_xy: float) -> float:
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = OFFSET_XY
    try:
        outer_half = sim.reset(cfg)
    except Exception:
        return 1000.0

    target_depth = cfg["target_insertion_depth"]
    prev_fe = np.zeros(2)
    best_depth = 0.0
    xy_err_tail_sum = 0.0
    tail_count = 0
    min_xy_err_tail = 1e9
    raw_depth_at_min_xy = 0.0
    success_step = None
    hold_count = 0
    for step in range(1, N_STEPS + 1):
        force, _torque = sim.get_force_torque()
        fe = force[:2]
        dfe = (fe - prev_fe) / DT
        prev_fe = fe
        delta_xy = -kp_xy * fe - kd_xy * dfe

        cur_tip = sim.get_peg_tip_pos()
        cur_hole = sim.get_hole_center_pos()
        cur_dx = float(cur_hole[0] - cur_tip[0])
        cur_dy = float(cur_hole[1] - cur_tip[1])
        cur_raw_depth = max(0.0, float(cur_hole[2] - cur_tip[2]))
        z_rate = adaptive_z_rate(cur_dx, cur_dy, outer_half, cur_raw_depth, target_depth)
        delta = np.array([delta_xy[0], delta_xy[1], -z_rate])
        try:
            sim.step(delta)
        except Exception:
            return 1000.0
        if not np.all(np.isfinite(sim.data.qpos)):
            return 1000.0

        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_err = (dx**2 + dy**2) ** 0.5
        xy_ok = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))
        depth = raw_depth if xy_ok else 0.0
        best_depth = max(best_depth, depth)
        if step > N_STEPS - TAIL_WINDOW:
            xy_err_tail_sum += xy_err
            tail_count += 1
            if xy_err < min_xy_err_tail:
                min_xy_err_tail = xy_err
                raw_depth_at_min_xy = raw_depth
        hold_count = hold_count + 1 if depth >= target_depth else 0
        if hold_count >= _SUCCESS_HOLD_STEPS:
            success_step = step
            break

    if success_step is not None:
        return -200.0 + success_step * 0.02

    tail_avg = xy_err_tail_sum / max(1, tail_count)
    cost = -25.0 * best_depth - 10.0 * raw_depth_at_min_xy + 2.0 * tail_avg + 6.0 * min_xy_err_tail
    return float(cost)


def main() -> None:
    sims = [BimanualPegInHoleOpenArmSim() for _ in range(8)]

    x0 = [0.000515, 2.4e-05]
    sigma0 = 1.0
    opts = {
        "bounds": [[KP_BOUNDS[0], KD_BOUNDS[0]], [KP_BOUNDS[1], KD_BOUNDS[1]]],
        "CMA_stds": [0.0004, 0.00015],
        "popsize": 8,
        "maxiter": 15,
        "seed": 5,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    print("[admittance_search] start cost:", eval_gains(sims[0], *x0), flush=True)

    gen = 0
    best_overall = None
    best_cost = 1e9
    while not es.stop():
        gen += 1
        xs = es.ask()
        costs = [eval_gains(sims[i % len(sims)], *x) for i, x in enumerate(xs)]
        es.tell(xs, costs)
        idx = int(np.argmin(costs))
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best_overall = np.array(xs[idx]).copy()
        print(
            f"[admittance_search] gen {gen}: best_cost_this_gen={costs[idx]:.4f} "
            f"overall_best={best_cost:.4f} params(Kp,Kd)={xs[idx][0]:.6f},{xs[idx][1]:.6f}",
            flush=True,
        )

    print(f"[admittance_search] FINAL best: Kp_xy={best_overall[0]:.6f} Kd_xy={best_overall[1]:.6f} cost={best_cost:.4f}")


if __name__ == "__main__":
    main()
