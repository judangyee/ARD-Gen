"""OpenArm peg-in-hole 오른팔 xy 위치 제어 게인(Kp_xy, Kd_xy) CMA-ES 탐색.

이전 버전(admittance/접촉힘 기반)은 폐기했다 -- "지금 역기구학으로 한거
아니야? 아직도 위치를 못잡는데?" 질문으로 드러난 근본 문제: xy 보정이
peg tip에 걸리는 접촉힘을 상쇄하는 방식(admittance)이었는데, 실측해보니
힘이 거의 항상 0(peg가 hole에 닿은 적이 없음)이라 xy를 어느 쪽으로
움직여야 할지 알려주는 신호 자체가 없었다. "어짜피 성공하는 데이터를
모아서 학습에 쓰는 게 목적이니 (시뮬레이션이 알고 있는) hole 실제 위치를
그냥 써도 되지 않냐"는 결정 이후, xy 제어를 접촉힘이 아니라 hole 실제
위치와 peg tip 사이의 **직접적인 위치 오차(dx,dy)**에 대한 PD로 바꿨다
(sim/peg_in_hole_bimanual_openarm_sim.py의 _run_episode_with_sim 주석
참고). 그래서 Kp_xy/Kd_xy의 단위/스케일이 완전히 달라졌다 -- 예전
admittance 값(0.000515, 2.4e-05, force 단위 기준)은 이제 아무 의미가
없고, 이 스크립트가 그 새 스케일을 처음부터 다시 찾는다.

관절 kp/kv/nullspace, home 자세는 지금 XML/모듈에 반영된 값으로 고정해두고
(peg_in_hole_openarm_gain_search.py / peg_in_hole_openarm_home_pose_search.py
참고), Kp_xy/Kd_xy 2개만 찾는다. 평가는 render 스크립트와 동일하게
adaptive_z_rate(오버슈트 정지 포함)와 위치 기반 xy PD를 그대로 재현한다.
평가 길이는 1500스텝(sim 모듈 docstring의 "짧은 구간 착시" 교훈 참고 --
너무 짧으면 실제로는 발산하는 걸 놓칠 수 있음), 성공 판정은 sim 모듈에
있는 raw_depth clip + 연속 유지 기준을 그대로 가져다 쓴다.

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

# dx,dy는 m 단위(대개 0.01~0.05 범위)이고 delta_xy는 Z_RATE(0.0006m/스텝)와
# 비슷한 스케일이어야 하니(너무 크면 한 스텝에 과하게 튀고, 너무 작으면
# 못 따라감) Kp_xy는 대략 0.01~0.1 자리, Kd_xy(속도 항, dx의 변화율에
# 곱함)는 그보다 한두 자릿수 작게 잡는다.
KP_BOUNDS = (0.001, 0.5)
KD_BOUNDS = (0.0, 0.05)


def eval_gains(sim: BimanualPegInHoleOpenArmSim, kp_xy: float, kd_xy: float) -> float:
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = OFFSET_XY
    try:
        outer_half = sim.reset(cfg)
    except Exception:
        return 1000.0

    target_depth = cfg["target_insertion_depth"]
    prev_dx, prev_dy = 0.0, 0.0
    best_depth = 0.0
    xy_err_tail_sum = 0.0
    tail_count = 0
    min_xy_err_tail = 1e9
    raw_depth_at_min_xy = 0.0
    success_step = None
    hold_count = 0
    for step in range(1, N_STEPS + 1):
        cur_tip = sim.get_peg_tip_pos()
        cur_hole = sim.get_hole_center_pos()
        cur_dx = float(cur_hole[0] - cur_tip[0])
        cur_dy = float(cur_hole[1] - cur_tip[1])
        d_dx = (cur_dx - prev_dx) / DT
        d_dy = (cur_dy - prev_dy) / DT
        prev_dx, prev_dy = cur_dx, cur_dy
        delta_xy = np.array([kp_xy * cur_dx + kd_xy * d_dx, kp_xy * cur_dy + kd_xy * d_dy])

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

    x0 = [0.05, 0.01]
    sigma0 = 1.0
    opts = {
        "bounds": [[KP_BOUNDS[0], KD_BOUNDS[0]], [KP_BOUNDS[1], KD_BOUNDS[1]]],
        "CMA_stds": [0.05, 0.01],
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
