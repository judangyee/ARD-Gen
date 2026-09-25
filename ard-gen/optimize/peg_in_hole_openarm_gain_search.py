"""OpenArm peg-in-hole 오른팔 위치제어 게인(kp/kv) + 널스페이스 게인 CMA-ES 탐색.

sim/peg_in_hole_bimanual_openarm_sim.py 모듈 docstring "이어서: CMA-ES로
kp/kv 체계적 탐색" 절의 실측 결과를 낸 스크립트를 정식으로 옮긴 것이다
(그 절이 처음 만들어질 때는 /tmp에 임시로만 있었음). 관절 종류별
(베이스/중간/손목) kp/kv 6개 + _NULLSPACE_GAIN 1개, 총 7개 파라미터를
찾는다.

시작점은 mj_fullM으로 실측한 관절별 유효 관성(대각 성분)에서 역산한
임계감쇠 근사치가 아니라(그 값으로 시작한 1차 탐색이 실제로는 훨씬 더
큰 kv로 수렴해버려서 -- 결합 동역학 때문으로 추정, 모듈 docstring 참고)
**이전 탐색이 찾은 최선의 값**을 그대로 쓴다(REFINE_FROM). 평가 길이를
늘리고(800스텝) 비용 함수에 "마지막 400스텝 중 최소 xy 오차"를 추가해서
드리프트가 한 번이라도 hole 반경 안으로 들어오는 순간을 직접 보상한다
(이전 버전은 평균/최고 삽입 깊이만 봐서, "거의 맞을 뻔한" 시도와 "전혀
안 맞은" 시도를 구분 못 했다).

사용법:
    python optimize/peg_in_hole_openarm_gain_search.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cma

import sim.peg_in_hole_bimanual_openarm_sim as simmod
from sim.peg_in_hole_bimanual_openarm_sim import (
    BimanualPegInHoleOpenArmSim,
    _default_scene_config,
    DT,
    Z_RATE,
)

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "peg_in_hole_bimanual_openarm.xml")
_TEMPLATE = open(_TEMPLATE_PATH).read()

# 이전 탐색(모듈 docstring 참고)이 찾은 값 -- 지금 파일에 반영된 것과 동일.
REFINE_FROM = [2273.98, 1568.18, 87.90, 894.41, 411.75, 663.96, 0.03]

N_STEPS = 800
TAIL_WINDOW = 400
OFFSET_XY = (0.014, 0.0)


def build_xml(kp_base: float, kv_base: float, kp_mid: float, kv_mid: float, kp_wrist: float, kv_wrist: float) -> str:
    txt = _TEMPLATE
    repl = [
        ('kp="2273.98" kv="1568.18"', f'kp="{kp_base:.2f}" kv="{kv_base:.2f}"'),
        ('kp="87.90" kv="894.41"', f'kp="{kp_mid:.2f}" kv="{kv_mid:.2f}"'),
        ('kp="411.75" kv="663.96"', f'kp="{kp_wrist:.2f}" kv="{kv_wrist:.2f}"'),
    ]
    for a, b in repl:
        if a not in txt:
            raise ValueError(f"template placeholder not found: {a!r} -- did REFINE_FROM defaults change?")
        txt = txt.replace(a, b)
    return txt


def eval_gains(params: np.ndarray, tag: str = "a") -> float:
    kp_base, kv_base, kp_mid, kv_mid, kp_wrist, kv_wrist, null_gain = np.abs(params)
    simmod._NULLSPACE_GAIN = float(null_gain)
    xml = build_xml(kp_base, kv_base, kp_mid, kv_mid, kp_wrist, kv_wrist)
    tmp_path = os.path.join(os.path.dirname(__file__), "..", "assets", f"_gain_search_tmp_{tag}.xml")
    with open(tmp_path, "w") as f:
        f.write(xml)
    try:
        sim = BimanualPegInHoleOpenArmSim(xml_path=tmp_path)
    except Exception:
        return 1000.0

    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = OFFSET_XY
    try:
        outer_half = sim.reset(cfg)
    except Exception:
        return 1000.0

    kp_xy, kd_xy = 0.000515, 2.4e-05
    prev_fe = np.zeros(2)
    target_depth = cfg["target_insertion_depth"]
    best_depth = 0.0
    xy_err_tail_sum = 0.0
    tail_count = 0
    min_xy_err_tail = 1e9
    raw_depth_at_min_xy = 0.0
    success_step = None
    for step in range(1, N_STEPS + 1):
        force, torque = sim.get_force_torque()
        fe = force[:2]
        dfe = (fe - prev_fe) / DT
        prev_fe = fe
        delta_xy = -kp_xy * fe - kd_xy * dfe
        delta = np.array([delta_xy[0], delta_xy[1], -Z_RATE])
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
        raw_depth = max(0.0, float(hole_center[2] - peg_tip[2]))
        depth = raw_depth if xy_ok else 0.0
        best_depth = max(best_depth, depth)
        if step > N_STEPS - TAIL_WINDOW:
            xy_err_tail_sum += xy_err
            tail_count += 1
            # xy가 가장 잘 맞았던 그 순간의 "판정 없는" 원시 삽입 깊이 --
            # best_depth(게이트 걸림, 한 번도 동시에 안 맞으면 0으로 고정)와
            # 달리 이건 항상 매끄러운 신호를 줘서, "정렬은 되는데 z가 못
            # 따라온" 실패와 "아예 정렬도 안 되는" 실패를 CMA-ES가 구분할
            # 수 있게 한다(모듈 docstring의 근접 실패 사례 참고).
            if xy_err < min_xy_err_tail:
                min_xy_err_tail = xy_err
                raw_depth_at_min_xy = raw_depth
        if depth >= target_depth:
            success_step = step
            break

    if success_step is not None:
        return -200.0 + success_step * 0.02

    tail_avg = xy_err_tail_sum / max(1, tail_count)
    cost = -25.0 * best_depth - 10.0 * raw_depth_at_min_xy + 2.0 * tail_avg + 6.0 * min_xy_err_tail
    return float(cost)


def main() -> None:
    x0 = REFINE_FROM
    sigma0 = [150, 150, 80, 150, 60, 80, 0.02]
    opts = {"seed": 3, "maxiter": 18, "popsize": 8, "verbose": -9, "CMA_stds": sigma0}
    es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
    gen = 0
    best_overall = None
    best_cost = 1e9
    while not es.stop():
        gen += 1
        xs = es.ask()
        costs = [eval_gains(x, tag=str(i % 8)) for i, x in enumerate(xs)]
        es.tell(xs, costs)
        idx = int(np.argmin(costs))
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best_overall = np.abs(xs[idx]).copy()
        print(f"gen {gen}: best_cost_this_gen={costs[idx]:.4f} overall_best={best_cost:.4f} "
              f"params={np.abs(xs[idx]).round(2)}", flush=True)

    print("FINAL best:", best_overall.round(3), "cost", best_cost)

    for i in range(8):
        p = os.path.join(os.path.dirname(__file__), "..", "assets", f"_gain_search_tmp_{i}.xml")
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
