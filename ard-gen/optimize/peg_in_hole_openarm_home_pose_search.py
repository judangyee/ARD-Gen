"""OpenArm peg-in-hole bimanual home 자세(14 DOF) CMA-ES 재탐색.

assets/peg_in_hole_bimanual_openarm.xml 상단 docstring "2. home 자세" 절이
설명하는 원래 탐색을 그대로 재현한다(그 절이 처음 만들어질 때 스크립트를
저장 안 해뒀다고 적어놓은 바로 그것) -- 다만 비용 함수의 peg tip / hole
목표점 로컬 오프셋을, grasp anchor 재조정(1~5차, sim 모듈의
_PEG_LOCAL_OFFSET / hole_grasp relpose 주석 참고) 이후의 새 값으로
바꿔서 다시 풀었다.

왜 다시 풀어야 했는지: grasp anchor를 "진짜 손끝 corner" 기준으로 옮기면서
peg가 손목에서 더 멀리 뻗게 됐고(오른팔), hole도 왼팔 ee의 기운 orientation
때문에 y로 9cm 가까이 이동했다(왼팔) -- 둘 다 이 파일의 home 자세(고정된
14개 관절 각도)는 그대로 둔 채였다. 실측해보니(sim/peg_in_hole_bimanual_openarm_sim.py를
1500스텝 그대로 돌려봄) 오른팔 joint4가 range의 90%, joint7이 94.3%
지점까지 붙어 있었고 -- 이 프로젝트가 이미 겪은 "관절 한계 페널티 없이
찾은 첫 home 자세" 실패와 똑같은 패턴(게인을 아무리 올려도 안 줄어드는
드리프트, 이번엔 108mm/1500스텝까지 벌어짐) -- 이번에도 grasp anchor
변경이 IK 목표점 자체를 옮겨놨으니 home 자세를 그 새 목표점에 맞춰
다시 풀어야 한다.

비용 함수 항(원래 탐색과 동일, 로컬 오프셋만 교체):
  1. 오른팔 ee_base_link 로컬 -z가 world -z와 일치(방향 오차)
  2. peg tip(오른팔 ee 로컬 _PEG_TIP_LOCAL)과 hole 목표점(왼팔 ee 로컬
     _HOLE_TARGET_LOCAL) 사이 xy 오차
  3. 둘의 z차가 호버 간격(_HOVER_GAP_M, sim 모듈과 동일한 값)과 얼마나 다른지
  4. 두 ee_base_link 원점 거리가 0.18m 미만이면 페널티
  5. peg tip이 작업 공간(x>0.15, 0.15<z<0.6) 밖으로 안 나가게
  6. 관절이 자기 range의 15% 이내로 한계에 붙으면 제곱 페널티(처음부터 포함
     -- 원래 탐색의 "실패한 첫 시도"를 반복하지 않으려고)

시작점(x0)은 완전히 새로 탐색하는 대신 **현재 home 자세**로 준다 -- grasp
anchor가 옮겨간 거리 자체는 몇 cm 수준이라, 처음부터 다시 찾는 것보다
기존 해 근방에서 재수렴하는 편이 훨씬 빠르고, 궤도 자체가 크게 안 바뀌어야
(이미 검증된 방향 정렬/작업공간 제약을 계속 만족하며) 다른 부수효과가
덜 생긴다.

사용법:
    python optimize/peg_in_hole_openarm_home_pose_search.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cma
import mujoco
import numpy as np

from sim.peg_in_hole_bimanual_openarm_sim import (
    _ARM_JOINTS,
    _HOME_QPOS,
    _LEFT_ARM_HOME_QPOS,
    _PEG_LOCAL_OFFSET,
    _HOVER_GAP_M,
)

_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "peg_in_hole_bimanual_openarm.xml")

_PEG_TIP_LOCAL = _PEG_LOCAL_OFFSET + np.array([0.0, 0.0, -0.04])
_HOLE_TARGET_LOCAL = np.array([-0.01772, -0.00214, -0.19274])  # hole_grasp relpose xyz(재조정 2차)

_LEFT_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
_ALL_JOINTS = _ARM_JOINTS + _LEFT_JOINTS  # 순서: 오른팔 7 + 왼팔 7 (docstring과 동일)

_MIN_HAND_DIST = 0.18
_WORKSPACE_X_MIN = 0.15
_WORKSPACE_Z_MIN = 0.15
_WORKSPACE_Z_MAX = 0.6
_MARGIN_FRAC = 0.15


def _load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(_XML_PATH)


def evaluate(params: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData) -> float:
    mujoco.mj_resetData(model, data)
    for name, val in zip(_ALL_JOINTS, params):
        data.qpos[model.joint(name).qposadr[0]] = val
    mujoco.mj_forward(model, data)

    right_ee_id = model.body("openarm_right_ee_base_link").id
    left_ee_id = model.body("openarm_left_ee_base_link").id
    right_pos = data.xpos[right_ee_id]
    left_pos = data.xpos[left_ee_id]
    right_mat = data.xmat[right_ee_id].reshape(3, 3)
    left_mat = data.xmat[left_ee_id].reshape(3, 3)

    # 1. 방향 오차
    right_negz_world = -right_mat[:, 2]
    orient_err = float(np.sum((right_negz_world - np.array([0.0, 0.0, -1.0])) ** 2))

    # 2/3. peg tip vs hole 목표점
    peg_tip = right_pos + right_mat @ _PEG_TIP_LOCAL
    hole_target = left_pos + left_mat @ _HOLE_TARGET_LOCAL
    xy_err = float(np.sum((peg_tip[:2] - hole_target[:2]) ** 2))
    z_diff = float(peg_tip[2] - hole_target[2])
    z_err = (z_diff - _HOVER_GAP_M) ** 2

    # 4. 손 간 거리
    hand_dist = float(np.linalg.norm(right_pos - left_pos))
    clearance_pen = max(0.0, _MIN_HAND_DIST - hand_dist) ** 2

    # 5. peg tip 작업 공간
    workspace_pen = 0.0
    if peg_tip[0] <= _WORKSPACE_X_MIN:
        workspace_pen += (_WORKSPACE_X_MIN - peg_tip[0]) ** 2
    if peg_tip[2] <= _WORKSPACE_Z_MIN:
        workspace_pen += (_WORKSPACE_Z_MIN - peg_tip[2]) ** 2
    if peg_tip[2] >= _WORKSPACE_Z_MAX:
        workspace_pen += (peg_tip[2] - _WORKSPACE_Z_MAX) ** 2

    # 6. 관절 한계 여유
    margin_pen = 0.0
    for name in _ALL_JOINTS:
        lo, hi = model.jnt_range[model.joint(name).id]
        val = data.qpos[model.joint(name).qposadr[0]]
        frac = (val - lo) / (hi - lo)
        if frac < _MARGIN_FRAC:
            margin_pen += (_MARGIN_FRAC - frac) ** 2
        elif frac > 1.0 - _MARGIN_FRAC:
            margin_pen += (frac - (1.0 - _MARGIN_FRAC)) ** 2

    cost = (
        50.0 * orient_err
        + 400.0 * xy_err
        + 200.0 * z_err
        + 30.0 * clearance_pen
        + 30.0 * workspace_pen
        + 10.0 * margin_pen
    )
    return float(cost)


def report(params: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    for name, val in zip(_ALL_JOINTS, params):
        data.qpos[model.joint(name).qposadr[0]] = val
    mujoco.mj_forward(model, data)

    right_ee_id = model.body("openarm_right_ee_base_link").id
    left_ee_id = model.body("openarm_left_ee_base_link").id
    right_pos = data.xpos[right_ee_id]
    left_pos = data.xpos[left_ee_id]
    right_mat = data.xmat[right_ee_id].reshape(3, 3)
    left_mat = data.xmat[left_ee_id].reshape(3, 3)
    peg_tip = right_pos + right_mat @ _PEG_TIP_LOCAL
    hole_target = left_pos + left_mat @ _HOLE_TARGET_LOCAL

    print(f"[home_pose] peg_tip world = {peg_tip}")
    print(f"[home_pose] hole_target world = {hole_target}")
    print(f"[home_pose] xy_err_mm = {np.linalg.norm(peg_tip[:2]-hole_target[:2])*1000:.3f}")
    print(f"[home_pose] z_diff_mm = {(peg_tip[2]-hole_target[2])*1000:.3f} (target {_HOVER_GAP_M*1000:.1f})")
    print(f"[home_pose] hand_dist = {np.linalg.norm(right_pos-left_pos):.4f}")
    print("[home_pose] joint fractions:")
    for name in _ALL_JOINTS:
        lo, hi = model.jnt_range[model.joint(name).id]
        val = data.qpos[model.joint(name).qposadr[0]]
        frac = (val - lo) / (hi - lo)
        flag = "  <-- within margin!" if frac < _MARGIN_FRAC or frac > 1 - _MARGIN_FRAC else ""
        print(f"    {name}: {val:.6f} frac={frac:.3f}{flag}")


def main() -> None:
    model = _load_model()
    data = mujoco.MjData(model)

    x0 = [_HOME_QPOS[n] for n in _ARM_JOINTS] + [_LEFT_ARM_HOME_QPOS[n] for n in _LEFT_JOINTS]
    sigma0 = 0.3
    opts = {"seed": 1, "maxiter": 150, "popsize": 12, "verbose": -9}
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    print("[home_pose] start cost:", evaluate(np.array(x0), model, data))

    gen = 0
    best_cost = np.inf
    best_params = np.array(x0)
    while not es.stop():
        gen += 1
        xs = es.ask()
        costs = [evaluate(np.array(x), model, data) for x in xs]
        es.tell(xs, costs)
        idx = int(np.argmin(costs))
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best_params = np.array(xs[idx])
        if gen % 10 == 0 or gen == 1:
            print(f"[home_pose] gen {gen}: best_cost_this_gen={costs[idx]:.5f} overall_best={best_cost:.5f}")

    print(f"[home_pose] FINAL best_cost={best_cost:.6f}")
    report(best_params, model, data)

    right_vals = best_params[: len(_ARM_JOINTS)]
    left_vals = best_params[len(_ARM_JOINTS) :]
    print("[home_pose] _HOME_QPOS (right):")
    for name, val in zip(_ARM_JOINTS, right_vals):
        print(f'    "{name}": {val!r},')
    print("[home_pose] _LEFT_ARM_HOME_QPOS (left):")
    for name, val in zip(_LEFT_JOINTS, left_vals):
        print(f'    "{name}": {val!r},')


if __name__ == "__main__":
    main()
