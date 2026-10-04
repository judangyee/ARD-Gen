"""OpenArm 7-DOF 팔의 resolved-rate IK 공통 코어.

peg_in_hole_bimanual_openarm_sim.py/tacker_openarm_env.py/
screw_driving_bimanual_openarm_sim.py 세 파일의 `_advance_virtual*()`가
거의 글자 그대로 중복하고 있던 로직(ponytail-audit로 발견)을 여기 하나로
뽑았다 -- 널스페이스 투영(홈 자세 복귀) + 관절 한계 회피(Jacobian-column-
freezing, 사후 clip이 아니라 그 관절을 "이번 스텝엔 없는 자유도"로 취급해서
다시 풀기) + (선택) 한 스텝 최대 관절 변화량 클램프. 원리는
peg_in_hole_bimanual_openarm_sim.py의 _advance_virtual() docstring에
자세히 적혀 있다 -- 수학/동작은 그대로, 세 파일에 흩어져 있던 코드를
모았을 뿐이다.
"""
from __future__ import annotations

import mujoco
import numpy as np


def resolved_rate_step(
    model: mujoco.MjModel,
    J: np.ndarray,
    target: np.ndarray,
    virtual_qpos: dict[str, float],
    arm_joints: list[str],
    arm_dofadr: dict[str, int],
    home_qpos: dict[str, float],
    jac_damping: float,
    nullspace_gain: float,
    limit_freeze_margin: float,
    max_dq_per_step: float | None = None,
) -> dict[str, float]:
    """virtual_qpos를 J@dq=target을 (관절 한계 안에서) 만족시키는 쪽으로
    한 스텝 전진시킨 새 dict를 돌려준다. J는 (k, model.nv), target은 (k,)
    -- k(과제 차원)는 임의(위치만 3, 위치+롤 4 등)."""

    def _solve(Jm: np.ndarray) -> np.ndarray:
        jjt = Jm @ Jm.T + jac_damping * np.eye(Jm.shape[0])
        return Jm.T @ np.linalg.inv(jjt)

    # 1단계: 미정정(unconstrained) 풀이로 이번 스텝에 한계 쪽으로 더 밀릴
    # 관절을 찾는다.
    pinv0 = _solve(J)
    dq_task0 = pinv0 @ target
    J_frozen = J.copy()
    frozen_dofs = []
    for name in arm_joints:
        dof = arm_dofadr[name]
        lo, hi = model.jnt_range[model.joint(name).id]
        frac = (virtual_qpos[name] - lo) / (hi - lo)
        if (frac < limit_freeze_margin and dq_task0[dof] < 0.0) or (
            frac > 1.0 - limit_freeze_margin and dq_task0[dof] > 0.0
        ):
            J_frozen[:, dof] = 0.0
            frozen_dofs.append(dof)

    # 2단계: 얼린 관절을 뺀 Jacobian으로 다시 풀어서 나머지 관절이
    # 대신하게 한다 -- (수정된) J@dq=target 관계가 계속 성립해서 사후
    # clip류의 누출이 구조적으로 없다.
    pinv = _solve(J_frozen)
    dq_task = pinv @ target

    if max_dq_per_step is not None:
        dq_task_norm = float(np.max(np.abs(dq_task))) if dq_task.size else 0.0
        if dq_task_norm > max_dq_per_step:
            dq_task = dq_task * (max_dq_per_step / dq_task_norm)

    nv = model.nv
    null_proj = np.eye(nv) - pinv @ J_frozen
    dq_null = np.zeros(nv)
    for name in arm_joints:
        dof = arm_dofadr[name]
        dq_null[dof] = nullspace_gain * (home_qpos[name] - virtual_qpos[name])
    dq = dq_task + null_proj @ dq_null
    for dof in frozen_dofs:
        dq[dof] = 0.0

    new_virtual_qpos = dict(virtual_qpos)
    for name in arm_joints:
        dof = arm_dofadr[name]
        lo, hi = model.jnt_range[model.joint(name).id]
        new_virtual_qpos[name] = float(np.clip(virtual_qpos[name] + dq[dof], lo, hi))
    return new_virtual_qpos


def apply_openarm_gravcomp(model: mujoco.MjModel) -> None:
    """openarm_left_*/openarm_right_* 바디에 gravcomp=1을 켠다 -- vendor
    모델에는 기본으로 없어서 세 파일이 똑같이 켜고 있던 루프(ponytail-audit
    로 발견)."""
    for i in range(model.nbody):
        name = model.body(i).name
        if name.startswith("openarm_left_") or name.startswith("openarm_right_"):
            model.body_gravcomp[i] = 1.0
