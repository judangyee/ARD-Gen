"""RoboTwin 2.0 이식 Part 3-1: torque-reactive 패턴.

실제 예시: cap_twist. 측정 토크에 비례해 속도를 낮추는 admittance
게인 1개(Kp_tau) + 목표 방향 누적 회전(forward_progress) 기반
reward/success. 저항을 Coulomb이 아니라 속도-의존 dof_damping으로
모델링해서 "느리게 돌리면 필요 토크가 실제로 줄어드는" 관계를
만든 설계(assets/cap_twist.xml 모듈 docstring 참고)가 이 패턴의 핵심.

compute_reward/is_success는 sim/cap_twist_env.py:CapTwistEnv의 원래
구현을 글자 그대로 옮긴 것(계수만 coefs dict로 뺐다) --
tests/test_patterns_identity.py가 리팩토링 전/후 reward가
bit-identical함을 확인한다.
"""
from __future__ import annotations

from typing import Any

from patterns.common import shaped_reward

GAIN_SCHEMA = {
    "n_gains": 1,
    "roles": ["kp_torque"],
    "description": (
        "측정 토크(N*m)에 곱해 속도를 낮추는 단일 admittance 게인 -- "
        "dof_damping 기반 속도-의존 저항 태스크용(cap_twist가 실제 예시)."
    ),
}

# sim/cap_twist_env.py:CapTwistEnv.compute_reward()/is_success()의 원래
# 하드코딩 값과 완전히 동일.
DEFAULT_COEFS = {
    "shortfall_coef": 1.0,
    "progress_coef": 30.0,
    "torque_excess_coef": 0.02,
    "safe_torque": 0.5,
    "step_coef": 0.01,
    "success_bonus": 50.0,
    "torque_drop_threshold": 0.05,
}


def compute_reward(episode_result: dict[str, Any], coefs: dict[str, float]) -> float:
    target_mag = abs(episode_result["target_rotation"])
    forward_progress = episode_result["forward_progress"]
    progress_fraction = max(0.0, min(forward_progress / target_mag, 1.0)) if target_mag > 0 else 0.0
    shortfall = max(0.0, target_mag - forward_progress)
    excess = max(0.0, episode_result["max_torque"] - coefs["safe_torque"])
    return shaped_reward(
        progress_term=progress_fraction,
        progress_coef=coefs["progress_coef"],
        shortfall_term=shortfall,
        shortfall_coef=coefs["shortfall_coef"],
        excess_terms=[(excess, coefs["torque_excess_coef"])],
        step_count=episode_result["step_count"],
        step_coef=coefs["step_coef"],
        flat_bonuses=[coefs["success_bonus"] if episode_result.get("success") else 0.0],
    )


def is_success(episode_result: dict[str, Any], coefs: dict[str, Any]) -> bool:
    target_mag = abs(episode_result["target_rotation"])
    reached_target = episode_result["forward_progress"] >= target_mag
    torque_dropped = episode_result.get("disengaged", False) and abs(
        episode_result.get("current_torque", 1.0)
    ) < coefs["torque_drop_threshold"]
    return bool(reached_target or torque_dropped)


def mjcf_substructure(params: dict[str, Any]) -> str:
    """torque-reactive 패턴의 MJCF 골격(scaffold_task.py가 새 태스크
    생성 시 씀) -- 힌지 1개 + 위치 액추에이터 + jointactuatorfrc
    센서(= "토크 센서", assets/cap_twist.xml 모듈 docstring의
    "jointactuatorfrc = 토크 센서" 절 참고). dof_damping은 여기서
    0으로 채워두고 reset() 시점에 scene_config의 resistance_torque로
    덮어쓴다(cap_twist의 damping=resistance_torque/OMEGA_BASE 변환과
    동일 패턴) -- scaffold_task.py가 생성하는 Env도 이 변환을 그대로
    가져다 쓴다.

    params 필수 키: hinge_body_name, hinge_joint_name, actuator_kp,
    actuator_forcerange(2), body_half_size(3), body_mass. frictionloss는
    선택(기본 0.05) -- body_half_size/body_mass로 geom을 같이 만드는
    이유: 힌지는 움직이는 바디라 질량/관성이 0이면 MuJoCo가 컴파일을
    거부한다(실측 확인: "mass and inertia of moving bodies must be
    larger than mjMINVAL")."""
    body = params["hinge_body_name"]
    joint = params["hinge_joint_name"]
    kp = params["actuator_kp"]
    frlo, frhi = params["actuator_forcerange"]
    frictionloss = params.get("frictionloss", 0.05)
    hs = " ".join(str(x) for x in params["body_half_size"])
    mass = params["body_mass"]
    return f"""<!-- torque-reactive 패턴: {body}(힌지 1개, 저항은 dof_damping으로 모델링) -->
<body name="{body}" pos="0 0 0">
  <joint name="{joint}" type="hinge" axis="0 0 1" pos="0 0 0"
         damping="0.0" frictionloss="{frictionloss}" limited="false"/>
  <geom name="{body}_geom" type="box" size="{hs}" mass="{mass}" rgba="0.8 0.3 0.2 1"/>
</body>
<actuator>
  <position name="{joint}_drive" joint="{joint}" kp="{kp}" forcerange="{frlo} {frhi}"/>
</actuator>
<sensor>
  <jointactuatorfrc name="{joint}_torque" joint="{joint}"/>
</sensor>"""
