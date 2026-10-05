"""RoboTwin 2.0 이식 Part 3-1: position-correction 패턴.

실제 예시: peg_in_hole(OpenArm). 2D 위치 오차(목표 지점까지의 xy
거리)를 PD(Kp/Kd) 게인으로 줄이는 Actuator + 접촉/삽입 깊이 기반
reward/success. "접촉힘이 아니라 목표 위치로 직접 타겟팅"하는 태스크용
(sim/peg_in_hole_openarm_env.py 모듈 docstring의 "게인의 의미가 VX300s와
다르다" 절 참고 -- 이 패턴은 그 설계를 그대로 모듈화한 것).

compute_reward/is_success는 sim/peg_in_hole_openarm_env.py:
PegInHoleOpenArmEnv의 원래 구현을 글자 그대로 옮긴 것뿐이다(계수만
coefs dict로 뺐다) -- tests/test_patterns_identity.py가 리팩토링
전/후 reward가 bit-identical함을 확인한다.
"""
from __future__ import annotations

from typing import Any

from patterns.common import shaped_reward

GAIN_SCHEMA = {
    "n_gains": 2,
    "roles": ["kp_position", "kd_position"],
    "description": (
        "2D 위치 오차(m)에 곱하는 P/D 게인 -- 접촉힘이 아니라 목표 위치로 "
        "직접 타겟팅하는 태스크용(peg_in_hole OpenArm 버전이 실제 예시)."
    ),
}

# peg_in_hole_openarm_env.py:PegInHoleOpenArmEnv.compute_reward()의 원래
# 하드코딩 값과 완전히 동일 -- 리팩토링으로 숫자를 "옮겼을 뿐" 바꾸지
# 않았다는 걸 보여주기 위해 그 파일이 이 상수를 그대로 가져다 쓴다.
DEFAULT_COEFS = {
    "distance_coef": 2.0,
    "progress_coef": 20.0,
    "force_excess_coef": 0.001,
    "force_safe_threshold": 5.0,
    "step_coef": 0.01,
    "success_bonus": 50.0,
}


def compute_reward(episode_result: dict[str, Any], coefs: dict[str, float]) -> float:
    excess = max(0.0, episode_result["max_force"] - coefs["force_safe_threshold"])
    return shaped_reward(
        progress_term=episode_result["insertion_depth"],
        progress_coef=coefs["progress_coef"],
        shortfall_term=episode_result["final_distance"],
        shortfall_coef=coefs["distance_coef"],
        excess_terms=[(excess, coefs["force_excess_coef"])],
        step_count=episode_result["step_count"],
        step_coef=coefs["step_coef"],
        flat_bonuses=[coefs["success_bonus"] if episode_result.get("success") else 0.0],
    )


def is_success(episode_result: dict[str, Any], coefs: dict[str, Any]) -> bool:
    """coefs["target_depth"]는 씬마다 달라서(target_insertion_depth)
    reward 계수들과 달리 매 호출 때 Env가 scene_config에서 읽어 넣어줘야
    한다 -- 원래 코드의 self._target_depth(reset()에서 저장)와 동일한
    역할."""
    return episode_result["insertion_depth"] >= coefs["target_depth"]


def mjcf_substructure(params: dict[str, Any]) -> str:
    """position-correction 패턴의 MJCF 골격(scaffold_task.py가 새 태스크
    생성 시 씀) -- 오른팔이 쥐는 자유 바디(moving_object) + 왼팔/고정
    프레임이 쥐는 자유 바디(target_object) + 자기 그리퍼와의 접촉 제외.
    assets/peg_in_hole_bimanual_openarm.xml의 peg/hole_socket 바디를
    일반화한 것 -- 치수/anchor는 숫자만 채워 넣는다(자유형 작성 아님).

    params 필수 키: moving_object_name, moving_half_size(3), target_object_name,
    target_half_size(3), moving_grasp_relpose(7: xyz+quat),
    target_grasp_relpose(7), moving_gripper_body, target_gripper_bodies(list[str])."""
    mo = params["moving_object_name"]
    to = params["target_object_name"]
    mhs = " ".join(str(x) for x in params["moving_half_size"])
    ths = " ".join(str(x) for x in params["target_half_size"])
    exclude_lines = "\n    ".join(
        f'<exclude body1="{mo}" body2="{gripper}"/>' for gripper in params.get("moving_gripper_bodies", [])
    ) + "\n    " + "\n    ".join(
        f'<exclude body1="{to}" body2="{gripper}"/>' for gripper in params.get("target_gripper_bodies", [])
    )
    return f"""<!-- position-correction 패턴: {mo}(이동 물체, Actuator가 쥠) + {to}(목표, Stabilizer/고정이 쥠) -->
<body name="{mo}" pos="0 0 0">
  <freejoint name="{mo}_free"/>
  <geom name="{mo}_geom" type="box" size="{mhs}" mass="0.04"/>
</body>
<body name="{to}" pos="0 0 0">
  <freejoint name="{to}_free"/>
  <geom name="{to}_geom" type="box" size="{ths}"/>
</body>
<!-- equality/weld, contact/exclude는 각 그리퍼 anchor(relpose)를 실측해서
     별도로 채운다(scaffold_task.py가 패턴별 기본 relpose를 제공) -->
<!-- contact excludes:
    {exclude_lines}
-->"""
