"""RoboTwin 2.0 이식 Part 3-1: impact-recoil 패턴.

실제 예시: tacker. 접근 단계에서만 댐핑 게인(Kp_approach) 하나를
쓰고, 충격/반동 자체는 제어하지 않는다 -- Stabilizer(왼팔)가 반동을
흡수한다(사용자 요청 "발사 반동은 Stabilizer가 흡수"). reward/success는
발사(fired) + 3단계(발사 정착/후퇴 중/최종 정착) 변위 허용치 + 안전
거리까지 후퇴(retracted) 조건으로 구성된다.

compute_reward/is_success는 sim/tacker_openarm_env.py:TackerOpenArmEnv의
원래 구현을 글자 그대로 옮긴 것(계수만 coefs dict로 뺐다) --
tests/test_patterns_identity.py가 리팩토링 전/후 reward가
bit-identical함을 확인한다.
"""
from __future__ import annotations

from typing import Any

from patterns.common import shaped_reward

GAIN_SCHEMA = {
    "n_gains": 1,
    "roles": ["kp_approach"],
    "description": (
        "접근 단계에서만 쓰는 단일 댐핑 게인 -- 반동 자체는 제어하지 "
        "않고 Stabilizer가 흡수한다(impact-recoil 패턴, tacker가 실제 예시)."
    ),
}

# sim/tacker_openarm_env.py:TackerOpenArmEnv.compute_reward()/is_success()의
# 원래 하드코딩 값과 완전히 동일.
DEFAULT_COEFS = {
    "distance_coef": 2.0,
    "fired_bonus_coef": 20.0,
    "displacement_excess_coef": 500.0,
    "overshoot_coef": 20.0,
    "step_coef": 0.01,
    "retract_bonus": 10.0,
    "success_bonus": 50.0,
    "default_success_displacement_m": 0.006,  # tacker_openarm_env.py의 SUCCESS_DISPLACEMENT_M
}


def compute_reward(episode_result: dict[str, Any], coefs: dict[str, float]) -> float:
    success_disp = episode_result.get("success_displacement_m", coefs["default_success_displacement_m"])
    excess_terms = [
        (max(0.0, episode_result.get("displacement", 0.0) - success_disp), coefs["displacement_excess_coef"]),
        (max(0.0, episode_result.get("retract_bump", 0.0) - success_disp), coefs["displacement_excess_coef"]),
        (max(0.0, episode_result.get("final_bump", 0.0) - success_disp), coefs["displacement_excess_coef"]),
        (episode_result.get("overshoot_penalty", 0.0), coefs["overshoot_coef"]),
    ]
    flat_bonuses = []
    if episode_result.get("retracted"):
        flat_bonuses.append(coefs["retract_bonus"])
    if episode_result.get("success"):
        flat_bonuses.append(coefs["success_bonus"])
    return shaped_reward(
        progress_term=1.0 if episode_result.get("fired") else 0.0,
        progress_coef=coefs["fired_bonus_coef"],
        shortfall_term=episode_result["final_distance"],
        shortfall_coef=coefs["distance_coef"],
        excess_terms=excess_terms,
        step_count=episode_result["step_count"],
        step_coef=coefs["step_coef"],
        flat_bonuses=flat_bonuses,
    )


def is_success(episode_result: dict[str, Any], coefs: dict[str, Any]) -> bool:
    if not episode_result.get("fired") or not episode_result.get("retracted"):
        return False
    success_disp = episode_result.get("success_displacement_m", coefs["default_success_displacement_m"])
    return bool(
        episode_result.get("displacement", float("inf")) <= success_disp
        and episode_result.get("retract_bump", float("inf")) <= success_disp
        and episode_result.get("final_bump", float("inf")) <= success_disp
    )


def mjcf_substructure(params: dict[str, Any]) -> str:
    """impact-recoil 패턴의 MJCF 골격(scaffold_task.py가 새 태스크
    생성 시 씀) -- workpiece(자유 바디, 충격으로 밀림) + Stabilizer가
    쥐는 weld(반동 흡수). 발사 메커니즘(tool의 fire 동작) 자체는 태스크
    control 코드(sim 모듈)가 담당하므로 여기엔 없다 -- 이 템플릿은
    "물체 + Stabilizer 앵커"라는 정적 구조만 제공한다.

    params 필수 키: workpiece_name, workpiece_half_size(3),
    workpiece_mass, stabilizer_grasp_relpose(7)."""
    wp = params["workpiece_name"]
    hs = " ".join(str(x) for x in params["workpiece_half_size"])
    mass = params["workpiece_mass"]
    return f"""<!-- impact-recoil 패턴: {wp}(자유 바디, 충격으로 밀리고 Stabilizer가 weld로 반동을 흡수) -->
<body name="{wp}" pos="0 0 0">
  <freejoint name="{wp}_free"/>
  <geom name="{wp}_geom" type="box" size="{hs}" mass="{mass}"/>
</body>
<!-- Stabilizer weld(relpose={params.get("stabilizer_grasp_relpose")})는
     scaffold_task.py가 Stabilizer 앵커 템플릿(sim/stabilizer.py 패턴)과
     함께 채운다 -- 이 패턴 모듈은 workpiece의 정적 구조만 책임진다. -->"""
