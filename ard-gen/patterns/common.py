"""RoboTwin 2.0 이식 Part 3-1: 3개 태스크 reward 공식의 공통 뼈대.

peg_in_hole/cap_twist/tacker의 compute_reward()를 나란히 놓고 보면
전부 같은 모양이다: "진행도 보상 + 거리/부족분 페널티 - 과도한
힘/토크 페널티(threshold 넘은 만큼만) - 스텝 페널티 + (조건부 flat
보너스들)". 이 함수는 그 뼈대를 공식화한 것뿐이고, 계수/필드 선택은
각 patterns/*.py가 맡는다 -- 이 모듈 자체는 어떤 태스크도 모른다.

tests/test_patterns_identity.py가 리팩토링 전/후 reward가
bit-identical한지 실측으로 확인한다(이 모듈이 "같은 공식을 다르게
썼을 뿐"이라는 주장의 증거)."""
from __future__ import annotations

from typing import Sequence


def shaped_reward(
    *,
    progress_term: float = 0.0,
    progress_coef: float = 0.0,
    shortfall_term: float = 0.0,
    shortfall_coef: float = 0.0,
    excess_terms: Sequence[tuple[float, float]] = (),
    step_count: float = 0.0,
    step_coef: float = 0.0,
    flat_bonuses: Sequence[float] = (),
) -> float:
    """reward = progress_coef*progress_term - shortfall_coef*shortfall_term
    - sum(coef*excess for excess, coef in excess_terms) - step_coef*step_count
    + sum(flat_bonuses).

    excess_terms는 이미 "임계값을 넘은 만큼"(예: max(0, max_force-5.0))을
    계산해서 넘겨야 한다 -- 이 함수는 threshold 빼기를 모르고, 그냥
    주어진 값에 계수만 곱해서 뺀다(각 패턴 모듈이 threshold 계산을
    맡는 이유: 태스크마다 "무엇의 초과분인지"가 다르기 때문)."""
    reward = progress_coef * progress_term - shortfall_coef * shortfall_term - step_coef * step_count
    for excess, coef in excess_terms:
        reward -= coef * excess
    reward += sum(flat_bonuses)
    return float(reward)
