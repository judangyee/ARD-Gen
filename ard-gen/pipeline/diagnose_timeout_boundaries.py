"""후속 작업("timeout 진단 -> 실패 유형 규칙 재정의") 작업1 -- timeout으로
판정된 서브태스크 경계를 모아 구간 전체 시계열로 "실제로 무슨 일이
있었는지" 유형화한다. 결과 표/그림이 곧 결과물(2-1 스윕 스크립트와 같은
성격) -- 운영 생성 경로(`generate_closed_loop_episodes.run_one_episode()`)
는 안 건드리고 진단 전용으로 따로 짰다.

## 왜 run_one_episode()를 그대로 안 쓰나

운영 경로는 에피소드당 `closed_loop.progress`(스칼라, align/insert가
서로 다른 의미로 재사용 + overshoot 때 clip됨)만 저장하고, dx/dy
컴포넌트·원시 깊이(clip 전)·벽 접촉 여부 같은 진단용 시계열은 저장하지
않는다(용량 문제, 5단계 설계 그대로). 이 스크립트는 그 시계열을 전부
메모리에 들고 분석한 뒤 버린다 -- npz에 영구히 남는 운영 스키마를 안
건드리려고 분리했다(사용자 지시 "코드 변경 최소"). 루프 자체(align/
insert + R2 복구 최대 1회)는 `generate_closed_loop_episodes.run_one_
episode()`와 동일한 구조를 그대로 복제했다(5단계/후속작업2와 같은
단순화 로컬 규칙).

사용법:
    python pipeline/diagnose_timeout_boundaries.py
"""
from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from closedloop.peg_in_hole_adapter import PegInHoleClosedLoopEnv, ScriptedOraclePolicy
from pipeline.generate_closed_loop_episodes import (
    _MAX_ATTEMPTS_PER_NODE,
    _NOMINAL_GAINS,
    _load_graph,
    load_disturbance_config,
    sample_scene_A,
    sample_scene_and_disturbance_B,
    sample_scene_and_disturbance_C,
)
from sim.peg_in_hole_bimanual_openarm_sim import BimanualPegInHoleOpenArmSim

_MAX_ITERATIONS = 2000
_TARGET_PER_SPLIT = {"A": 100, "B": 150, "C": 100}  # timeout 50개+ 모으는 데 필요한 소규모 N(실측으로 조정됨)


def _run_episode_instrumented(
    rng: np.random.Generator, split: str, seed: int, disturbance_config: dict | None = None
) -> dict[str, Any]:
    """run_one_episode()와 같은 구조(align->insert, R2 최대 1회)지만
    매 스텝 dx/dy/원시깊이(clip 전)/벽접촉/행동크기/접촉힘노름을 전부
    기록해서 돌려준다(diag.*). 진단 전용이라 렌더링은 안 한다."""
    action_noise_std = 0.0
    action_delay_steps = 0
    xy_bias = np.zeros(2)
    gains = dict(_NOMINAL_GAINS)
    disturbance: dict[str, Any] | None = None
    grasp_slip_config = None

    if split == "A":
        cfg = sample_scene_A(rng)
    elif split == "B":
        cfg, disturbance, gains, grasp_slip_config = sample_scene_and_disturbance_B(rng, disturbance_config)
        if disturbance["type"] == "action_noise":
            action_noise_std = disturbance["std_m"]
        elif disturbance["type"] == "xy_target_bias":
            xy_bias = np.array(disturbance["bias_m"])
    elif split == "C":
        cfg, disturbance, action_delay_steps = sample_scene_and_disturbance_C(rng)
    else:
        raise ValueError(f"unknown split {split!r}")

    sim = BimanualPegInHoleOpenArmSim(grasp_slip_config=grasp_slip_config)
    # legacy_failure_rule=True로 고정한다 -- 이 스크립트 자체가 "timeout
    # 라벨 뒤에 뭐가 숨어있는지" 알아내려고 만든 진단 도구라서, 어댑터의
    # 기본값이 F1a/F1b/timeout(이 스크립트의 classify_timeout()을 그대로
    # 반영한 결과물, 작업2)로 바뀐 지금은 legacy 없이 돌리면 "timeout"
    # 라벨이 이미 "거의성공"류만 남도록 좁혀져서(F1a/F1b가 더는 안
    # 섞임) 진단 표본이 왜곡된다(실측으로 발견). 이 진단 자체는 "판정
    # 순간 단일 시점이 F1/timeout을 어떻게 가르는지"를 보는 게 원래
    # 목적이었으므로 legacy가 맞다.
    env = PegInHoleClosedLoopEnv(sim, _load_graph(), cfg, gains, render=False, legacy_failure_rule=True)
    env.reset_episode("align")
    policy = ScriptedOraclePolicy(env)

    diag: dict[str, list] = {"dx": [], "dy": [], "raw_depth": [], "wall_contact": [], "action_norm": [], "force_norm": []}
    subtask_boundaries: list[dict[str, Any]] = []
    action_queue: list[np.ndarray] = []
    attempt = 1
    total_steps = 0
    success_final = False
    recovery_happened = False

    for _ in range(_MAX_ITERATIONS):
        actions, decisions = policy.predict_action_chunk_with_decisions(None)
        action = actions[0, 0].astype(np.float64).copy()
        if action_noise_std > 0:
            action = action + rng.normal(0.0, action_noise_std, size=3)
        action[:2] += xy_bias
        if action_delay_steps > 0:
            action_queue.append(action.copy())
            action = action_queue.pop(0) if len(action_queue) > action_delay_steps else np.zeros(3)

        env.execute(action)
        total_steps += 1

        dx, dy = env.current_dx_dy()
        hole_center = sim.get_hole_center_pos()
        peg_tip = sim.get_peg_tip_pos()
        raw_depth_unclipped = float(hole_center[2] - peg_tip[2])  # xy_ok로 0-clip하기 *전* 원시값(진단용)
        force, _torque = sim.get_force_torque()
        diag["dx"].append(dx)
        diag["dy"].append(dy)
        diag["raw_depth"].append(raw_depth_unclipped)
        diag["wall_contact"].append(bool(sim.is_peg_touching_hole_wall()))
        diag["action_norm"].append(float(np.linalg.norm(action)))
        diag["force_norm"].append(float(np.linalg.norm(force)))

        if env.subtask_finished():
            env.settle()
            node = env.current_node
            start_step = total_steps - env._step_in_subtask
            _, fresh_decisions = policy.predict_action_chunk_with_decisions(None)
            boundary = {
                "node": node,
                "start_step": start_step,
                "end_step": total_steps,
                "finished_naturally": bool(env._subtask_done_naturally),
                "failure_type": fresh_decisions["failure_type_label"],
                "attempt": attempt,
                "recovery_applied": None,
            }
            if env._subtask_done_naturally:
                subtask_boundaries.append(boundary)
                if node == "insert":
                    success_final = True
                    break
                env.set_subtask("insert")
                attempt = 1
            elif attempt < _MAX_ATTEMPTS_PER_NODE:
                boundary["recovery_applied"] = "R2"
                subtask_boundaries.append(boundary)
                env.run_recovery("R2")
                recovery_happened = True
                attempt += 1
            else:
                subtask_boundaries.append(boundary)
                success_final = False
                break

    return {
        "split": split,
        "scene_config": cfg,
        "disturbance": disturbance,
        "gains": gains,
        "subtask_boundaries": subtask_boundaries,
        "success": success_final,
        "loop_mode": "closed" if recovery_happened else "open",
        "outer_half": env._outer_half,
        "target_depth": env._target_depth,
        "diag": {k: np.asarray(v) for k, v in diag.items()},
    }


def _longest_stall_fraction(series: np.ndarray, eps: float) -> float:
    """series가 eps 이내로만 변하는 가장 긴 연속 구간의 길이 비율(0~1) --
    "정체 구간"(사용자 지시 "진행도 정체 구간 길이")."""
    if len(series) < 2:
        return 0.0
    best = cur = 1
    for i in range(1, len(series)):
        if abs(series[i] - series[i - 1]) <= eps:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best / len(series)


def _count_local_extrema(series: np.ndarray) -> int:
    """부호가 바뀌는 1차 차분 개수 -- "진동" 정도(사용자 지시 "진동·발산")."""
    d = np.diff(series)
    d = d[np.abs(d) > 1e-9]
    if len(d) < 2:
        return 0
    signs = np.sign(d)
    return int(np.sum(signs[1:] != signs[:-1]))


def compute_features(diag: dict[str, np.ndarray], start: int, end: int, node: str, outer_half: float, target_depth: float) -> dict[str, Any]:
    dx = diag["dx"][start:end]
    dy = diag["dy"][start:end]
    xy_err = np.hypot(dx, dy)
    raw_depth = diag["raw_depth"][start:end]
    wall = diag["wall_contact"][start:end]
    action_norm = diag["action_norm"][start:end]
    force_norm = diag["force_norm"][start:end]
    n = end - start

    key_series = xy_err if node == "align" else raw_depth
    eps = outer_half * 0.05 if node == "align" else target_depth * 0.05
    feats = {
        "node": node,
        "n_steps": n,
        "xy_err_start_mm": float(xy_err[0] * 1000),
        "xy_err_min_mm": float(xy_err.min() * 1000),
        "xy_err_end_mm": float(xy_err[-1] * 1000),
        "depth_max_mm": float(raw_depth.max() * 1000),
        "depth_end_mm": float(raw_depth[-1] * 1000),
        "wall_contact_ratio": float(wall.mean()),
        "last_contact_step_frac": float(np.nonzero(wall)[0][-1] / n) if wall.any() else -1.0,
        "action_norm_mean": float(action_norm.mean()),
        "force_norm_max": float(force_norm.max()),
        "stall_len_frac": _longest_stall_fraction(key_series, eps),
        "n_local_extrema": _count_local_extrema(key_series),
    }
    feats["oscillation_rate"] = feats["n_local_extrema"] / max(n, 1)
    return feats


def classify_timeout(feats: dict[str, Any], outer_half: float, target_depth: float) -> str:
    """실측 특징으로 timeout 사례를 유형화한다(작업1 산출물 자체 --
    실제 임계값/후보는 아래 PIPELINE.md "작업1" 절에 실측 분포와 함께
    기록, 여기 상수는 그 실측을 반영해 고른 값)."""
    outer_half_mm = outer_half * 1000
    target_depth_mm = target_depth * 1000

    if feats["node"] == "align":
        if feats["xy_err_end_mm"] <= outer_half_mm:
            return "거의성공(align, hold 직전 예산소진)"
        if feats["oscillation_rate"] > 0.2:
            return "진동_발산"
        if feats["xy_err_end_mm"] < feats["xy_err_start_mm"] * 0.7:
            return "그냥_느림"
        return "정렬_미수렴"
    else:  # insert
        if feats["depth_max_mm"] >= target_depth_mm * 0.9:
            return "거의성공(insert, hold 직전 예산소진)"
        if feats["wall_contact_ratio"] > 0.3 and feats["depth_max_mm"] < target_depth_mm * 0.8:
            return "걸림(jam)"
        if feats["oscillation_rate"] > 0.2:
            return "진동_발산"
        if feats["depth_end_mm"] > feats["depth_max_mm"] * 0.5 and feats["stall_len_frac"] < 0.5:
            return "그냥_느림"
        return "정렬_미수렴(insert 중 재정렬 실패)"


def main() -> None:
    rng = np.random.default_rng(31415)
    disturbance_cfg = load_disturbance_config()

    timeout_cases = []
    n_run = {"A": 0, "B": 0, "C": 0}
    for split, n in _TARGET_PER_SPLIT.items():
        for i in range(n):
            ep = _run_episode_instrumented(rng, split, seed=i, disturbance_config=disturbance_cfg if split == "B" else None)
            n_run[split] += 1
            for b in ep["subtask_boundaries"]:
                if b["failure_type"] != "timeout":
                    continue
                feats = compute_features(ep["diag"], b["start_step"], b["end_step"], b["node"], ep["outer_half"], ep["target_depth"])
                category = classify_timeout(feats, ep["outer_half"], ep["target_depth"])
                timeout_cases.append(
                    {
                        "split": split, "episode_seed": i, "node": b["node"],
                        "start_step": b["start_step"], "end_step": b["end_step"],
                        "disturbance_type": (ep["disturbance"] or {}).get("type"),
                        "category": category, "feats": feats,
                    }
                )
        print(f"[{split}] {n_run[split]}개 에피소드 실행, 누적 timeout 경계 {len(timeout_cases)}개")
        if len(timeout_cases) >= 50:
            break

    print(f"\n총 timeout 경계 {len(timeout_cases)}개 수집 (목표 50+)")

    # 유형별 개수
    cat_counts: dict[str, int] = {}
    for c in timeout_cases:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    print("\n유형별 개수:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<35} {cnt:>3}")

    # 서브태스크별 개수
    node_counts: dict[str, int] = {}
    for c in timeout_cases:
        node_counts[c["node"]] = node_counts.get(c["node"], 0) + 1
    print("\n서브태스크별 개수:", node_counts)

    # 교란 종류 x 유형 교차표
    print("\n교란종류 x 유형 교차표:")
    cross: dict[tuple, int] = {}
    for c in timeout_cases:
        key = (c["disturbance_type"] or f"(no-disturbance:{c['split']})", c["category"])
        cross[key] = cross.get(key, 0) + 1
    dtypes = sorted({k[0] for k in cross})
    cats = sorted({k[1] for k in cross})
    header = "disturbance".ljust(22) + "".join(c[:18].rjust(20) for c in cats)
    print(header)
    for dt in dtypes:
        row = dt.ljust(22) + "".join(str(cross.get((dt, c), 0)).rjust(20) for c in cats)
        print(row)

    return timeout_cases


if __name__ == "__main__":
    main()
