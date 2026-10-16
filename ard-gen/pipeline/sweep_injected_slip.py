"""작업 2 (2-1 스윕) -- `injected` 미끄러짐의 magnitude x direction x
timing(서브태스크 진행도 비율) x target arm(peg_grasp=F2/hole_grasp=F3)
그리드를 무작위 씬에서 돌려 실패율/경계-판정 분포/"미끄러짐은 났지만
성공"률을 측정한다. **결과 표가 곧 결과물**(사용자 지시) -- 그래서
세분화된 CLI 없이 아래 상수들만 고치면 다시 돌릴 수 있게 최소한으로
짰다. force_limited는 2단계에서 이미 "admittance가 실시간으로
보상해서 어떤 임계값으로도 실패를 못 만든다"로 게이트됐던 것과 같은
결론이라 여기서 다시 스윕하지 않는다(사용자 지시 범위 -- injected만).

## 타이밍: "진행도 비율"을 2-패스로 구현

`ArmSlipConfig.injected_at_step`은 절대 스텝 번호가 필요하지만, 사용자
지시는 타이밍을 "서브태스크 진행도 비율"로 표현하라고 했다. 씬마다
insert 서브태스크 길이가 다르므로(무작위 오프셋), 교란 없이 먼저
한 번 돌려(dry run) 그 씬의 insert 시작 스텝/길이를 알아낸 뒤, **같은
씬·같은 게인**으로 다시 돌리면서 그 비율에 해당하는 절대 스텝에
주입한다 -- mujoco 물리가 완전히 결정론적이라는 걸 5단계에서 이미
실측 확인했으므로, 주입 이전 구간은 두 패스가 항상 똑같다는 전제가
성립한다.

## "미끄러짐은 났지만 성공" 판정

`classify_failure()`는 `success=True`면 미끄러짐 여부와 무관하게 항상
"none"을 돌려주므로(의도된 설계, sim 모듈 docstring 참고), 에피소드가
끝난 뒤 해당 팔의 최종 `cumulative_shift_pos` 노름을 직접 읽어서
"3mm 임계값을 넘었는데도 success=True"인 경우를 따로 센다(injected는
한 번 영구히 더해질 뿐 되돌아가지 않으므로, 주입이 일어났다면 거의
항상 이 조건을 만족한다).

사용법:
    python pipeline/sweep_injected_slip.py
"""
from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from closedloop.peg_in_hole_adapter import PegInHoleClosedLoopEnv, ScriptedOraclePolicy
from pipeline.generate_closed_loop_episodes import _NOMINAL_GAINS, _base_scene, _load_graph
from sim.grasp_slip import ArmSlipConfig, GraspSlipConfig
from sim.peg_in_hole_bimanual_openarm_sim import BimanualPegInHoleOpenArmSim

_SLIP_THRESHOLD_M = 0.003  # classify_failure()의 slip_distance_threshold_m 기본값과 동일.
_MAX_ITERATIONS = 2000

_MAGNITUDES_MM = [10, 20, 30]
# 후속 작업("y방향 포함 8방향 스윕") -- 처음 스윕(PIPELINE.md "작업2"
# 절에 기록된 결과)은 ±x만 봤다. F3가 -x에서만 실패한다는 결과가
# "ee 로컬 x축이라는 우연한 축"에 특수한 건지, 아니면 일반적으로
# "hole 쪽에서 밀어내는 방향"처럼 의미가 있는 축인지 구분하려고
# ±y·대각선 4방향을 추가해 총 8방향으로 넓혔다(사용자 지시,
# 크기는 10/20/30mm로 줄임 -- 실행시간 기록).
_DIRECTIONS = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+x+y": np.array([0.70710678, 0.70710678, 0.0]),
    "+x-y": np.array([0.70710678, -0.70710678, 0.0]),
    "-x+y": np.array([-0.70710678, 0.70710678, 0.0]),
    "-x-y": np.array([-0.70710678, -0.70710678, 0.0]),
}
_TIMING_FRACTIONS = [0.3, 0.6]
_ARMS = ["right", "left"]  # right=peg_grasp(Actuator,F2), left=hole_grasp(Stabilizer,F3)
_N_SCENES = 30


def _sample_random_scene(rng: np.random.Generator) -> dict:
    """B세트와 같은 범위(사용자 지시 "무작위 씬, 고정 씬 아님") -- 고정
    씬 하나가 아니라 매 시도 다른 오프셋/마찰/clearance를 뽑는다."""
    return _base_scene(rng, (-np.pi / 4, np.pi / 4), (0.009, 0.014))


def _run_episode(scene_config: dict, gains: dict, grasp_slip_config: GraspSlipConfig | None) -> dict[str, Any]:
    """재시도/복구 없이 align -> insert 딱 한 번만 돈다(스윕은 "이 교란이
    그 자체로 뭘 만드는지"가 관심사라, 복구 로직으로 섞이면 신호가
    흐려진다). `generate_closed_loop_episodes.run_one_episode()`의 단순
    버전."""
    sim = BimanualPegInHoleOpenArmSim(grasp_slip_config=grasp_slip_config)
    # legacy_failure_rule=True로 고정 -- 이 스윕의 표 열(F2%/F3%/timeout%/
    # F1%/none%)이 처음 스윕(PIPELINE.md "작업2" 절) 때와 같은 의미를
    # 유지해야 그 결과와 직접 비교할 수 있다. 작업2에서 어댑터 기본값이
    # F1a/F1b/timeout(새 규칙)으로 바뀐 뒤 이 플래그 없이 돌리면 "F1" 라벨이
    # 더는 안 나와서 표의 F1% 열이 항상 0으로 보이는 문제가 있었다(실측으로
    # 발견, diagnose_timeout_boundaries.py와 같은 문제).
    env = PegInHoleClosedLoopEnv(sim, _load_graph(), scene_config, gains, render=False, legacy_failure_rule=True)
    env.reset_episode("align")
    policy = ScriptedOraclePolicy(env)

    boundaries: list[dict[str, Any]] = []
    total_steps = 0
    success = False
    for _ in range(_MAX_ITERATIONS):
        actions, decisions = policy.predict_action_chunk_with_decisions(None)
        env.execute(actions[0, 0])
        total_steps += 1
        if env.subtask_finished():
            env.settle()
            node = env.current_node
            start_step = total_steps - env._step_in_subtask
            _, fresh_decisions = policy.predict_action_chunk_with_decisions(None)
            boundaries.append(
                {
                    "node": node,
                    "start_step": start_step,
                    "end_step": total_steps,
                    "finished_naturally": bool(env._subtask_done_naturally),
                    "failure_type": fresh_decisions["failure_type_label"],
                }
            )
            if env._subtask_done_naturally:
                if node == "insert":
                    success = True
                    break
                env.set_subtask("insert")
            else:
                success = False
                break

    right_state, left_state = sim.get_grasp_states()
    right_shift = 0.0 if right_state is None else float(np.linalg.norm(right_state.cumulative_shift_pos))
    left_shift = 0.0 if left_state is None else float(np.linalg.norm(left_state.cumulative_shift_pos))
    return {
        "success": success,
        "boundaries": boundaries,
        "right_shift_m": right_shift,
        "left_shift_m": left_shift,
    }


def main() -> None:
    import time

    t_start = time.perf_counter()
    rng = np.random.default_rng(2024)
    scenes = [_sample_random_scene(rng) for _ in range(_N_SCENES)]

    dry_runs = []
    for scene in scenes:
        result = _run_episode(scene, dict(_NOMINAL_GAINS), None)
        insert_b = next((b for b in result["boundaries"] if b["node"] == "insert"), None)
        dry_runs.append({"scene": scene, "insert_b": insert_b})
    n_no_insert = sum(1 for d in dry_runs if d["insert_b"] is None)
    print(f"dry run: {_N_SCENES}개 씬, insert 서브태스크에 도달 못한 씬 {n_no_insert}개 (align 자체 실패, 스윕에서 제외)")

    print(
        f"\n{'arm':<6}{'mag_mm':>7}{'dir':>5}{'frac':>6}{'N':>5}  "
        f"{'fail%':>7}{'F2%':>6}{'F3%':>6}{'timeout%':>9}{'F1%':>6}{'none%':>7}  {'slip_but_ok%':>13}"
    )
    rows = []
    for arm in _ARMS:
        for mag_mm in _MAGNITUDES_MM:
            for dir_name, dir_vec in _DIRECTIONS.items():
                for frac in _TIMING_FRACTIONS:
                    verdict_counts = {"F1": 0, "F2": 0, "F3": 0, "timeout": 0, "none": 0}
                    n_valid = 0
                    n_fail = 0
                    n_slip_but_ok = 0
                    for dr in dry_runs:
                        insert_b = dr["insert_b"]
                        if insert_b is None:
                            continue
                        insert_len = insert_b["end_step"] - insert_b["start_step"]
                        inject_step = insert_b["start_step"] + int(frac * insert_len)
                        offset = dir_vec * (mag_mm / 1000.0)
                        arm_cfg = ArmSlipConfig(mode="injected", injected_at_step=inject_step, injected_offset_local=offset)
                        grasp_cfg = GraspSlipConfig(right=arm_cfg) if arm == "right" else GraspSlipConfig(left=arm_cfg)
                        result = _run_episode(dr["scene"], dict(_NOMINAL_GAINS), grasp_cfg)
                        n_valid += 1
                        label = result["boundaries"][-1]["failure_type"] if result["boundaries"] else "none"
                        verdict_counts[label] = verdict_counts.get(label, 0) + 1
                        if not result["success"]:
                            n_fail += 1
                        shift = result["right_shift_m"] if arm == "right" else result["left_shift_m"]
                        if result["success"] and shift > _SLIP_THRESHOLD_M:
                            n_slip_but_ok += 1
                    if n_valid == 0:
                        continue
                    row = {
                        "arm": arm, "mag_mm": mag_mm, "dir": dir_name, "frac": frac, "n": n_valid,
                        "fail_rate": n_fail / n_valid, "verdicts": dict(verdict_counts),
                        "slip_but_ok_rate": n_slip_but_ok / n_valid,
                    }
                    rows.append(row)
                    print(
                        f"{arm:<6}{mag_mm:>7}{dir_name:>5}{frac:>6}{n_valid:>5}  "
                        f"{row['fail_rate']*100:>6.1f}%"
                        f"{verdict_counts['F2']/n_valid*100:>5.1f}%"
                        f"{verdict_counts['F3']/n_valid*100:>5.1f}%"
                        f"{verdict_counts['timeout']/n_valid*100:>8.1f}%"
                        f"{verdict_counts['F1']/n_valid*100:>5.1f}%"
                        f"{verdict_counts['none']/n_valid*100:>6.1f}%  "
                        f"{row['slip_but_ok_rate']*100:>12.1f}%"
                    )
    elapsed = time.perf_counter() - t_start
    n_cells = len(rows)
    n_runs = sum(row["n"] for row in rows) + _N_SCENES
    print(
        f"\n실행 시간: {elapsed:.1f}s 총 / {n_cells}칸 / 에피소드 {n_runs}개 "
        f"({elapsed/max(n_runs,1):.3f}s/episode)"
    )
    return rows


if __name__ == "__main__":
    main()
