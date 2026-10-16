"""후속 작업("실패 유형 재정의") 작업2 검증 요구사항 -- 새 유형별 기본
복구 매핑(tasks/peg_in_hole.graph.yaml의 failure_recovery)이 실제로
회복시키는지 유형당 N>=20으로 소규모 확인한다. `generate_closed_loop_
episodes.run_one_episode()`를 그대로 반복 실행해서 각 (failure_type,
recovery_applied) 조합이 몇 번 나왔고, 그 다음(attempt=2) 시도가
자연 완료(복구 성공)했는지를 센다 -- 특정 유형을 억지로 만들지 않고
A/B/C의 기존 샘플링이 자연히 내는 비율 그대로 모은다(사용자 지시
"소규모 확인").

사용법:
    python pipeline/measure_recovery_success.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pipeline.generate_closed_loop_episodes import load_disturbance_config, run_one_episode

_MIN_N_PER_TYPE = 20
_MAX_EPISODES_PER_SPLIT = 400


def main() -> None:
    rng = np.random.default_rng(99001)
    disturbance_cfg = load_disturbance_config()
    # (failure_type, recovery_applied) -> [n_total, n_recovered(=다음 attempt가 자연완료)]
    stats: dict[tuple[str, str], list[int]] = {}

    for split in ("A", "B", "C"):
        for i in range(_MAX_EPISODES_PER_SPLIT):
            ep = run_one_episode(
                rng, split, seed=i, disturbance_config=disturbance_cfg if split == "B" else None
            )
            boundaries = ep["subtask_boundaries"]
            for idx, b in enumerate(boundaries):
                if b["recovery_applied"] is None:
                    continue
                key = (b["failure_type"], b["recovery_applied"])
                stats.setdefault(key, [0, 0])
                stats[key][0] += 1
                # 바로 다음 경계가 같은 노드의 attempt+1이면 그게 복구 결과.
                if idx + 1 < len(boundaries):
                    nxt = boundaries[idx + 1]
                    if nxt["node"] == b["node"] and nxt["attempt"] == b["attempt"] + 1 and nxt["finished_naturally"]:
                        stats[key][1] += 1
            if min((v[0] for v in stats.values()), default=0) >= _MIN_N_PER_TYPE and len(stats) >= 4:
                break
        print(f"[{split}] {i + 1}개 에피소드 실행")

    print(f"\n유형별 복구 성공률 (목표 N>={_MIN_N_PER_TYPE}):")
    print(f"{'failure_type':<14}{'recovery':<10}{'N':>6}{'recovered':>11}{'rate':>8}")
    for (ftype, recovery), (n_total, n_recovered) in sorted(stats.items()):
        rate = n_recovered / n_total * 100 if n_total else 0.0
        flag = "" if n_total >= _MIN_N_PER_TYPE else "  (N<20, 참고용)"
        print(f"{ftype:<14}{recovery:<10}{n_total:>6}{n_recovered:>11}{rate:>7.1f}%{flag}")
    return stats


if __name__ == "__main__":
    main()
