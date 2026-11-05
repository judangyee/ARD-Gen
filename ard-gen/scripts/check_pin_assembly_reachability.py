"""pin_assembly: 데이터 생성 전에 모든 웨이포인트(핀 4개 + 스테이징
슬롯 5개, 호버+놓기 두 높이)에 대해 왼팔 IK가 실제로 수렴하는지 검사하고
실패율을 출력한다(사용자 지시 3 "도달 가능성... 생성 전에 검사하고
실패율을 보고한다"). S3의 "오류 위치"(sun이 임의 planet 위)도 몇 개
샘플로 같이 검사한다.

판정 기준: solve_ik_left()가 수렴(위치오차<1e-4) AND 관절 한계 여유가
모든 관절에서 2% 이상이면 OK -- 수렴은 하지만 여유가 2% 미달인 경우는
"주의"로 따로 집계한다(완전 실패는 아님, assets/pin_assembly_openarm.xml
모듈 docstring의 "16/18 OK, 2개는 여유 미달" 실측과 동일한 기준).

사용법:
    python scripts/check_pin_assembly_reachability.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.pin_assembly_openarm_env import (
    PIN_SITES,
    STAGE_SITES,
    DISK_HALF_HEIGHT,
    build_initial_disk_poses,
    generate_error_state,
)
from sim.pin_assembly_openarm_sim import PinAssemblyOpenArmSim

MARGIN_OK_THRESHOLD = 0.02
HOVER_DZ = 0.15
PLACE_DZ = 0.015


def _margin(sim: PinAssemblyOpenArmSim, qpos: dict[str, float]) -> float:
    ranges = sim._left_ranges
    fracs = [(qpos[n] - ranges[n][0]) / (ranges[n][1] - ranges[n][0]) for n in qpos]
    return min(min(v, 1.0 - v) for v in fracs)


def check_targets(sim: PinAssemblyOpenArmSim, targets: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, p in targets.items():
        for dz, label in ((HOVER_DZ, "hover"), (PLACE_DZ, "place")):
            tp = p + np.array([0.0, 0.0, dz])
            qpos, converged, err = sim.solve_ik_left(tp)
            margin = _margin(sim, qpos) if converged else 0.0
            status = "OK" if converged and margin >= MARGIN_OK_THRESHOLD else ("CAUTION" if converged else "FAIL")
            rows.append({"name": f"{name}/{label}", "converged": converged, "err": err, "margin": margin, "status": status})
    return rows


def main() -> None:
    sim = PinAssemblyOpenArmSim()
    static_targets = {}
    for site in list(PIN_SITES.values()) + list(STAGE_SITES.values()):
        static_targets[site] = sim.site_world_pos(site)

    rows = check_targets(sim, static_targets)

    rng = np.random.default_rng(123)
    for i in range(5):
        poses = build_initial_disk_poses(sim, "S3", rng)
        sim.reset(poses)
        wrong_pos, _ = sim.disk_world_pose("sun")
        rows += check_targets(sim, {f"S3_error_{i}": wrong_pos - np.array([0, 0, DISK_HALF_HEIGHT["sun"]])})

    print(f"{'target':<28}{'status':<10}{'converged':<12}{'err(m)':<12}{'margin':<10}")
    n_ok = n_caution = n_fail = 0
    for r in rows:
        print(f"{r['name']:<28}{r['status']:<10}{str(r['converged']):<12}{r['err']:<12.6f}{r['margin']:<10.4f}")
        if r["status"] == "OK":
            n_ok += 1
        elif r["status"] == "CAUTION":
            n_caution += 1
        else:
            n_fail += 1
    total = len(rows)
    print(f"\n[check_pin_assembly_reachability] 총 {total}개: OK {n_ok}, CAUTION(수렴하나 여유<2%) {n_caution}, FAIL(미수렴) {n_fail}")
    print(f"[check_pin_assembly_reachability] 실패율(FAIL만) {n_fail/total*100:.1f}%, CAUTION 포함 {(n_fail+n_caution)/total*100:.1f}%")


if __name__ == "__main__":
    main()
