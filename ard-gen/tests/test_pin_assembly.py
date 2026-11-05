"""pin_assembly 태스크 테스트. pytest 없이 plain assert(이 저장소
관례, 직접 실행).

렌더링 없이(run_episode의 renderer=None 기본값) 빠르게 돈다 -- 이미지
캡처는 pipeline/to_lerobot_pin_assembly.py가 실제 데이터 생성 시에만
쓴다.

사용법:
    python tests/test_pin_assembly.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import sim.roco_interface as roco
from sim.pin_assembly_openarm_env import (
    DISK_HALF_HEIGHT,
    PIN_SITES,
    STAGE_SITES,
    build_initial_disk_poses,
    count_correctly_assembled,
    generate_error_state,
    is_seated,
    recover_misplaced_part,
    run_episode,
)
from sim.pin_assembly_openarm_sim import DISK_BODIES, PinAssemblyOpenArmSim


def test_s1_initial_state_all_staged() -> None:
    """S1: 5개 부품 전부 스테이징 슬롯에, 핀 위는 비어 있어야 한다."""
    sim = PinAssemblyOpenArmSim()
    rng = np.random.default_rng(0)
    poses = build_initial_disk_poses(sim, "S1", rng)
    sim.reset(poses)
    assert count_correctly_assembled(sim) == 0
    for body, site in STAGE_SITES.items():
        pos, _ = sim.disk_world_pose(body)
        target = sim.site_world_pos(site)
        assert np.linalg.norm(pos[:2] - target[:2]) < 0.01, (body, pos, target)
    print("[OK] S1 initial state: all 5 parts staged, nothing assembled")


def test_s2_initial_state_planets_seated() -> None:
    """S2: planet 3개는 이미 핀에 seated, sun/ring은 스테이징."""
    sim = PinAssemblyOpenArmSim()
    rng = np.random.default_rng(1)
    poses = build_initial_disk_poses(sim, "S2", rng)
    sim.reset(poses)
    for body in ("planet_1", "planet_2", "planet_3"):
        assert is_seated(sim, body, PIN_SITES[body]), body
    assert count_correctly_assembled(sim) == 3
    sun_pos, _ = sim.disk_world_pose("sun")
    sun_stage = sim.site_world_pos(STAGE_SITES["sun"])
    assert np.linalg.norm(sun_pos[:2] - sun_stage[:2]) < 0.01
    print("[OK] S2 initial state: 3 planets seated (n_assembled=3), sun staged")


def test_s3_error_state_physically_valid() -> None:
    """S3: sun이 무작위 planet 위에 물리적으로 유효하게(실제로 얹혀서
    안 미끄러져 떨어짐) 놓여 있어야 한다 -- 사용자 지시 4."""
    sim = PinAssemblyOpenArmSim()
    rng = np.random.default_rng(2)
    poses = build_initial_disk_poses(sim, "S3", rng)
    sim.reset(poses)

    sun_pos, _ = sim.disk_world_pose("sun")
    # sun이 어느 planet 위든, 그 xy 근처(핀 자리가 아님)여야 하고, z가
    # "두 원판 두께만큼 planet 위"에 있어야 한다 -- 핀에 꽂혀 있으면
    # 안 된다(그건 정상 상태지 오류 상태가 아님).
    best_planet = None
    best_dist = float("inf")
    for body in ("planet_1", "planet_2", "planet_3"):
        p, _ = sim.disk_world_pose(body)
        d = float(np.linalg.norm(sun_pos[:2] - p[:2]))
        if d < best_dist:
            best_dist, best_planet = d, body
    assert best_dist < 0.015, f"sun이 어느 planet 위에도 없음(최소거리 {best_dist})"
    planet_pos, _ = sim.disk_world_pose(best_planet)
    expected_z = planet_pos[2] + DISK_HALF_HEIGHT[best_planet] + DISK_HALF_HEIGHT["sun"]
    assert abs(sun_pos[2] - expected_z) < 0.01, (sun_pos[2], expected_z)
    assert not is_seated(sim, "sun", PIN_SITES["sun"])
    print(f"[OK] S3 initial state: sun physically resting on {best_planet}, not on center pin")


def test_generate_error_state_rejects_unstable() -> None:
    """generate_error_state()가 실제로 짧은 물리 시뮬레이션으로 검사하는지
    -- 여러 시드에서 반환되는 상태가 전부 물리적으로 안정(큰 drift 없음)
    해야 한다(사용자 지시 4 "폐기" 로직이 실제로 동작한다는 증거)."""
    sim = PinAssemblyOpenArmSim()
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        poses = build_initial_disk_poses(sim, "S2", rng)
        for body in ("planet_1", "planet_2", "planet_3"):
            pass
        planet_poses = {b: poses[b] for b in ("planet_1", "planet_2", "planet_3")}
        chosen, (sun_pos, sun_quat) = generate_error_state(sim, planet_poses, rng)
        planet_pos = planet_poses[chosen][0]
        xy_drift = float(np.linalg.norm(sun_pos[:2] - planet_pos[:2]))
        assert xy_drift <= 0.015, (seed, xy_drift)
        assert sun_pos[2] > planet_pos[2] + DISK_HALF_HEIGHT[chosen] + 0.003
    print("[OK] generate_error_state: 8/8 시드 모두 물리적으로 안정된 오류 상태 반환")


def test_recovery_subskill_moves_part_away() -> None:
    """복구 서브스킬(분리된 함수) 단위 테스트: 잘못 놓인 sun을 집어서
    discard 위치로 옮기면, 그 뒤 sun이 원래 잘못 놓인 자리가 아니라
    discard 위치 근처에 있어야 한다."""
    sim = PinAssemblyOpenArmSim()
    rng = np.random.default_rng(3)
    poses = build_initial_disk_poses(sim, "S3", rng)
    sim.reset(poses)

    wrong_pos, _ = sim.disk_world_pose("sun")
    discard_pos = sim.site_world_pos(STAGE_SITES["sun"]) + np.array([0, 0, DISK_HALF_HEIGHT["sun"]])

    from sim.pin_assembly_openarm_env import EpisodeRecording

    rec = EpisodeRecording()
    recover_misplaced_part(sim, "sun", wrong_pos, discard_pos, rec)

    final_pos, _ = sim.disk_world_pose("sun")
    assert np.linalg.norm(final_pos - discard_pos) < 0.012, (final_pos, discard_pos)
    assert np.linalg.norm(final_pos[:2] - wrong_pos[:2]) > 0.02
    assert all(p == "recovery" for p in rec.phase)
    print("[OK] recover_misplaced_part: sun moved from wrong position to discard slot, all frames phase=recovery")


def test_episode_success_and_labels() -> None:
    """S1 전체 에피소드 1개를 실행해서: 성공 판정, 프레임 라벨 스키마
    (phase/failure_type/loss_weight), 초반 HOLD 프레임이 실제로 거의
    안 움직이는지(사용자 지시 3-2)까지 확인."""
    rng = np.random.default_rng(0)
    rec, info = run_episode("S1", rng)
    assert info["episode_success"] is True
    assert info["n_assembled"] == 5
    assert len(rec) == info["n_frames"]
    assert all(p in ("normal", "failure", "recovery") for p in rec.phase)
    assert all(0.0 <= w <= 1.0 for w in rec.loss_weight)

    from sim.pin_assembly_openarm_env import HOLD_FRAMES

    hold_disp = [float(np.linalg.norm(rec.left_ee_pos[i] - rec.left_ee_pos[0])) for i in range(HOLD_FRAMES)]
    assert max(hold_disp) <= 0.005, f"초반 {HOLD_FRAMES}프레임 유지 중 이동이 5mm 초과: {max(hold_disp)}"
    print(f"[OK] S1 episode: success, n_frames={info['n_frames']}, hold-phase max displacement={max(hold_disp)*1000:.2f}mm")


def test_s3_frame_labels_mark_initial_failure() -> None:
    """S3 에피소드의 맨 처음(오류 상태를 그대로 들고 시작하는 구간)은
    phase="failure"/loss_weight=0이어야 하고, 복구 구간은 phase="recovery"
    로 표시돼야 한다(사용자 지시 5)."""
    from sim.pin_assembly_openarm_env import HOLD_FRAMES

    rng = np.random.default_rng(1)
    rec, info = run_episode("S3", rng)
    assert rec.phase[0] == "failure"
    assert rec.loss_weight[0] == 0.0
    assert rec.failure_type[0] == "sun_on_planet"
    assert "recovery" in rec.phase
    recovery_idx = rec.phase.index("recovery")
    assert recovery_idx >= HOLD_FRAMES
    print("[OK] S3 labels: initial frames phase=failure/loss_weight=0, recovery phase present")


def test_state_action_schema_matches_roco_interface() -> None:
    """build_action_sequence()에 OPENARM_EE_DOWNWARD_OFFSET_MAT +
    분기 선택을 넘겨서(사용자 지시 3-1) 실제 궤적에서도 왼팔 rx가 +pi
    쪽, 오른팔 rx가 -pi 쪽에서 시작하고, 연속 프레임 간 pi 이상 점프가
    없는지 확인한다."""
    rng = np.random.default_rng(0)
    rec, info = run_episode("S1", rng)

    left_pos = np.stack(rec.left_ee_pos)
    left_quat = np.stack(rec.left_ee_quat)
    right_pos = np.stack(rec.right_ee_pos)
    right_quat = np.stack(rec.right_ee_quat)
    left_grip = np.full(len(rec), 0.2)
    right_grip = np.full(len(rec), 0.2)

    action = roco.build_action_sequence(
        left_pos, left_quat, left_grip, right_pos, right_quat, right_grip,
        ee_frame_offset_mat=roco.OPENARM_EE_DOWNWARD_OFFSET_MAT,
        left_initial_rx_branch="positive_pi",
        right_initial_rx_branch="negative_pi",
    )
    assert action.shape == (len(rec), 14)
    left_rx = action[:, 3]
    right_rx = action[:, 10]
    assert left_rx[0] > 0, left_rx[0]
    assert right_rx[0] < 0, right_rx[0]
    assert np.all(np.abs(np.diff(left_rx)) < np.pi)
    assert np.all(np.abs(np.diff(right_rx)) < np.pi)
    print(
        f"[OK] action schema: shape (T,14), left rx starts +{left_rx[0]:.3f} "
        f"right rx starts {right_rx[0]:.3f}, no jumps >= pi"
    )


if __name__ == "__main__":
    test_s1_initial_state_all_staged()
    test_s2_initial_state_planets_seated()
    test_s3_error_state_physically_valid()
    test_generate_error_state_rejects_unstable()
    test_recovery_subskill_moves_part_away()
    test_episode_success_and_labels()
    test_s3_frame_labels_mark_initial_failure()
    test_state_action_schema_matches_roco_interface()
    print()
    print("ALL PIN_ASSEMBLY TESTS PASSED")
