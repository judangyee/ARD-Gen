"""ARD-VLA 폐루프 통합 데이터 스키마(5단계, PIPELINE.md "폐루프 데이터" 절
참고) 검증 -- pipeline/episode_io.py의 확장 필드(closed_loop/
subtask_boundaries/disturbance/candidate_chunks) 왕복, pipeline/
generate_closed_loop_episodes.py의 소규모 생성, pipeline/
render_closed_loop_boundaries.py의 경계 이미지 렌더링까지 전부 작게
(에피소드 2~3개) 돌려서 끝까지 동작하는지 확인한다 -- 실제 data/
closed_loop/{A,B,C}의 수십 개짜리 생성은 이 테스트가 매번 다시 하지
않는다(느림, PIPELINE.md에 한 번 실행한 결과를 기록해뒀다).

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 스크립트들과 같은 스타일.

이 파일은 (렌더링 테스트가 섞여 있어서) 맨 위에서 `MUJOCO_GL=osmesa`를
설정한다 -- PyTorch Adam을 만드는 테스트가 전혀 없어서(세그폴트 조건,
PIPELINE.md 참고) 안전하다. mujoco는 그 값을 **자기 모듈이 import되는
시점**에 한 번 읽어서 렌더 백엔드를 고정하므로(그 뒤에 바꿔도 효과
없음, 실측 확인), 이 파일의 다른 어떤 import보다도 먼저 설정해야 한다.

사용법:
    python tests/test_closed_loop_data_schema.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# mujoco는 MUJOCO_GL을 import 시점에 읽어서 렌더 백엔드를 고른다 -- 이 파일
# 아래의 다른 import(pipeline.generate_closed_loop_episodes 등)가 먼저
# mujoco를 불러오기 전에 여기서 osmesa로 정해둬야 한다(render_episode.py/
# render_closed_loop_boundaries.py와 같은 순서 규칙).
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from pipeline.episode_io import add_boundary_images, load_episode, save_episode
from pipeline.generate_closed_loop_episodes import run_one_episode, sample_scene_A
from pipeline.render_closed_loop_boundaries import render_episode_boundaries

_TMP_DIR = os.path.join(os.path.dirname(__file__), "_tmp_closed_loop_schema")


def test_episode_io_round_trip_with_new_fields() -> None:
    episode = {
        "task": "peg_in_hole",
        "right_arm": {
            "traj": np.zeros((5, 4), dtype=np.float32),
            "action": np.zeros((4, 3), dtype=np.float32),
            "gain_names": ["Kp_xy", "Kd_xy"],
            "gains": [0.1, 0.001],
            "force": np.zeros((4, 3), dtype=np.float32),
            "torque": np.zeros((4, 3), dtype=np.float32),
            "role": "actuator",
        },
        "left_arm": {"traj": np.zeros((5, 3), dtype=np.float32), "role": "stabilizer"},
        "scene_config": {"friction": 0.5},
        "success": True,
        "closed_loop": {
            "joint_16dof": np.zeros((5, 16), dtype=np.float32),
            "progress": np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32),
            "subtask_id": np.array(["align", "align", "insert", "insert"]),
            "grasp_right": {
                "weld_active": np.array([True, True, True, True]),
                "weld_force": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "slipped": np.array([False, False, False, False]),
                "cumulative_shift": np.zeros((4, 3), dtype=np.float32),
            },
        },
        "subtask_boundaries": [
            {
                "node": "align", "start_step": 0, "end_step": 10, "finished_naturally": True,
                "failure_type": "none", "attempt": 1, "recovery_applied": None,
            },
        ],
        "disturbance": {"type": "injected_slip_F2", "magnitude_m": 0.015},
        "candidate_chunks": [{"node": "align", "branches": [{"kp_mult": 1.1, "end_progress": 0.3}]}],
        "split": "train",
        "loop_mode": "closed",
    }
    os.makedirs(_TMP_DIR, exist_ok=True)
    path = os.path.join(_TMP_DIR, "round_trip.npz")
    save_episode(path, episode)
    loaded = load_episode(path)

    assert list(loaded["closed_loop"].keys()) == ["joint_16dof", "progress", "subtask_id", "grasp_right"]
    assert loaded["closed_loop"]["subtask_id"].tolist() == ["align", "align", "insert", "insert"]
    assert loaded["subtask_boundaries"] == episode["subtask_boundaries"]
    assert loaded["disturbance"] == episode["disturbance"]
    assert loaded["candidate_chunks"] == episode["candidate_chunks"]
    assert loaded["split"] == "train"
    assert loaded["loop_mode"] == "closed"
    assert loaded["boundary_images"] == {}

    add_boundary_images(path, {"boundary_step0_top_cam": np.zeros((4, 4, 3), dtype=np.uint8)})
    reloaded = load_episode(path)
    assert list(reloaded["boundary_images"].keys()) == ["boundary_step0_top_cam"]
    print("[OK] episode_io round trip with closed_loop/subtask_boundaries/disturbance/candidate_chunks/boundary_images")


def test_old_style_episode_unaffected() -> None:
    """새 필드가 전혀 없는 옛 에피소드는 기존과 똑같이 저장/복원돼야
    한다("기존 키는 유지" 사용자 지시)."""
    episode = {
        "task": "peg_in_hole",
        "right_arm": {
            "traj": np.zeros((5, 4), dtype=np.float32),
            "action": np.zeros((4, 3), dtype=np.float32),
            "gain_names": ["Kp_xy", "Kd_xy"],
            "gains": [0.1, 0.001],
            "force": np.zeros((4, 3), dtype=np.float32),
            "torque": np.zeros((4, 3), dtype=np.float32),
            "role": "actuator",
        },
        "scene_config": {"friction": 0.5},
        "success": True,
        "insertion_depth": 0.02,
        "force_max": 5.0,
    }
    path = os.path.join(_TMP_DIR, "old_style.npz")
    save_episode(path, episode)
    loaded = load_episode(path)
    assert "closed_loop" not in loaded
    assert "subtask_boundaries" not in loaded
    assert "disturbance" not in loaded
    assert "candidate_chunks" not in loaded
    assert loaded["boundary_images"] == {}
    print("[OK] old-style episode (no closed_loop fields) round-trips unaffected")


def test_generate_small_episode_each_split() -> None:
    """generate_closed_loop_episodes.run_one_episode()가 A/B/C 전부에서
    스키마가 맞는 에피소드를 만드는지(에피소드 2개씩, 빠르게)."""
    for split in ("A", "B", "C"):
        rng = np.random.default_rng(12345)
        for i in range(2):
            episode = run_one_episode(rng, split, seed=i)
            assert episode["task"] == "peg_in_hole"
            assert episode["closed_loop"]["joint_16dof"].shape[1] == 16
            assert episode["closed_loop"]["joint_16dof"].shape[0] == episode["right_arm"]["action"].shape[0] + 1
            assert len(episode["subtask_boundaries"]) >= 1
            assert episode["split"] in ("train", "val", "test_B", "test_C", "test_A")
            if split == "A":
                assert episode["split"] == "test_A"
                assert episode.get("disturbance") is None
            if split == "B":
                assert episode["split"] in ("train", "val")
                assert episode["disturbance"]["type"] in (
                    "action_noise", "xy_target_bias", "bad_gains", "narrow_clearance",
                    "high_friction", "injected_slip_F2", "injected_slip_F3",
                )
            if split == "C":
                assert episode["split"] == "test_C"
                assert episode["disturbance"]["type"] in ("wide_friction", "very_narrow_clearance", "action_delay")
            path = os.path.join(_TMP_DIR, f"gen_{split}_{i}.npz")
            save_episode(path, episode)
            reloaded = load_episode(path)
            assert reloaded["success"] == episode["success"]
        print(f"[OK] generate_closed_loop_episodes: split={split} schema sane (2 episodes)")


def test_sample_scene_A_matches_calibrated_failure_range() -> None:
    """PIPELINE.md "5단계" 절에 실측 기록한 A세트 범위(반경 0.009-0.020,
    각도 전체 원)가 코드에도 그대로인지 -- 범위가 조용히 바뀌면 10-30%
    실패율 보증이 깨진다."""
    rng = np.random.default_rng(0)
    radii = []
    for _ in range(200):
        cfg = sample_scene_A(rng)
        ox, oy = cfg["peg_init_offset_xy"]
        radii.append(float(np.hypot(ox, oy)))
    assert min(radii) >= 0.0089 and max(radii) <= 0.0201
    print(f"[OK] sample_scene_A radius range [{min(radii)*1000:.2f}, {max(radii)*1000:.2f}]mm matches calibrated A range")


def test_render_boundaries_adds_images_and_skips_closed_loop() -> None:
    import mujoco  # noqa: F401 -- osmesa 렌더러가 이 환경에 있는지 먼저 확인(없으면 스킵).

    rng = np.random.default_rng(7)
    # 재시도 없이 바로 성공하는(open) 에피소드가 나올 때까지 몇 번 시도한다
    # (A세트가 압도적으로 open+success이므로 몇 번이면 충분).
    episode = None
    for i in range(10):
        candidate = run_one_episode(rng, "A", seed=i)
        if candidate["loop_mode"] == "open":
            episode = candidate
            break
    assert episode is not None, "open-loop 에피소드를 못 뽑았다(재시도 늘려야 할 수도)"
    path = os.path.join(_TMP_DIR, "render_open.npz")
    save_episode(path, episode)

    n_images = render_episode_boundaries(path)
    assert n_images > 0, "open-loop 에피소드인데 이미지가 하나도 안 캡처됐다"
    reloaded = load_episode(path)
    assert len(reloaded["boundary_images"]) == n_images
    for img in reloaded["boundary_images"].values():
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8
    print(f"[OK] render_closed_loop_boundaries: {n_images} images added for an open-loop episode")

    # closed-loop(복구 발생) 에피소드는 skip돼야 한다.
    closed_episode = None
    for i in range(10, 30):
        candidate = run_one_episode(rng, "B", seed=i)
        if candidate["loop_mode"] == "closed":
            closed_episode = candidate
            break
    assert closed_episode is not None, "closed-loop 에피소드를 못 뽑았다"
    closed_path = os.path.join(_TMP_DIR, "render_closed.npz")
    save_episode(closed_path, closed_episode)
    n_skipped = render_episode_boundaries(closed_path)
    assert n_skipped == 0, "closed-loop 에피소드는 렌더링을 skip해야 한다(모듈 docstring의 한계 참고)"
    print("[OK] render_closed_loop_boundaries correctly skips a closed-loop (recovered) episode")


if __name__ == "__main__":
    test_episode_io_round_trip_with_new_fields()
    test_old_style_episode_unaffected()
    test_generate_small_episode_each_split()
    test_sample_scene_A_matches_calibrated_failure_range()
    test_render_boundaries_adds_images_and_skips_closed_loop()
    print()
    print("ALL CLOSED LOOP DATA SCHEMA TESTS PASSED")
